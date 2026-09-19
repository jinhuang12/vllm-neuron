# SPDX-License-Identifier: Apache-2.0
"""One captured graph serves every position, whatever position it was captured at.

THE DECLARED ACCEPTANCE, Tier N, CPU mode:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_capture_shape_identity_117c.py \\
      -s -rA -p no:randomly -p no:cacheprovider

Twelve items, one test each, no ``parametrize`` and no skip.

WHAT THIS FILE IS ABOUT. A graph is captured once and replayed at every position, so
anything whose SHAPE or whose CONTROL FLOW comes from a position value pins the graph
to the position it was captured at. Two such things existed: the MLA layer read its
cache as ``[: start + tokens]``, whose length grows with every decode step, and the KDA
layer chose its entering state with a python branch on the same value.

ITEMS 9 TO 12 ARE THE RUNNER'S SIDE OF THE SAME RULE, on the four things a layer can
no longer do for itself: the ALLOCATION, which grows no bank past the blocks the
scheduler hands out, the write bound that moved out of the layer once the position
became a tensor, the indexer's sequence bound, which stays a python int and
therefore had to stop moving, and the two ring positions, which stopped being python
ints and are now built here.

WHAT THESE ITEMS OBSERVE, AND WHAT THEY DO NOT. Items 1 to 4, 7 to 11 are
BEHAVIOURAL: they drive the runner's own carrier builder -- items 7, 8 and 11 through the
converter that sizes the window, one per leg -- and read what it hands a layer. Items 5 and 6 are
STRUCTURAL, in the form this test suite already uses for the state hook
(``test_kda_runner_state.py`` B01 and B02): they read the source of the two methods and
assert the position no longer reaches a python int there. Nothing here captures a real
dynamo graph -- that reading is a host-side end-to-end one on the serving run, and this
file must not be read as making it.

EVERY ITEM FAILS AT THE BASE. Items 1 and 2 because the base hands a slice of the bank
whose length moves with the position, and no block table beside it; item 3 because the
base refuses a request whose blocks are not one ascending run and refuses a bank that
carries no headroom; item 4 because the base hands a python int; items 5 and 6
because the base's source carries the two host reads this change removed; item 7
because the base sizes the window from the request's own blocks and an earlier draft
sized it from the block table's width, and the item names both wrong answers; item 8
because the base's decode window is this step's own two blocks where the bucket is ten;
item 9 because the base grows every latent bank by one window of headroom it no longer
has a reader for; item 10 because the base builds the carrier without a word and leaves
the write to a layer whose own bound is a window it no longer receives; item 11 because the base's bound is
this step's end position, which is a different number at each of the two steps; item 12
because the base builds both ring positions from a python int, so neither reaches a ring
seam as a tensor and the builder's own source spells an int cast and a sum where the
tensor helper is required.

WHAT ITEM 8 DOES NOT SEPARATE. On the decode leg the bucket's span and the table's width
are the SAME number, so item 8 fails at the base and passes for either way of sizing the
window. Item 7 is the one that separates the two, and it is on the prefill leg because
that is the leg where the two answers differ.

WHY ITEM 8 OPENS A SEQUENCE FIRST. The converter refuses any step that does not continue
the live indexer ring, so a decode dropped onto a fresh shell raises on the cursor before
a window is built. An earlier draft of this file did exactly that: item 8 failed on every
tree, base and candidate alike, for the cursor and not for the window. It now runs a
prefill at 0 through the same converter, reads the cursor that prefill left, and carries
the position it names.
"""

from __future__ import annotations

import ast
import inspect
import io
import os
import textwrap
import tokenize
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# --------------------------------------------------------------------------- #
# The declared geometry. Every number here is this file's own fixture, not a
# registered dial: the block's subject is the RELATION between the window's length
# and the position, and that relation has to hold at any legal geometry.
# --------------------------------------------------------------------------- #
#: Slots per block in the bank and in the group's page.
DECLARED_PAGE_SIZE = 8
#: The latent width of the fixture's bank. Nothing under test reads it -- the carrier
#: builder slices rows and never inspects the width -- so it is this file's own number
#: and deliberately not the checkpoint's, which would suggest a dial was registered here.
DECLARED_HEAD_SIZE = 16
#: The bucket's block-table width, which is the width every request's block table is
#: padded to and the window those blocks make.
DECLARED_WINDOW_BLOCKS = 4
#: Blocks the bank holds. More than any request here is given, so a table that named the
#: bank's own base rather than the request's pages would be visible.
DECLARED_BANK_BLOCKS = 12
#: How many sequences the modelled engine admits at once, which is the axis the
#: converter's per-sequence caches carry. One: this file drives one request's shapes.
DECLARED_MAX_NUM_SEQS = 1
#: The id of that one request. The converter keys a sequence's state by its id and refuses a
#: real step served without one, so a shell with no id can only ever be served the opening
#: bucket: no item here reads that classification, and three of them read a position past it.
DECLARED_REQUEST = "capture-shape-request"
#: The first block this request is given, deliberately not block 0, so a window that
#: started at the bank's own base rather than the request's would be visible.
DECLARED_FIRST_BLOCK = 2
#: This request's slot in the per-sequence caches. RE-PINNED: the state slot is now a
#: request slot on an axis as wide as the engine's bound, so it is no longer a block id
#: and no longer moves with the pages a request is given. The original reading, verbatim:
#: "state_slot": DECLARED_FIRST_BLOCK -- the first block the request holds.
DECLARED_STATE_SLOT = 0
#: The two positions items 1 and 2 compare. They differ by enough to change how many
#: BLOCKS the request occupies, which is what moved the base's window length.
DECLARED_LOW_POSITION = 3
DECLARED_HIGH_POSITION = 25
DECLARED_TOKENS = 1
#: The carrier keys a sparse layer is handed, as the landed builder writes them.
DECLARED_SPARSE_CARRIER_KEYS = {
    "latent_cache",
    "block_table_row",
    "latent_slots",
    "pool_cache",
    "seq_lens",
    "start_position",
    "softmax_scale",
    "max_seq_len",
    "page_size",
    "tail",
    "position",
}


#: Item 7's own numbers. The table is deliberately far wider than the span, and the
#: request occupies far less than either, so the three candidate answers -- the bucket's
#: span, the table's width and the request's own blocks -- are three different numbers.
DECLARED_TABLE_WIDTH = 10
DECLARED_SEGMENT = 16
DECLARED_PREFILL_TOKENS = 8
DECLARED_DECODE_THRESHOLD = 1

#: The position a prefill of ``DECLARED_PREFILL_TOKENS`` tokens leaves the live indexer ring
#: standing at, which is the only position a following decode step may carry: the converter
#: advances its cursor to ``start_position + tokens`` and refuses a step that does not
#: continue it (``neuron_model_runner.py:5366-5373``, ``:5402``). Derived, so the two cannot
#: drift apart.
DECLARED_CONTINUED_POSITION = DECLARED_PREFILL_TOKENS


def say(name: str, *values) -> None:
    """One reading per line, tagged so a launcher can anchor on it."""
    print("CAPSHAPE|" + name + "|" + "|".join(str(value) for value in values))


_PROSE_TOKENS = {tokenize.COMMENT, tokenize.STRING} | {
    kind for kind in (getattr(tokenize, name, None)
                      for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"))
    if kind is not None
}


def _code_of(obj) -> str:
    """``obj``'s source with every comment and string constant blanked out.

    A SOURCE SCAN IS ABOUT CODE. A docstring that names the expression the code no
    longer uses, to say what changed and why, is not the code using it -- and a scan
    that cannot tell the two apart fails a file for explaining itself. Blanking rather
    than deleting keeps every remaining character on its own line and column, so an
    exact match still reads as one. An f-string's ``{...}`` stays, because that part
    of it really is code.
    """
    source = textwrap.dedent(inspect.getsource(obj))
    rows = [list(row) for row in source.splitlines(keepends=True)]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _PROSE_TOKENS:
            continue
        (first_row, first_col), (last_row, last_col) = token.start, token.end
        for number in range(first_row, last_row + 1):
            row = rows[number - 1]
            start = first_col if number == first_row else 0
            stop = last_col if number == last_row else len(row)
            for column in range(start, min(stop, len(row))):
                if row[column] != "\n":
                    row[column] = " "
    return "".join("".join(row) for row in rows)


def _require_cpu_mode() -> None:
    """The declared acceptance runs under VLLM_NEURON_CPU_MODE=1, so read it, not set it."""
    assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1", (
        "the declared acceptance runs under VLLM_NEURON_CPU_MODE=1 and this process "
        "does not carry it, so nothing below would be measuring the declared mode"
    )


def _bank(head_size: int, *, blocks: int = DECLARED_BANK_BLOCKS, device="cpu") -> dict:
    """One sparse bank, carrying only the four keys the carrier builder reads.

    Built by hand for the reason ``test_kda_runner_state.py`` gives for its own banks: a
    builder that started reading a fifth key would raise here rather than quietly find a
    stand-in value.
    """
    return {
        "name": "model.layers.0.self_attn",
        "family": "self_attn",
        "block_size": DECLARED_PAGE_SIZE,
        "latent_cache": torch.zeros(
            (blocks * DECLARED_PAGE_SIZE, 1, head_size),
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _geometry(*, position: int, window_blocks: int = DECLARED_WINDOW_BLOCKS,
              tokens: int = DECLARED_TOKENS) -> dict:
    """This request's pages at ``position``, plus the block table's width.

    The blocks are one ascending run because that is the simplest thing to write, not
    because the builder requires it any more; item 3 hands it a scattered row. The run is
    exactly as long as the request needs at this position -- the number the base's window
    length followed, and a number nothing downstream reads now.
    """
    used = max(1, -(-(position + tokens) // DECLARED_PAGE_SIZE))
    return {
        "block_ids": [DECLARED_FIRST_BLOCK + offset for offset in range(used)],
        "state_slot": DECLARED_STATE_SLOT,
        "page_size": DECLARED_PAGE_SIZE,
        "window_blocks": int(window_blocks),
    }


def _carrier(bank: dict, text_config, *, position: int, geometry: dict,
             tokens: int = DECLARED_TOKENS, is_prefill: bool = False,
             max_seq_len: int | None = None) -> dict:
    """The one carrier the builder hands this bank's layer at ``position``.

    The leg, the token count and the bound are keywords because item 12 drives BOTH legs
    and needs the bound to stand still across two positions. Every other item drives one
    decode token at the bound the base derived, which is this signature's default.
    """
    side = NeuronModelRunner._glm5next_side_caches(
        [bank],
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE,
        request_slots=DECLARED_MAX_NUM_SEQS,
    )
    carriers = NeuronModelRunner._glm5next_layer_carriers(
        [bank],
        side,
        geometries=[geometry],
        is_prefill=is_prefill,
        tokens=tokens,
        start_position=position,
        softmax_scale=1.0,
        max_seq_len=position + tokens if max_seq_len is None else int(max_seq_len),
        index_kpool=int(text_config.index_kpool),
    )
    assert len(carriers) == 1, f"one bank was handed {len(carriers)} carrier(s)"
    return carriers[0]


def _world():
    """A text config and one sparse bank, the pair every behavioural item drives."""
    text_config = Glm5NextTextConfig()
    return text_config, _bank(DECLARED_HEAD_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 1. Every operand has ONE shape at two positions.
# ══════════════════════════════════════════════════════════════════════════════
def test_every_operand_the_layer_is_handed_has_one_shape_at_two_positions() -> None:
    """The defect, read directly: no shape the layer sees may move with the position.

    The two positions are chosen so the request occupies a DIFFERENT number of blocks at
    each -- that difference is exactly what the base's window length followed, and it is
    what a captured graph cannot survive. Three operands carry the risk now: the bank,
    whose length is the bank's; the block table, whose width is the bucket's; and the
    slots, one row per token. The tables at the two positions must DIFFER in content, or
    this item would pass on a builder that named the same pages for every request.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    head_size = int(bank["latent_cache"].shape[2])

    low = _carrier(bank, text_config, position=DECLARED_LOW_POSITION,
                   geometry=_geometry(position=DECLARED_LOW_POSITION))
    high = _carrier(bank, text_config, position=DECLARED_HIGH_POSITION,
                    geometry=_geometry(position=DECLARED_HIGH_POSITION))

    low_blocks = len(_geometry(position=DECLARED_LOW_POSITION)["block_ids"])
    high_blocks = len(_geometry(position=DECLARED_HIGH_POSITION)["block_ids"])
    say("I1_BLOCKS_THE_REQUEST_OCCUPIES", f"low={low_blocks}", f"high={high_blocks}")
    assert low_blocks != high_blocks, (
        "the two positions occupy the same number of blocks, so this item could not "
        "tell a constant window from a request-shaped one"
    )

    say("I1_BANK_SHAPES", tuple(low["latent_cache"].shape),
        tuple(high["latent_cache"].shape))
    assert tuple(low["latent_cache"].shape) == tuple(high["latent_cache"].shape)
    assert tuple(low["latent_cache"].shape) == (
        DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE,
        1,
        head_size,
    )

    say("I1_TABLE_SHAPES", tuple(low["block_table_row"].shape),
        tuple(high["block_table_row"].shape))
    assert tuple(low["block_table_row"].shape) == (DECLARED_WINDOW_BLOCKS, 1)
    assert tuple(high["block_table_row"].shape) == (DECLARED_WINDOW_BLOCKS, 1)
    say("I1_SLOT_SHAPES", tuple(low["latent_slots"].shape),
        tuple(high["latent_slots"].shape))
    assert tuple(low["latent_slots"].shape) == (DECLARED_TOKENS,)
    assert tuple(high["latent_slots"].shape) == (DECLARED_TOKENS,)

    # THE TABLES DIFFER, which is what makes the three shape readings above a reading
    # about a constant SHAPE rather than about a constant operand.
    say("I1_TABLES_DIFFER", low["block_table_row"].flatten().tolist(),
        high["block_table_row"].flatten().tolist())
    assert not torch.equal(low["block_table_row"], high["block_table_row"])


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 2. The bank travels whole, and the table names the request's own pages.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_bank_travels_whole_and_the_table_names_the_requests_pages() -> None:
    """Three conjuncts: the bank is the bank, it is a VIEW, and the table is the request's.

    The bank alone says nothing about whose rows are read -- every request is handed the
    same tensor. What makes the rows this request's is the table: its first entries are
    the blocks the request holds, in order, and the rest is the padding the bucket's
    width needs. A table naming the bank's own base instead would read another
    sequence's rows, so the entries are read one by one rather than counted.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    geometry = _geometry(position=DECLARED_LOW_POSITION)
    carrier = _carrier(bank, text_config, position=DECLARED_LOW_POSITION,
                       geometry=geometry)

    slots = DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE
    say("I2_BANK_SLOTS", int(carrier["latent_cache"].shape[0]), f"want={slots}")
    assert int(carrier["latent_cache"].shape[0]) == slots
    assert len(geometry["block_ids"]) < DECLARED_WINDOW_BLOCKS, (
        "this request already occupies the whole table, so the item could not tell "
        "the request's own entries from the padding"
    )

    # NO COPY. A copy of the bank per layer per step is the cost the paging removes, and
    # a copy would also leave the layer writing into a tensor the next step never reads.
    say("I2_THE_CARRIER_IS_A_VIEW_OF_THE_BANK",
        carrier["latent_cache"].data_ptr() == bank["latent_cache"].data_ptr())
    assert carrier["latent_cache"].data_ptr() == bank["latent_cache"].data_ptr()

    row = carrier["block_table_row"]
    want = geometry["block_ids"] + [-1] * (DECLARED_WINDOW_BLOCKS - len(geometry["block_ids"]))
    say("I2_TABLE_ROW", row.flatten().tolist(), f"want={want}", row.dtype)
    assert row.dtype == torch.int32
    assert row.flatten().tolist() == want, (
        "the table does not name this request's own pages in order, so the rows the "
        "kernel gathers are not this request's"
    )

    # THE SLOTS ARE PHYSICAL ROWS OF THE BANK, derived from the very blocks above.
    first = geometry["block_ids"][0] * DECLARED_PAGE_SIZE + DECLARED_LOW_POSITION
    say("I2_LATENT_SLOTS", carrier["latent_slots"].tolist(), f"first={first}")
    assert carrier["latent_slots"].tolist() == [first]


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 3. SCATTERED pages are carried, and a bank with no headroom is carried.
# ══════════════════════════════════════════════════════════════════════════════
def test_scattered_pages_are_carried_and_a_bank_with_no_headroom_is_carried() -> None:
    """The two refusals this change removes, stated as what is now ACCEPTED.

    THE RUNNER USED TO REFUSE BOTH. A request whose blocks were not one ascending run
    could not be expressed, because the carrier was one contiguous slice of the bank; and
    a bank with no blocks past the last one the scheduler can hand out was refused,
    because that slice ran off the end. The layer reads the pages its table names now, so
    neither case is a refusal and both are read here directly.

    ACCEPTANCE IS NOT BLANKET, which is the falsifier this item needs: the same call with
    more blocks than the bucket's table holds still refuses by name, so what is being read
    is the removal of two particular refusals rather than a builder that stopped checking.
    """
    _require_cpu_mode()
    text_config, bank = _world()

    # ARM 1: pages in no ascending order at all, one of them behind the first.
    scattered = [DECLARED_FIRST_BLOCK + 5, DECLARED_FIRST_BLOCK, DECLARED_FIRST_BLOCK + 2]
    steps = [later - earlier for earlier, later in zip(scattered, scattered[1:])]
    say("I3_CONTROL_THE_PAGES_ARE_NOT_ONE_RUN", scattered, f"steps={steps}")
    assert any(step != 1 for step in steps), (
        "these pages are one ascending run, so the arm is not reading the scattered case"
    )
    geometry = {
        "block_ids": scattered,
        "state_slot": DECLARED_STATE_SLOT,
        "page_size": DECLARED_PAGE_SIZE,
        "window_blocks": DECLARED_WINDOW_BLOCKS,
    }
    carrier = _carrier(bank, text_config, position=DECLARED_LOW_POSITION,
                       geometry=geometry)
    want = scattered + [-1] * (DECLARED_WINDOW_BLOCKS - len(scattered))
    say("I3_SCATTERED_TABLE", carrier["block_table_row"].flatten().tolist(),
        f"want={want}")
    assert carrier["block_table_row"].flatten().tolist() == want
    first = scattered[0] * DECLARED_PAGE_SIZE + DECLARED_LOW_POSITION
    say("I3_SCATTERED_SLOTS", carrier["latent_slots"].tolist(), f"first={first}")
    assert carrier["latent_slots"].tolist() == [first], (
        "the slot is not in the page the table names first, so the write and the "
        "gather disagree about where this token's row lives"
    )

    # ARM 2: a bank holding exactly the blocks a request can be given and not one more.
    # The window those blocks make runs past its end, and nothing reads past them now.
    tight_blocks = DECLARED_FIRST_BLOCK + 1
    tight = _bank(DECLARED_HEAD_SIZE, blocks=tight_blocks)
    say("I3_CONTROL_THE_BANK_IS_SHORTER_THAN_THE_WINDOW",
        tight_blocks * DECLARED_PAGE_SIZE,
        f"window={DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE}")
    assert tight_blocks * DECLARED_PAGE_SIZE < DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    tight_carrier = _carrier(tight, text_config, position=DECLARED_LOW_POSITION,
                             geometry=_geometry(position=DECLARED_LOW_POSITION))
    say("I3_TIGHT_BANK_SLOTS", int(tight_carrier["latent_cache"].shape[0]))
    assert int(tight_carrier["latent_cache"].shape[0]) == tight_blocks * DECLARED_PAGE_SIZE

    # THE FALSIFIER: more blocks than the bucket's table holds is still a refusal.
    too_many = {
        "block_ids": [DECLARED_FIRST_BLOCK + offset
                      for offset in range(DECLARED_WINDOW_BLOCKS + 1)],
        "state_slot": DECLARED_STATE_SLOT,
        "page_size": DECLARED_PAGE_SIZE,
        "window_blocks": DECLARED_WINDOW_BLOCKS,
    }
    with pytest.raises(ValueError) as caught:
        _carrier(bank, text_config, position=DECLARED_LOW_POSITION, geometry=too_many)
    message = " ".join(str(caught.value).split())
    say("I3_MORE_BLOCKS_THAN_THE_BUCKET", message[:190])
    assert "block table holds" in message


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 4. The position reaches the layer as a TENSOR.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_position_reaches_the_layer_as_a_tensor() -> None:
    """The traced boundary takes a tensor; a python int there is baked into the graph.

    The value is read back host-side, which is legitimate outside a traced region and is
    how this item tells a tensor carrying the right position from a tensor carrying any
    position at all.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    carrier = _carrier(bank, text_config, position=DECLARED_HIGH_POSITION,
                       geometry=_geometry(position=DECLARED_HIGH_POSITION))

    assert set(carrier) == DECLARED_SPARSE_CARRIER_KEYS, (
        f"a sparse carrier holds {sorted(carrier)}, not "
        f"{sorted(DECLARED_SPARSE_CARRIER_KEYS)}"
    )
    position = carrier["start_position"]
    say("I4_START_POSITION", type(position).__name__,
        getattr(position, "dtype", None), int(position))
    assert torch.is_tensor(position), (
        "the position reaches the layer as a python int, which a captured graph turns "
        "into the constant it was captured with"
    )
    assert int(position) == DECLARED_HIGH_POSITION


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 5. The MLA read no longer depends on the position (STRUCTURAL).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_mla_read_no_longer_depends_on_the_position() -> None:
    """``attend()``'s own source: the read is the whole window and no host read remains.

    A structural reading, in the form this suite uses for the state hook. It observes
    the three lines that changed; it does not observe an attention output. It reads the
    CODE only: this method's own docstring names the old expression to say what changed,
    and prose that explains a form is not the code using it.
    """
    _require_cpu_mode()
    source = _code_of(model_fp8.Glm5NextMLAAttention.attend)

    say("I5_HOST_READS_OF_THE_POSITION", source.count("int(start_position)"))
    assert source.count("int(start_position)") == 0

    say("I5_READS_THE_WHOLE_WINDOW", "c_kv = latent_cache[:, 0, :]" in source)
    assert "c_kv = latent_cache[:, 0, :]" in source
    assert "[: start + tokens" not in source, (
        "the read is still cut at the position, so its length still moves with every "
        "decode step"
    )

    say("I5_WRITES_BY_INDEX", "index_copy_" in source)
    assert "index_copy_" in source

    say("I5_BUILDS_THE_START_AS_A_DEVICE_SCALAR", "_int64_scalar(start_position" in source)
    assert "_int64_scalar(start_position" in source
    assert "torch.as_tensor(" not in source and "torch.tensor(" not in source, (
        "the start is built from python data, which a meta trace keeps real"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 6. The KDA entering state is chosen without reading the position (STRUCTURAL).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_kda_entering_state_is_chosen_without_reading_the_position() -> None:
    """The branch became a device choice, and the helper that makes it returns a tensor.

    Two conjuncts, one structural and one a value reading: the forward carries no host
    read of the position and selects with ``torch.where``, and the helper it selects
    with answers in a 0-d bool tensor rather than a python bool.
    """
    _require_cpu_mode()
    source = _code_of(model_fp8.Glm5NextKDAAttention.forward)

    say("I6_HOST_READS_OF_THE_POSITION", source.count("int(start_position)"))
    assert source.count("int(start_position)") == 0
    say("I6_CHOOSES_ON_DEVICE", "torch.where(" in source, "_start_is_zero(" in source)
    assert "torch.where(" in source
    assert "_start_is_zero(" in source

    device = torch.device("cpu")
    at_zero = model_fp8._start_is_zero(torch.tensor(0, dtype=torch.int32), device)
    above = model_fp8._start_is_zero(torch.tensor(DECLARED_HIGH_POSITION), device)
    say("I6_HELPER", type(at_zero).__name__, at_zero.dtype, at_zero.dim(),
        bool(at_zero), bool(above))
    assert torch.is_tensor(at_zero) and torch.is_tensor(above)
    assert at_zero.dtype == torch.bool and at_zero.dim() == 0
    assert bool(at_zero) is True
    assert bool(above) is False


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 7. The converter sizes the window from the LEG's bucket, not the table's width.
# ══════════════════════════════════════════════════════════════════════════════
def _runner(text_config, banks) -> NeuronModelRunner:
    """A runner carrying only what the converter reads, built the landed way.

    ``__new__`` keeps every other attribute absent, so a converter that started reading
    something new raises here instead of quietly finding a stand-in value.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(text_config=text_config, glm5next_layer_banks=banks)
    runner.max_model_len = DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE
    # RE-PINNED: the converter now sizes its per-sequence caches by the engine's
    # concurrent-sequence bound, so a runner shell must model that bound too. One
    # sequence is what this file drives.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    # RE-PINNED AGAIN, and the bound above was only half of it: the converter also keys each
    # sequence's state by its request id, and a step served without one is the opening
    # bucket's. Every step this file drives belongs to the one request named above.
    runner.input_batch = SimpleNamespace(req_ids=[DECLARED_REQUEST])
    # The context-parallel width the block-table arithmetic divides by. One is the
    # single-rank case, which is what a shell with no parallel world can honestly say.
    runner._dcp_size = 1
    return runner


def _prefill_metadata(banks) -> dict:
    """One entry per layer, in the runner's own key set, on the PREFILL leg at position 0.

    The block table is WIDER than the bucket's span on purpose: its width is what the
    prefill leg's ``max_blocks_per_seq`` falls back to when a group has no context
    bucket, and reading the window off that width is the defect this item closes.

    THE HOST HALF AND THE DEVICE HALF CARRY THE SAME NUMBERS, which is what the runner's
    own builders do (``neuron_model_runner.py:4246-4249``, ``:4437-4445``). The converter
    reads the host half, because a step's geometry is decided before the traced call and a
    device tensor holds no readable value inside a capture. Both halves are written here so
    the entry is the runner's whole key set and not the subset one function happens to
    read; an item that wants to show WHICH half was read is
    ``test_host_geometry_119.py``'s, and this file does not repeat it.
    """
    table = torch.tensor(
        [[DECLARED_FIRST_BLOCK + offset for offset in range(DECLARED_TABLE_WIDTH)]],
        dtype=torch.int32,
    )
    entry = {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": torch.arange(DECLARED_PREFILL_TOKENS, dtype=torch.int32),
        "max_query_len": DECLARED_PREFILL_TOKENS,
        "block_size": DECLARED_PAGE_SIZE,
        "max_blocks_per_seq": DECLARED_TABLE_WIDTH,
        "decode_token_threshold": DECLARED_DECODE_THRESHOLD,
        "cached_seq_len": torch.tensor([[0]], dtype=torch.int32),
        "host_block_table": table.clone(),
        "host_num_computed_tokens": torch.zeros(1, dtype=torch.int32),
        "kv_segment_size": DECLARED_SEGMENT,
    }
    return {bank["name"]: dict(entry) for bank in banks}


def _converter_kwargs(banks, metadata: dict, tokens: int) -> dict:
    """The generic mapping a call site hands the converter, at this step's token count.

    The keys the converter drops are present because a real call site carries them, so
    an item reads the same translation a serve does.
    """
    return {
        "input_ids": torch.zeros(tokens, dtype=torch.long),
        "positions": torch.arange(tokens, dtype=torch.long),
        "attn_metadata": metadata,
        "sampling_positions": torch.tensor([tokens - 1], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }


def _blocks_for_decode() -> int:
    """The blocks this item's decode step occupies, which is what the base sliced."""
    return -(
        -(DECLARED_CONTINUED_POSITION + DECLARED_TOKENS) // DECLARED_PAGE_SIZE
    )


def _decode_metadata(banks) -> dict:
    """The same entry set on the DECODE leg, one token at the position the ring stands at.

    ``kv_segment_size`` is present and non-zero here on purpose: it is what a prefill
    chunk's span is built from, and a decode step that reached for it would be handed a
    window of three blocks instead of its context.
    """
    entry = dict(next(iter(_prefill_metadata(banks).values())))
    entry["max_query_len"] = DECLARED_TOKENS
    entry["cached_seq_len"] = torch.tensor(
        [[DECLARED_CONTINUED_POSITION]], dtype=torch.int32
    )
    entry["host_num_computed_tokens"] = torch.full(
        (1,), DECLARED_CONTINUED_POSITION, dtype=torch.int32
    )
    entry["slot_mapping"] = torch.arange(DECLARED_TOKENS, dtype=torch.int32)
    return {bank["name"]: dict(entry) for bank in banks}


def test_the_converter_sizes_the_window_from_the_legs_bucket() -> None:
    """Three candidate answers, and the item names all three so it cannot pass by luck.

    The bucket's span is ``ceil((kv_segment_size + max_query_len) / block_size)``; the
    table's width is what a prefill group's ``max_blocks_per_seq`` falls back to, which is
    the whole model length in the serving configuration; the request's own blocks are what
    the earlier code sliced. Only the first is a constant the captured graph can depend on
    AND small enough for the seam to copy.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    runner = _runner(text_config, [bank])
    span_blocks = -(
        -(DECLARED_SEGMENT + DECLARED_PREFILL_TOKENS) // DECLARED_PAGE_SIZE
    )

    converted = runner._glm5next_model_kwargs(_converter_kwargs(
        [bank], _prefill_metadata([bank]), DECLARED_PREFILL_TOKENS
    ))
    carriers = converted["layer_carriers"]
    assert len(carriers) == 1, f"one bank was handed {len(carriers)} carrier(s)"
    # THE WINDOW IS THE TABLE'S WIDTH NOW. The bank travels whole, so its row count says
    # nothing about the window; the rows the kernel assembles are the ones this row names.
    slots = int(carriers[0]["block_table_row"].shape[0]) * DECLARED_PAGE_SIZE

    say("I7_CANDIDATES",
        f"bucket_span={span_blocks * DECLARED_PAGE_SIZE}",
        f"table_width={DECLARED_TABLE_WIDTH * DECLARED_PAGE_SIZE}",
        f"request_blocks={1 * DECLARED_PAGE_SIZE}")
    say("I7_WINDOW_SLOTS", slots)
    assert slots == span_blocks * DECLARED_PAGE_SIZE
    assert slots != DECLARED_TABLE_WIDTH * DECLARED_PAGE_SIZE, (
        "the window is the block table's width, which on the prefill leg is the whole "
        "model length in blocks -- more rows than the seam can copy on chip"
    )
    assert slots != DECLARED_PAGE_SIZE, (
        "the window is this request's own blocks again, so its length still moves with "
        "the position"
    )


def test_the_decode_legs_window_is_its_context_bucket() -> None:
    """The other leg the block declares, on a sequence this item opens the production way.

    A decode group's block table IS its context bucket -- the bucket was chosen so its
    longest sequence fits -- so the table's width is the span on this leg and the prefill
    formula is not. THE SEQUENCE IS OPENED BY A PREFILL AT 0 through the same converter,
    because the live indexer ring refuses a step that continues no sequence, and a decode
    dropped onto a fresh shell would fail on THAT refusal and never reach a window at all
    (``neuron_model_runner.py:5343-5365``). An earlier draft of this item did exactly
    that, so the opening step is part of the item now and its premise is read rather
    than assumed.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    runner = _runner(text_config, [bank])
    segment_span = -(-(DECLARED_SEGMENT + DECLARED_TOKENS) // DECLARED_PAGE_SIZE)

    opened = runner._glm5next_model_kwargs(_converter_kwargs(
        [bank], _prefill_metadata([bank]), DECLARED_PREFILL_TOKENS
    ))
    # RE-PINNED: the ring stands at one position PER REQUEST SLOT, so the premise is read
    # off this request's own slot. The original reading, verbatim:
    # `cursor = getattr(runner, "_glm5next_side_cache_cursor", None)`.
    positions = dict(getattr(runner, "_glm5next_side_cache_positions", None) or {})
    cursor = positions.get(DECLARED_STATE_SLOT)
    say("I8_OPENED", f"carriers={len(opened['layer_carriers'])}",
        f"slots={sorted(positions)}", f"cursor={cursor}")
    assert cursor == DECLARED_CONTINUED_POSITION, (
        f"the opening prefill left slot {DECLARED_STATE_SLOT}'s ring at {cursor} of "
        f"{sorted(positions)} and this item's decode step carries position "
        f"{DECLARED_CONTINUED_POSITION}; without a ring that continues this sequence the "
        f"step below would be refused before any window is built, and the reading would "
        f"be about the cursor and not about the window"
    )

    converted = runner._glm5next_model_kwargs(_converter_kwargs(
        [bank], _decode_metadata([bank]), DECLARED_TOKENS
    ))
    slots = int(
        converted["layer_carriers"][0]["block_table_row"].shape[0]
    ) * DECLARED_PAGE_SIZE

    say("I8_WINDOW_SLOTS", slots,
        f"bucket={DECLARED_TABLE_WIDTH * DECLARED_PAGE_SIZE}",
        f"segment_span={segment_span * DECLARED_PAGE_SIZE}",
        f"own_blocks={_blocks_for_decode() * DECLARED_PAGE_SIZE}")
    assert slots == DECLARED_TABLE_WIDTH * DECLARED_PAGE_SIZE
    assert slots != segment_span * DECLARED_PAGE_SIZE, (
        "the decode window came from the prefill leg's segment span, which is shorter "
        "than this sequence's own context"
    )
    assert slots != _blocks_for_decode() * DECLARED_PAGE_SIZE, (
        "the decode window is this step's own blocks again, so its length still grows "
        "with every decode step"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 9. The ALLOCATOR grows no bank past the blocks the scheduler hands out.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_allocator_grows_no_bank_past_the_schedulers_blocks() -> None:
    """Item 3's production counterpart: the headroom that existed, and its reader.

    THE BASE GREW EVERY LATENT BANK BY ONE WINDOW. A layer was handed a contiguous slice
    whose length was the bucket's, so a request placed at the LAST block the scheduler can
    give it needed slots past that block, and the allocator bought them. Nothing reads
    past a request's own blocks now, so the headroom has no reader and the bytes go back.

    THE PLACEMENT IS THE READING. This drives the real carrier builder at that last block,
    on a bank holding exactly the schedulable blocks, and the source of the allocator
    beside it: the first says the placement needs no headroom, the second says none is
    bought. Either alone would leave the other open.
    """
    _require_cpu_mode()
    text_config, _ = _world()
    runner = _runner(text_config, [])

    schedulable_blocks = DECLARED_TABLE_WIDTH
    last_block = schedulable_blocks - 1
    bank = _bank(DECLARED_HEAD_SIZE, blocks=schedulable_blocks)
    geometry = {
        "block_ids": [last_block],
        "state_slot": DECLARED_STATE_SLOT,
        "page_size": DECLARED_PAGE_SIZE,
        "window_blocks": DECLARED_WINDOW_BLOCKS,
    }
    # The control for the placement: the window this table's width names ends well past
    # the bank, which is exactly the case the headroom was bought for.
    window_end = last_block * DECLARED_PAGE_SIZE + DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    say("I9_CONTROL_THE_WINDOW_WOULD_END_PAST_THE_BANK", window_end,
        f"bank={schedulable_blocks * DECLARED_PAGE_SIZE}")
    assert window_end > schedulable_blocks * DECLARED_PAGE_SIZE

    carrier = _carrier(bank, text_config, position=0, geometry=geometry)
    say("I9_THE_LAST_BLOCK_IS_SERVED_WITH_NO_HEADROOM",
        int(carrier["latent_cache"].shape[0]),
        carrier["block_table_row"].flatten().tolist(),
        carrier["latent_slots"].tolist())
    assert int(carrier["latent_cache"].shape[0]) == schedulable_blocks * DECLARED_PAGE_SIZE
    assert carrier["latent_slots"].tolist() == [last_block * DECLARED_PAGE_SIZE]

    # THE ALLOCATOR'S OWN SOURCE. The bytes it asks for are the bytes the configuration
    # names, and the method the base called to size the headroom is gone.
    source = _code_of(NeuronModelRunner.initialize_kv_cache)
    say("I9_ALLOCATES_EXACTLY_THE_CONFIGURED_SIZE",
        "torch.zeros(size, dtype=torch.int8" in source)
    assert "torch.zeros(size, dtype=torch.int8" in source
    assert "size + spare" not in source, (
        "the allocator still adds headroom to a configured size, so the bytes did not "
        "go back"
    )
    say("I9_NO_SPARE_METHOD_REMAINS",
        hasattr(NeuronModelRunner, "_latent_spare_bytes"))
    assert not hasattr(NeuronModelRunner, "_latent_spare_bytes")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 10. A write outside the request's own pages is refused RUNNER-SIDE.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_write_outside_the_requests_own_pages_is_refused_runner_side() -> None:
    """The guard the layer can no longer make, at the last place the value is a number.

    THE LAYER USED TO MAKE IT. It refused a write past the end of the cache slice it was
    handed, and that slice WAS the request's pages. The layer is handed the whole bank
    now, and the table's padded entries resolve to page 0, so a write past this request's
    last page lands on another sequence's rows with nothing short or out of range about
    it. Inside the graph the position is a tensor, so the layer cannot read it at all.
    The control below reads that silence on the width the table declares.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    tokens = DECLARED_TOKENS
    position = DECLARED_PAGE_SIZE + 1
    own_blocks = 1
    geometry = {
        "block_ids": [DECLARED_FIRST_BLOCK],
        "state_slot": DECLARED_STATE_SLOT,
        "page_size": DECLARED_PAGE_SIZE,
        "window_blocks": DECLARED_WINDOW_BLOCKS,
    }

    # THE CONTROL: the write is inside the window the table's width names, so nothing
    # downstream of the runner can catch it.
    own_slots = own_blocks * DECLARED_PAGE_SIZE
    window_slots = DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    say("I10_CONTROL", f"position={position}", f"tokens={tokens}",
        f"own_slots={own_slots}", f"window_slots={window_slots}")
    assert position + tokens > own_slots
    assert position + tokens <= window_slots

    with pytest.raises(ValueError) as caught:
        _carrier(bank, text_config, position=position, geometry=geometry)
    message = " ".join(str(caught.value).split())
    say("I10_MESSAGE", message[:190])
    assert "own pages" in message
    assert str(own_slots) in message


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 11. The indexer's sequence bound is ONE number at two consecutive steps.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_indexer_bound_is_one_number_at_two_consecutive_steps() -> None:
    """A python int that changes per step is a graph per step, which is the defect.

    The bound stays an int on purpose -- reading it off a tensor is a host read inside
    a traced region, which the indexer's own docstring records -- so the fix is that
    the int does not move. Two CONSECUTIVE decode steps are driven through the
    converter, because the live indexer ring admits only a step that continues it, and
    the two are compared against each other and against the engine's own length.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    runner = _runner(text_config, [bank])

    runner._glm5next_model_kwargs(_converter_kwargs(
        [bank], _prefill_metadata([bank]), DECLARED_PREFILL_TOKENS
    ))
    bounds = []
    for step in range(2):
        position = DECLARED_CONTINUED_POSITION + step
        metadata = _decode_metadata([bank])
        for entry in metadata.values():
            entry["cached_seq_len"] = torch.tensor([[position]], dtype=torch.int32)
            entry["host_num_computed_tokens"] = torch.full(
                (1,), position, dtype=torch.int32
            )
        converted = runner._glm5next_model_kwargs(
            _converter_kwargs([bank], metadata, DECLARED_TOKENS)
        )
        carrier = converted["layer_carriers"][0]
        bounds.append((position, int(carrier["max_seq_len"])))

    say("I11_BOUNDS", *[f"position={p}|bound={b}" for p, b in bounds])
    assert bounds[0][1] == bounds[1][1]
    assert bounds[0][1] == int(runner.max_model_len)
    # The two steps really are at different positions, so the equality above is the
    # bound standing still and not the same step read twice.
    assert bounds[0][0] != bounds[1][0]
    # And the base's answer is a different number at each of them, which is what the
    # candidate count used to be sized from.
    assert {position + DECLARED_TOKENS for position, _ in bounds} != {bounds[0][1]}


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 12. The runner hands BOTH ring positions down as tensors.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_runner_hands_both_ring_positions_down_as_tensors() -> None:
    """The two keys the ring seams are fed by: a tensor on both legs, at two positions.

    THE TWO KEYS ARE THE LAST PYTHON NUMBERS TO CROSS THE BOUNDARY. The decode leg's
    ``position`` chooses the ring slot this token's key is written to and the prefill
    leg's ``prefill_end_position`` chooses which slots the chunk's remainder occupies.
    An int there is a constant in the captured graph, exactly as it was for the cache
    write item 4 reads.

    THE INSTRUMENT IS THE DEVICE, in the form the landed seam acceptance
    (``test_indexer_position_tensor_117d.py`` A01) uses: a shape-only tensor carries no
    data, so a builder that reached for a VALUE on this bank would raise rather than
    answer. Reaching the end of both builds is the reading that it did not, and the
    shapes that come back are the reading that nothing sized itself from the position.

    WHAT THIS ITEM DOES NOT REPEAT. That the tensor route and the int route write the
    same bytes is measured by that same landed file, A02, on the seams themselves. This
    item is the RUNNER's half: that the runner is what hands the tensor down.
    """
    _require_cpu_mode()
    meta = torch.device("meta")
    text_config = Glm5NextTextConfig()
    bank = _bank(DECLARED_HEAD_SIZE, device=meta)
    legs = (
        ("prefill", True, DECLARED_PREFILL_TOKENS, "prefill_end_position"),
        ("decode", False, DECLARED_TOKENS, "position"),
    )

    forms: dict[str, list[dict[str, tuple[int, ...]]]] = {}
    for position in (0, DECLARED_SEGMENT):
        for name, is_prefill, tokens, key in legs:
            carrier = _carrier(
                bank, text_config, position=position, tokens=tokens,
                is_prefill=is_prefill,
                geometry=_geometry(position=position, tokens=tokens),
                max_seq_len=DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE,
            )
            value = carrier[key]
            say(f"I12_{name.upper()}_{key.upper()}", f"position={position}",
                type(value).__name__, getattr(value, "dtype", None),
                tuple(getattr(value, "shape", ())))
            assert torch.is_tensor(value), (
                f"the {name} leg hands the ring a python int at position {position}, "
                f"which a captured graph turns into the constant it was captured with"
            )
            assert value.dtype == torch.int32
            assert value.device.type == "meta"
            forms.setdefault(name, []).append({
                held: tuple(item.shape) for held, item in carrier.items()
                if torch.is_tensor(item)
            })

    for name, pair in forms.items():
        say(f"I12_SHAPES_{name.upper()}", pair[0] == pair[1], sorted(pair[0]))
        assert pair[0] == pair[1], (
            f"the {name} leg's carrier changes shape between position 0 and position "
            f"{DECLARED_SEGMENT}, so one captured graph cannot serve both"
        )
    # The two positions occupy a different number of the request's own blocks, so the
    # equality above is the window standing still and not one position read twice.
    assert (len(_geometry(position=0, tokens=DECLARED_TOKENS)["block_ids"])
            != len(_geometry(position=DECLARED_SEGMENT,
                            tokens=DECLARED_TOKENS)["block_ids"]))

    # AND THE SOURCE SIDE OF THE SAME SENTENCE, by ``ast`` over the builder: each key is
    # assigned ONCE, and from the helper that makes a tensor -- not from an ``int()``.
    # The builder's own host arithmetic stays an int on purpose; what this counts is the
    # value that leaves for the traced region.
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(NeuronModelRunner._glm5next_layer_carriers)
    ))
    made: dict[str, list[str]] = {"position": [], "prefill_end_position": []}
    for node in ast.walk(tree):
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if not isinstance(target, ast.Subscript):
            continue
        held = getattr(target.slice, "value", None)
        if getattr(target.value, "id", None) == "carrier" and held in made:
            value = node.value
            made[held].append(
                getattr(value.func, "attr", None) or getattr(value.func, "id", "?")
                if isinstance(value, ast.Call) else type(value).__name__
            )

    for held, spelled in made.items():
        say(f"I12_ASSIGNED_{held.upper()}", len(spelled), *spelled)
        # One equality carries both halves: assigned once, and from the tensor helper.
        assert spelled == ["_glm5next_start_position"], (
            f"'{held}' is assigned {len(spelled)} time(s) in the builder and the last is "
            f"{spelled[-1] if spelled else 'nothing'}; the one value that leaves for the "
            f"traced region has to be the tensor the helper makes"
        )
