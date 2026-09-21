# SPDX-License-Identifier: Apache-2.0
"""One captured graph serves every position, whatever position it was captured at.

Every shape on the traced path has to come from a bucket width rather than from a
position, and every position has to arrive as a tensor.
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
# the declared geometry. Every number here is this file's own fixture, not a
# registered dial: the block's subject is the relation between the window's length
# and the position, and that relation has to hold at any legal geometry.
# --------------------------------------------------------------------------- #
#: slots per block in the bank and in the group's page.
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
#: bucket: no test here reads that classification, and three of them read a position past it.
DECLARED_REQUEST = "capture-shape-request"
#: The first block this request is given, deliberately not block 0, so a window that
#: started at the bank's own base rather than the request's would be visible.
DECLARED_FIRST_BLOCK = 2
#: This request's slot in the per-sequence caches. The state slot is a request slot on an
#: axis as wide as the engine's bound, so it is not a block id and does not move with the
#: pages a request is given.
DECLARED_STATE_SLOT = 0
#: The two positions tests 1 and 2 compare. They differ by enough to change how many
#: blocks the request occupies, which is what moved the base's window length.
DECLARED_LOW_POSITION = 3
DECLARED_HIGH_POSITION = 25
DECLARED_TOKENS = 1
#: The carrier keys a sparse layer is handed, as the builder writes them.
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


#: Check 7's own numbers. The table is deliberately far wider than the span, and the
#: request occupies far less than either, so the three candidate answers -- the bucket's
#: span, the table's width and the request's own blocks -- are three different numbers.
DECLARED_TABLE_WIDTH = 10
DECLARED_SEGMENT = 16
DECLARED_PREFILL_TOKENS = 8
DECLARED_DECODE_THRESHOLD = 1

#: The position a prefill of ``DECLARED_PREFILL_TOKENS`` tokens leaves the live indexer ring
#: standing at, which is the only position a following decode step may carry: the converter
#: advances its cursor to ``start_position + tokens`` and refuses a step that does not
#: continue it (``neuron_model_runner.py``). Derived, so the two cannot
#: drift apart.
DECLARED_CONTINUED_POSITION = DECLARED_PREFILL_TOKENS


_PROSE_TOKENS = {tokenize.COMMENT, tokenize.STRING} | {
    kind for kind in (getattr(tokenize, name, None)
                      for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"))
    if kind is not None
}


def _code_of(obj) -> str:
    """``obj``'s source with every comment and string constant blanked out. """
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
    """One sparse bank, carrying only the four keys the carrier builder reads. """
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
    """This request's pages at ``position``, plus the block table's width. """
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
    """The one carrier the builder hands this bank's layer at ``position``. """
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
    """A text config and one sparse bank, the pair every behavioural test drives."""
    text_config = Glm5NextTextConfig()
    return text_config, _bank(DECLARED_HEAD_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
# check 1. Every operand has one shape at two positions.
# ══════════════════════════════════════════════════════════════════════════════
def test_every_operand_the_layer_is_handed_has_one_shape_at_two_positions() -> None:
    """The defect, read directly: no shape the layer sees may move with the position.
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
    assert low_blocks != high_blocks, (
        "the two positions occupy the same number of blocks, so this test could not "
        "tell a constant window from a request-shaped one"
    )

    assert tuple(low["latent_cache"].shape) == tuple(high["latent_cache"].shape)
    assert tuple(low["latent_cache"].shape) == (
        DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE,
        1,
        head_size,
    )

    assert tuple(low["block_table_row"].shape) == (DECLARED_WINDOW_BLOCKS, 1)
    assert tuple(high["block_table_row"].shape) == (DECLARED_WINDOW_BLOCKS, 1)
    assert tuple(low["latent_slots"].shape) == (DECLARED_TOKENS,)
    assert tuple(high["latent_slots"].shape) == (DECLARED_TOKENS,)

    # The tables differ, which is what makes the three shape readings above a reading
    # about a constant shape rather than about a constant operand.
    assert not torch.equal(low["block_table_row"], high["block_table_row"])


# ══════════════════════════════════════════════════════════════════════════════
# check 2. The bank travels whole, and the table names the request's own pages.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_bank_travels_whole_and_the_table_names_the_requests_pages() -> None:
    """Three properties: the bank is the bank, it is a view, and the table is the request's.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    geometry = _geometry(position=DECLARED_LOW_POSITION)
    carrier = _carrier(bank, text_config, position=DECLARED_LOW_POSITION,
                       geometry=geometry)

    slots = DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE
    assert int(carrier["latent_cache"].shape[0]) == slots
    assert len(geometry["block_ids"]) < DECLARED_WINDOW_BLOCKS, (
        "this request already occupies the whole table, so the test could not tell "
        "the request's own entries from the padding"
    )

    # No copy. A copy of the bank per layer per step is the cost the paging removes, and
    # a copy would also leave the layer writing into a tensor the next step never reads.
    assert carrier["latent_cache"].data_ptr() == bank["latent_cache"].data_ptr()

    row = carrier["block_table_row"]
    want = geometry["block_ids"] + [-1] * (DECLARED_WINDOW_BLOCKS - len(geometry["block_ids"]))
    assert row.dtype == torch.int32
    assert row.flatten().tolist() == want, (
        "the table does not name this request's own pages in order, so the rows the "
        "kernel gathers are not this request's"
    )

    # The slots are physical rows of the bank, derived from the very blocks above.
    first = geometry["block_ids"][0] * DECLARED_PAGE_SIZE + DECLARED_LOW_POSITION
    assert carrier["latent_slots"].tolist() == [first]


# ══════════════════════════════════════════════════════════════════════════════
# check 3. Scattered pages are carried, and a bank with no headroom is carried.
# ══════════════════════════════════════════════════════════════════════════════
def test_scattered_pages_are_carried_and_a_bank_with_no_headroom_is_carried() -> None:
    """The two refusals this change removes, stated as what is now accepted. """
    _require_cpu_mode()
    text_config, bank = _world()

    # Arm 1: pages in no ascending order at all, one of them behind the first.
    scattered = [DECLARED_FIRST_BLOCK + 5, DECLARED_FIRST_BLOCK, DECLARED_FIRST_BLOCK + 2]
    steps = [later - earlier for earlier, later in zip(scattered, scattered[1:])]
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
    assert carrier["block_table_row"].flatten().tolist() == want
    first = scattered[0] * DECLARED_PAGE_SIZE + DECLARED_LOW_POSITION
    assert carrier["latent_slots"].tolist() == [first], (
        "the slot is not in the page the table names first, so the write and the "
        "gather disagree about where this token's row lives"
    )

    # Arm 2: a bank holding exactly the blocks a request can be given and not one more.
    # The window those blocks make runs past its end, and nothing reads past them now.
    tight_blocks = DECLARED_FIRST_BLOCK + 1
    tight = _bank(DECLARED_HEAD_SIZE, blocks=tight_blocks)
    assert tight_blocks * DECLARED_PAGE_SIZE < DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    tight_carrier = _carrier(tight, text_config, position=DECLARED_LOW_POSITION,
                             geometry=_geometry(position=DECLARED_LOW_POSITION))
    assert int(tight_carrier["latent_cache"].shape[0]) == tight_blocks * DECLARED_PAGE_SIZE

    # The falsifier: more blocks than the bucket's table holds is still a refusal.
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
    assert "block table holds" in message


# ══════════════════════════════════════════════════════════════════════════════
# check 4. The position reaches the layer as a tensor.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_position_reaches_the_layer_as_a_tensor() -> None:
    """The traced boundary takes a tensor; a python int there is baked into the graph.
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
    assert torch.is_tensor(position), (
        "the position reaches the layer as a python int, which a captured graph turns "
        "into the constant it was captured with"
    )
    assert int(position) == DECLARED_HIGH_POSITION


# ══════════════════════════════════════════════════════════════════════════════
# check 5. The MLA read no longer depends on the position (structural).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_mla_read_no_longer_depends_on_the_position() -> None:
    """``attend``'s own source: the read is the whole window and no host read remains.
    """
    _require_cpu_mode()
    source = _code_of(model_fp8.Glm5NextMLAAttention.attend)

    assert source.count("int(start_position)") == 0

    assert "c_kv = latent_cache[:, 0, :]" in source
    assert "[: start + tokens" not in source, (
        "the read is still cut at the position, so its length still moves with every "
        "decode step"
    )

    assert "index_copy_" in source

    assert "_int64_scalar(start_position" in source
    assert "torch.as_tensor(" not in source and "torch.tensor(" not in source, (
        "the start is built from python data, which a meta trace keeps real"
    )


# ══════════════════════════════════════════════════════════════════════════════
# check 6. The KDA entering state is chosen without reading the position (structural).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_kda_entering_state_is_chosen_without_reading_the_position() -> None:
    """The branch became a device choice, and the helper that makes it returns a tensor.
    """
    _require_cpu_mode()
    source = _code_of(model_fp8.Glm5NextKDAAttention.forward)

    assert source.count("int(start_position)") == 0
    assert "torch.where(" in source
    assert "_start_is_zero(" in source

    device = torch.device("cpu")
    at_zero = model_fp8._start_is_zero(torch.tensor(0, dtype=torch.int32), device)
    above = model_fp8._start_is_zero(torch.tensor(DECLARED_HIGH_POSITION), device)
    assert torch.is_tensor(at_zero) and torch.is_tensor(above)
    assert at_zero.dtype == torch.bool and at_zero.dim() == 0
    assert bool(at_zero) is True
    assert bool(above) is False


# ══════════════════════════════════════════════════════════════════════════════
# check 7. The converter sizes the window from the leg's bucket, not the table's width.
# ══════════════════════════════════════════════════════════════════════════════
def _runner(text_config, banks) -> NeuronModelRunner:
    """A runner carrying only what the converter reads, built the way. """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(text_config=text_config, glm5next_layer_banks=banks)
    runner.max_model_len = DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE
    # The converter sizes its per-sequence caches by the engine's concurrent-sequence
    # bound, so a runner shell must model that bound too. One sequence is what this file
    # drives.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    # The bound above is only half of it: the converter also keys each sequence's state by
    # its request id, and a step served without one is the opening bucket's. Every step
    # this file drives belongs to the one request named above.
    runner.input_batch = SimpleNamespace(req_ids=[DECLARED_REQUEST])
    # The context-parallel width the block-table arithmetic divides by. One is the
    # single-rank case, which is what a shell with no parallel world can honestly say.
    runner._dcp_size = 1
    return runner


def _prefill_metadata(banks) -> dict:
    """One entry per layer, in the runner's own key set, on the prefill leg at position 0.
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
    """The blocks this test's decode step occupies, which is what the base sliced."""
    return -(
        -(DECLARED_CONTINUED_POSITION + DECLARED_TOKENS) // DECLARED_PAGE_SIZE
    )


def _decode_metadata(banks) -> dict:
    """The same entry set on the decode leg, one token at the position the ring stands at.
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
    """Three candidate answers, and the test names all three so it cannot pass by luck.
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
    # The window is the table's width now. The bank travels whole, so its row count says
    # nothing about the window; the rows the kernel assembles are the ones this row names.
    slots = int(carriers[0]["block_table_row"].shape[0]) * DECLARED_PAGE_SIZE

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
    """The other leg the block declares, on a sequence this test opens the production way.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    runner = _runner(text_config, [bank])
    segment_span = -(-(DECLARED_SEGMENT + DECLARED_TOKENS) // DECLARED_PAGE_SIZE)

    # The sequence is opened by a prefill at position 0 through the same converter,
    # because the indexer ring refuses a step that continues no sequence: a decode
    # dropped onto a fresh shell would fail on that refusal and never reach a window.
    runner._glm5next_model_kwargs(
        _converter_kwargs([bank], _prefill_metadata([bank]), DECLARED_PREFILL_TOKENS)
    )
    # The ring stands at one position per request slot, so the premise is read off this
    # request's own slot rather than off a single runner-wide cursor.
    positions = dict(getattr(runner, "_glm5next_side_cache_positions", None) or {})
    cursor = positions.get(DECLARED_STATE_SLOT)
    assert cursor == DECLARED_CONTINUED_POSITION, (
        f"the opening prefill left slot {DECLARED_STATE_SLOT}'s ring at {cursor} of "
        f"{sorted(positions)} and this test's decode step carries position "
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
# check 9. The allocator grows no bank past the blocks the scheduler hands out.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_allocator_grows_no_bank_past_the_schedulers_blocks() -> None:
    """Check 3's production counterpart: the headroom that existed, and its reader. """
    _require_cpu_mode()
    text_config, _ = _world()
    _runner(text_config, [])

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
    assert window_end > schedulable_blocks * DECLARED_PAGE_SIZE

    carrier = _carrier(bank, text_config, position=0, geometry=geometry)
    assert int(carrier["latent_cache"].shape[0]) == schedulable_blocks * DECLARED_PAGE_SIZE
    assert carrier["latent_slots"].tolist() == [last_block * DECLARED_PAGE_SIZE]

    # The allocator's own source. The bytes it asks for are the bytes the configuration
    # names, and the method the base called to size the headroom is gone.
    source = _code_of(NeuronModelRunner.initialize_kv_cache)
    assert "torch.zeros(size, dtype=torch.int8" in source
    assert "size + spare" not in source, (
        "the allocator still adds headroom to a configured size, so the bytes did not "
        "go back"
    )
    assert not hasattr(NeuronModelRunner, "_latent_spare_bytes")


# ══════════════════════════════════════════════════════════════════════════════
# check 10. A write outside the request's own pages is refused runner-side.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_write_outside_the_requests_own_pages_is_refused_runner_side() -> None:
    """The guard the layer can no longer make, at the last place the value is a number.
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

    # The control: the write is inside the window the table's width names, so nothing
    # downstream of the runner can catch it.
    own_slots = own_blocks * DECLARED_PAGE_SIZE
    window_slots = DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    assert position + tokens > own_slots
    assert position + tokens <= window_slots

    with pytest.raises(ValueError) as caught:
        _carrier(bank, text_config, position=position, geometry=geometry)
    message = " ".join(str(caught.value).split())
    assert "own pages" in message
    assert str(own_slots) in message


# ══════════════════════════════════════════════════════════════════════════════
# check 11. The indexer's sequence bound is one number at two consecutive steps.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_indexer_bound_is_one_number_at_two_consecutive_steps() -> None:
    """A python int that changes per step is a graph per step, which is the defect. """
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

    assert bounds[0][1] == bounds[1][1]
    assert bounds[0][1] == int(runner.max_model_len)
    # The two steps really are at different positions, so the equality above is the
    # bound standing still and not the same step read twice.
    assert bounds[0][0] != bounds[1][0]
    # And the base's answer is a different number at each of them, which is what the
    # candidate count used to be sized from.
    assert {position + DECLARED_TOKENS for position, _ in bounds} != {bounds[0][1]}


# ══════════════════════════════════════════════════════════════════════════════
# check 12. The runner hands both ring positions down as tensors.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_runner_hands_both_ring_positions_down_as_tensors() -> None:
    """The two keys the ring seams are fed by: a tensor on both legs, at two positions.
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
        assert pair[0] == pair[1], (
            f"the {name} leg's carrier changes shape between position 0 and position "
            f"{DECLARED_SEGMENT}, so one captured graph cannot serve both"
        )
    # The two positions occupy a different number of the request's own blocks, so the
    # equality above is the window standing still and not one position read twice.
    assert (len(_geometry(position=0, tokens=DECLARED_TOKENS)["block_ids"])
            != len(_geometry(position=DECLARED_SEGMENT,
                            tokens=DECLARED_TOKENS)["block_ids"]))

    # And the source side of the same sentence, by ``ast`` over the builder: each key is
    # assigned once, and from the helper that makes a tensor -- not from an ``int``.
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
        # One equality carries both halves: assigned once, and from the tensor helper.
        assert spelled == ["_glm5next_start_position"], (
            f"'{held}' is assigned {len(spelled)} time(s) in the builder and the last is "
            f"{spelled[-1] if spelled else 'nothing'}; the one value that leaves for the "
            f"traced region has to be the tensor the helper makes"
        )
