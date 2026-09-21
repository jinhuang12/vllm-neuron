"""The sequence's length is the request's own, not the bucket it was padded to.

A prefill reaches the model padded up to its bucket, so the tensors the geometry converter
is handed are the bucket's width while the sequence reaches only the request's real
scheduled length. Two numbers therefore leave the input builder: the width, which every
traced operand keeps because that is what a captured graph carries, and the real length,
which is where the sequence ends. The second one is what this file reads -- the array the
input builder leaves, and the four places the converter reads it: the block run, the ring
cursor, the ring's end and the pools a chunk completes.

Each of those is a serving failure rather than a tidiness point. Blocks are allocated for
the real count, so a table taken from the padded width names row entries the request was
never given -- zero on a fresh table, which is a real page -- and the rows this step writes
are resolved through that table, so they land on another sequence's slots. A cursor left at
the bucket refuses the request's own first decode, which arrives at the real position. A
ring end past the sequence seeds slots that hold no token of it, and a pool completed on
padding rows is pooled again by the next completion.

The operand widths are deliberately not read here: ``seq_lens``, ``slot_mapping`` and the
block table all keep the bucket's width by design. Every step carries one request.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_request_tokens.py
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# The tiny stack's dials and its runner-shaped fixture, imported rather than re-implemented.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# The bucket a prefill is padded up to here, which is the tiny stack's token count.
BUCKET_TOKENS = tiny.STACK_TOKENS

# The request's real length: one whole pool short of the bucket, so the prompt is shorter
# than what it was padded to and still ends on a pool boundary. The boundary keeps a
# prefill remainder out of these readings, so a failure here is the position and nothing
# else.
REAL_TOKENS = BUCKET_TOKENS - tiny.MLA_INDEX_KPOOL

# The one request these steps carry.
REQUEST = "req-0"

# The blocks the scheduler really allocated: the real length's own pages.
REAL_BLOCKS = -(-REAL_TOKENS // tiny.MLA_PAGE_SIZE)

# The two widths the write tests run, both multiples of the dense seam's tile so the root's
# forward admits each of them: a chunk padded to twice the stack's tokens, and the real
# prompt on its own.
WIDE_PADDED = 2 * tiny.STACK_TOKENS
WIDE_REAL = tiny.STACK_TOKENS

# The request's own pages, and the block-table width the padded chunk needs: the attention
# half refuses a table whose pages name fewer rows than the step it is handed carries.
WIDE_REAL_BLOCKS = -(-WIDE_REAL // tiny.MLA_PAGE_SIZE)
WIDE_WINDOW_BLOCKS = -(-WIDE_PADDED // tiny.MLA_PAGE_SIZE)

# The block-table row's width, which is the padded chunk's pages.
ROW_WIDTH = -(-BUCKET_TOKENS // tiny.MLA_PAGE_SIZE)


def _counts(length: int) -> np.ndarray:
    """The array the input builder leaves for a one-request step of ``length`` tokens."""
    return NeuronModelRunner._glm5next_request_token_counts(
        [REQUEST], {REQUEST: length}
    )


def _runner():
    """The tiny root with its caches bound, on a runner shell."""
    root = e2e._fixture()["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = e2e.E2E_MAX_SEQ_LEN
    # The converter keys a request's state by its id, so a real step served without one is
    # refused by name, and it sizes the indexer's per-sequence caches by the concurrency
    # bound.
    runner.input_batch = SimpleNamespace(req_ids=[REQUEST])
    runner.max_num_reqs = e2e.E2E_MAX_NUM_SEQS
    return runner


def _step(runner, *, width: int, cached: int, sampling_row: int) -> dict:
    """One step through the converter, its tensors ``width`` wide, at ``cached``."""
    return e2e._model_kwargs(
        runner,
        input_ids=torch.zeros(width, dtype=torch.long),
        cached=cached,
        sampling_row=sampling_row,
    )


def _served_row() -> list[int]:
    """The row a served request carries: its own blocks, then entries it never got.

    The block table is one buffer per batch slot, handed over at its full width, and the
    scheduler allocated the real count's pages, so the entries past them are whatever the
    slot last held -- zero on a fresh table. A run taken from the padded width reaches into
    them.
    """
    return list(range(REAL_BLOCKS)) + [0] * (ROW_WIDTH - REAL_BLOCKS)


def _step_on_row(runner, *, row, width: int, cached: int, sampling_row: int) -> dict:
    """One step whose block-table row is given, not derived from the padded width."""
    entry = e2e._entry(
        row=row,
        tokens=width,
        cached=cached,
        threshold=1,
        block_size=tiny.MLA_PAGE_SIZE,
    )
    generic = {
        "input_ids": torch.zeros(width, dtype=torch.long),
        "positions": torch.arange(width, dtype=torch.long) + cached,
        "attn_metadata": {
            str(bank["name"]): entry for bank in runner.model.glm5next_layer_banks
        },
        "sampling_positions": torch.tensor([sampling_row], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }
    return runner._glm5next_model_kwargs(generic)


def test_the_request_length_array_is_in_batch_order_with_the_real_counts():
    """Batch order, one entry per scheduled request, the real counts, host int32.

    The order comes from the batch and not from the mapping, which the reversed reading
    shows: the consumer indexes this array by the batch's own position, so an array in the
    mapping's insertion order would name another request's length. The mapping also holds a
    request the batch did not schedule, which a builder that walked the mapping would
    include.

    Which mapping the call site reads is settled by a source read rather than a run: the
    padded and the real mapping are one dict lookup apart, and every other reading in this
    file passes just as well with the padded one.
    """
    e2e._require_cpu_mode()
    scheduled = {"a": 7, "b": 3, "c": 5, "not-scheduled-this-step": 11}
    order = ["b", "c", "a"]

    counts = NeuronModelRunner._glm5next_request_token_counts(order, scheduled)
    reversed_counts = NeuronModelRunner._glm5next_request_token_counts(
        list(reversed(order)), scheduled
    )
    assert counts.tolist() == [3, 5, 7], "entry i must be the batch's i-th request"
    assert reversed_counts.tolist() == [7, 5, 3], (
        "the order comes from the batch's request ids, so reversing them has to reverse "
        "the array with them"
    )
    assert len(counts) == len(order), (
        "one entry per scheduled request; the mapping holds a request this step did not "
        "schedule and it must not appear"
    )
    assert int(counts.sum()) == 15, "the entries are the counts, so they sum to the total"
    assert isinstance(counts, np.ndarray), "the array is a host array, not a tensor"
    assert counts.dtype == np.int32, f"host int32 is the declared dtype; got {counts.dtype}"

    source = inspect.getsource(NeuronModelRunner._prepare_model_input_impl)
    lines = source.splitlines()
    at = [i for i, line in enumerate(lines) if "_glm5next_request_token_counts(" in line]
    call = "\n".join(lines[at[0] : at[0] + 3]) if len(at) == 1 else ""
    assert len(at) == 1, "the array is built once, where the input builder holds both mappings"
    assert "scheduler_output.num_scheduled_tokens" in call, (
        "the call site must read the scheduler's real per-request counts"
    )
    assert "num_scheduled_tokens_padded" not in call, (
        "the padded mapping is the defect this block removes; reading it here would "
        "restore the bucket count under a new name"
    )


def test_a_padded_prefill_is_served_on_the_row_the_request_was_really_allocated():
    """The table names the sequence's pages; the row's later entries are not the request's.

    The row here is the production shape: the scheduler allocates for the real count and
    the block table is handed over at its full width, so the entries past the allocation
    belong to no request. A table taken from the padded width names them, and every row
    this step writes is resolved through that table, so the write would land on pages
    another sequence holds.

    The kernel gathers the pages the table names, so a scattered table is served and there
    is no refusal to catch. The readings are positive ones instead: the carrier names the
    request's own blocks, in the order it was given them, with ``-1`` in every entry the
    bucket pads, and every token's physical slot lies inside those blocks.

    The row itself is checked before the step is taken. If its padded entries continued the
    request's own run, a table taken from the padded width would name the same pages and
    nothing here would separate the two.
    """
    e2e._require_cpu_mode()
    row = _served_row()
    padded_run = row[:ROW_WIDTH]
    ascending = all(later - earlier == 1 for earlier, later in zip(padded_run, padded_run[1:]))
    assert not ascending, (
        "the padded run must reach entries the request was never allocated, or a table "
        "built from the padded width would name the same pages"
    )

    runner = _runner()
    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    kwargs = _step_on_row(
        runner, row=row, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1
    )
    served = len(kwargs["layer_carriers"])
    # Read from every sparse carrier, as sets, so a stack whose layers disagree about which
    # pages the request holds fails on the count of answers rather than on whichever layer
    # was looked at first.
    tables = {
        tuple(int(entry) for entry in carrier["block_table_row"].flatten().tolist())
        for carrier in kwargs["layer_carriers"]
        if "block_table_row" in carrier
    }
    slot_ranges = {
        (int(carrier["latent_slots"].min()), int(carrier["latent_slots"].max()))
        for carrier in kwargs["layer_carriers"]
        if "latent_slots" in carrier
    }
    expected_table = tuple(range(REAL_BLOCKS)) + (-1,) * (ROW_WIDTH - REAL_BLOCKS)
    assert served == len(runner.model.glm5next_layer_banks), (
        "the step must be served: the table names the request's own pages and pads the rest"
    )
    assert tables == {expected_table}, (
        f"every sparse carrier must name the request's {REAL_BLOCKS} allocated block(s) in "
        f"the order the row gives them and pad the rest of its {ROW_WIDTH} entry(ies) with "
        f"-1; a table built from the padded width names entries this request never got"
    )
    assert slot_ranges == {(0, REAL_TOKENS - 1)}, (
        f"the physical slots must run from this request's first row to its last real one "
        f"({REAL_TOKENS - 1}), which are inside the {REAL_BLOCKS} page(s) it holds; a slot "
        f"past them was resolved through an entry the request was never given"
    )


def test_a_padded_prefill_leaves_the_ring_at_the_requests_position_and_the_decode_serves():
    """The prefill's tensors are the bucket's width; the ring stands at the real length.

    The second step is the one a ring left at the bucket cannot serve: it arrives at the
    request's own position, which is the length the scheduler counted, and the converter
    refuses a position the ring does not stand at.
    """
    e2e._require_cpu_mode()
    assert REAL_TOKENS < BUCKET_TOKENS, (
        "the prompt has to be shorter than its bucket; equal lengths would make both "
        "readings below pass on any tree"
    )
    runner = _runner()

    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1)
    cursor = runner._glm5next_side_cache_cursor
    assert cursor == REAL_TOKENS, (
        f"the ring must stand where the request's sequence ends ({REAL_TOKENS}); it "
        f"stands at {cursor}"
    )

    runner._glm5next_request_tokens = _counts(1)
    kwargs = _step(runner, width=1, cached=REAL_TOKENS, sampling_row=0)
    served = len(kwargs["layer_carriers"])
    assert served == len(runner.model.glm5next_layer_banks), (
        "the first decode must be served: one carrier per layer of the stack"
    )
    assert runner._glm5next_side_cache_cursor == REAL_TOKENS + 1, (
        "and the decode advances the ring by its own one token"
    )


def test_the_ring_end_and_the_pool_tail_stand_at_the_requests_own_position():
    """The end position is the sequence's end, and no pool completes on padding rows.

    The prefill leg seeds the ring's remainder up to the end position it is handed, and
    marks the rows where a pool completes. Taken from the padded width, both reach rows
    that carry no token of this sequence: the ring's slots past the sequence would be
    seeded from padding keys, and the pool completing inside the padding would be pooled
    again by the next completion.

    The width is read beside them because it must not move: the mapping is one entry per
    packed row and a captured graph was compiled for that width.
    """
    e2e._require_cpu_mode()
    pool = int(tiny.MLA_INDEX_KPOOL)
    assert REAL_TOKENS % pool == 0, (
        "the last real position has to complete a pool, so the reading below is a present "
        "pool and not an absent one"
    )
    runner = _runner()
    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    kwargs = _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1)

    seeded = [carrier for carrier in kwargs["layer_carriers"] if "slot_mapping" in carrier]
    ends = {int(carrier["prefill_end_position"]) for carrier in seeded}
    widths = {int(carrier["slot_mapping"].shape[0]) for carrier in seeded}
    last_real = {int(carrier["slot_mapping"][REAL_TOKENS - 1]) for carrier in seeded}
    past_real = set()
    for carrier in seeded:
        past_real.update(int(value) for value in carrier["slot_mapping"][REAL_TOKENS:])
    assert ends == {REAL_TOKENS}, (
        f"the ring is seeded to where the sequence ends ({REAL_TOKENS}); this step seeded "
        f"to {sorted(ends)}"
    )
    assert widths == {BUCKET_TOKENS}, (
        "the mapping is one entry per packed row, so its width stays the bucket's"
    )
    assert last_real == {(REAL_TOKENS - 1) // pool}, (
        "the last real position completes its pool and must carry that pool's id"
    )
    assert past_real == {-1}, (
        "a padding row carries no token of the sequence, so no pool may complete on it"
    )


def test_a_step_the_input_builder_never_prepared_reads_its_own_width():
    """Warmup and capture build their own tensors, and that width is their real length.

    The converter takes the array rather than leaving it, which the last reading shows: a
    warmup step between two real steps must not inherit the previous request's length.
    Seven of the eight model call sites reach the converter without an array at all.
    """
    e2e._require_cpu_mode()
    runner = _runner()

    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=BUCKET_TOKENS - 1)
    unprepared = runner._glm5next_side_cache_cursor
    assert unprepared == BUCKET_TOKENS, (
        "a step with no array left for it reaches the end of the tensor it built"
    )

    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1)
    prepared = runner._glm5next_side_cache_cursor
    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=BUCKET_TOKENS - 1)
    after = runner._glm5next_side_cache_cursor
    assert prepared == REAL_TOKENS, "the prepared step reads the request's length"
    assert after == BUCKET_TOKENS, (
        "the step after it was never prepared, so the previous request's length must "
        "not reach it"
    )


def _wide_arm(*, width: int, text_config, banks, side, ids):
    """One prefill of ``width`` rows whose real length is ``WIDE_REAL``, through the root.

    The carriers come from the runner's own builder, so this arm exercises the production
    chain and not a hand-built mapping.
    """
    carriers = NeuronModelRunner._glm5next_layer_carriers(
        banks,
        side,
        geometries=e2e._geometries(
            banks,
            block_ids=range(WIDE_REAL_BLOCKS),
            state_slot=0,
            window_blocks=WIDE_WINDOW_BLOCKS,
        ),
        is_prefill=True,
        tokens=width,
        real_tokens=WIDE_REAL,
        start_position=0,
        softmax_scale=tiny.MLA_SOFTMAX_SCALE,
        max_seq_len=WIDE_PADDED,
        index_kpool=int(text_config.index_kpool),
    )
    padded_ids = torch.cat([ids, torch.zeros(width - WIDE_REAL, dtype=torch.int64)])
    return carriers, padded_ids


def _wide_written(root, *, width: int, ids, positions, call=None) -> list[torch.Tensor]:
    """One run of ``width`` rows on fresh caches; the banks' first slot afterwards.

    ``call`` is the entry point, so a compiled forward can be handed in where the eager
    one is the default and the two arms share every other operand.
    """
    text_config = root.text_config
    caches = e2e._runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=WIDE_PADDED,
        request_slots=e2e.E2E_MAX_NUM_SEQS,
    )
    carriers, input_ids = _wide_arm(
        width=width,
        text_config=text_config,
        banks=banks,
        side=side,
        ids=ids,
    )
    (call or root.forward)(
        input_ids, layer_carriers=carriers, sampling_positions=positions
    )
    return [caches[bank["name"]][0].clone() for bank in banks]


def _wide_ids():
    """The one prompt both write tests run, from the tiny stack's own seed."""
    return torch.randint(
        0,
        tiny.STACK_VOCAB_SIZE,
        (WIDE_REAL,),
        generator=torch.Generator().manual_seed(tiny.SEED_STACK_IDS),
        dtype=torch.int64,
    )


# The depth the chunked paths accumulate over, which is what multiplies the dtype's own
# unit round-off. It is the chunk width those paths use, not a number fitted to a
# measurement.
ACCUMULATION_DEPTH = 8


def _worst_offender(own, reference) -> tuple[float, int, float]:
    """The worst gap as a multiple of the bank's bound, how many values moved, and the bound.

    The bound is absolute at the bank's scale and derived from the bank's own dtype:
    ``ACCUMULATION_DEPTH * finfo(dtype).eps * max(1, max|reference|)``. Reassociation error
    is absolute at the scale of the operands, so a bound taken relative to each value reads
    cancellation into a small output as a large relative move and fails on a value that
    moved by one ulp near one. A ratio at or under 1 means every value sits inside the
    bound.
    """
    wide, other = own.to(torch.float32), reference.to(torch.float32)
    scale = max(1.0, float(other.abs().max()))
    bound = ACCUMULATION_DEPTH * torch.finfo(own.dtype).eps * scale
    gap = float((wide - other).abs().max())
    return gap / bound, int((wide != other).sum()), bound


def test_the_padded_rows_write_the_last_real_slot_and_leave_the_bank_unpadded():
    """Two runs of one prompt -- padded and not -- leave the request's pages equal.

    The rows past the real length carry no token of the sequence, and the pages the request
    was allocated stop at that length, so a write from those rows would land on pages this
    request does not hold. The second reading states that positively: every block past the
    request's own pages is still zero after the padded arm.

    The comparison is a tolerance and not byte equality. The write itself is exact, because
    the padded rows are collapsed onto the last real row and store the value that row
    stores anyway. What the two arms do not share is the row count the layers above reduce
    over, and a sum reassociated over 256 rows instead of 128 lands on a neighbouring
    representable value: a measured run moved 80 of 16384 values by one bfloat16 ulp of
    their own magnitude, in both directions.

    How many values move is therefore not asserted. The share grows with depth -- layer 0
    moved nothing, layer 1 moved 80 of 16384 over 3 pages, layer 2 moved 1164 over 12
    pages -- because a later query reduces over more keys, so more of its sums cross a
    blocking boundary. A ceiling on the share would have to be raised each time the stack
    grows a layer.

    What catches a leak instead is elementwise: a padded row's own data reaching this bank
    moves at least a whole pool of values by a fraction of their magnitude, which cannot
    sit inside an eight-step bound. Beside it, nothing may be written past the pages the
    request holds, and the selection reading below counts the pools a real query chose.

    The unpadded arm is the reference, run on its own caches with the same ids, the same
    window and the same pooled store, so the only difference between the arms is the
    padding.
    """
    e2e._require_cpu_mode()
    assert WIDE_PADDED > WIDE_REAL, "the padded arm must be wider than the prompt"
    root = e2e._fixture()["root"]
    ids = _wide_ids()
    positions = torch.tensor([WIDE_REAL - 1], dtype=torch.long)

    written = {
        label: _wide_written(root, width=width, ids=ids, positions=positions)
        for label, width in (("padded", WIDE_PADDED), ("unpadded", WIDE_REAL))
    }

    for index, (padded, unpadded) in enumerate(zip(written["padded"], written["unpadded"])):
        own = padded[:WIDE_REAL_BLOCKS]
        beyond = padded[WIDE_REAL_BLOCKS:]
        touched = int((beyond != 0).sum())
        stored = int((own != 0).sum())
        # Two banks that were never written are equal to each other, so the tolerance
        # below would pass on a run that stored nothing at all.
        assert stored > 0, (
            f"layer {index} holds nothing in the {WIDE_REAL_BLOCKS} page(s) the request "
            f"owns, so the comparison below would compare two empty banks"
        )
        worst, _moved, bound = _worst_offender(own, unpadded[:WIDE_REAL_BLOCKS])
        assert worst <= 1.0, (
            f"layer {index} holds a value {worst:.6g} times the bank's own bound {bound:.6g} "
            f"away from the unpadded run; reassociation over the wider row count moves the "
            f"last bits of a value, and a padded row's own data does not"
        )
        assert touched == 0, (
            f"layer {index} wrote {touched} value(s) past the {WIDE_REAL_BLOCKS} page(s) "
            f"the request holds; those slots belong to other sequences"
        )


# A real length that does not divide into pools, so the chunk leaves an open pool and the
# seeding actually writes. Derived, so a change to either dial moves it.
SEED_REAL = BUCKET_TOKENS - tiny.MLA_INDEX_KPOOL - 2

# The value planted in every padded row. No real row can hold it, because the real rows
# carry their own 1-based position, so a ring holding this number read a row that is not
# the sequence's. It is exact in bfloat16, which the test checks before planting it: a
# value the bank's dtype rounds away is stored as something else, and the comparison
# against it could then never be true.
PADDING_MARKER = -1024.0

# The request slot whose ring is seeded. The side caches hold one set per request slot, so
# the ring an indexer is handed is one slot's row of that set. One slot is allocated here,
# so this request owns the first one.
SEED_SLOT = 0


def test_the_rings_remainder_comes_from_the_chunks_last_real_rows():
    """The open pool's rows are the sequence's last ones, not the operand's last ones.

    A prefill pools only whole pools; the positions after its last complete pool are
    stashed in the ring, and the first decode step completes that pool from the ring. Rows
    read out of a padded operand's tail would therefore be pooled as if they were the
    sequence's own keys.

    Every padded row carries a value no real row can hold, so a ring that holds it read a
    row that is not this sequence's. The ring itself comes from the runner's own side-cache
    builder, so nothing here can pass on a ring the production path would not produce.
    """
    e2e._require_cpu_mode()
    pool = int(tiny.MLA_INDEX_KPOOL)
    assert SEED_REAL % pool != 0, (
        "an open pool is needed here; a real length that divides into pools seeds nothing "
        "and the reading would pass on any tree"
    )
    root = e2e._fixture()["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=pool,
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=BUCKET_TOKENS,
        request_slots=e2e.E2E_MAX_NUM_SEQS,
    )
    ring = next(entry["tail"] for entry in side if "tail" in entry)
    ring.zero_()
    # The ring set carries a leading request-slot axis and an indexer is handed one
    # request's row of it, never the whole set. This is a view, so every write below
    # reaches the set the marker count reads.
    request_ring = ring[SEED_SLOT]

    dim = int(text_config.index_head_dim)
    rows = torch.arange(1, BUCKET_TOKENS + 1, dtype=torch.float32).reshape(-1, 1)
    key = rows.expand(BUCKET_TOKENS, dim).clone().to(torch.bfloat16)
    # The marker has to survive the dtype it is planted in: a value bfloat16 rounds away
    # is stored as something else, and the comparison below could then never be true.
    planted = float(torch.tensor(PADDING_MARKER, dtype=key.dtype))
    assert planted == PADDING_MARKER, (
        f"the marker {PADDING_MARKER} is stored as {planted} in {key.dtype}, so the row "
        f"below would compare against a value the ring can never hold"
    )
    key[SEED_REAL:] = PADDING_MARKER
    gate_score = key.clone()

    indexer = root.model.layers[0].self_attn.indexer
    written = indexer.seed_tail(
        request_ring,
        key,
        gate_score,
        torch.tensor(SEED_REAL, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
    )

    open_rows = SEED_REAL % pool
    expected = [float(SEED_REAL - open_rows + slot + 1) for slot in range(open_rows)]
    got = [float(request_ring[0][slot][0]) for slot in range(open_rows)]
    marker_hits = int((ring.to(torch.float32) == PADDING_MARKER).sum())
    assert int(written) == open_rows, (
        f"the open pool holds {open_rows} row(s) of this chunk and {int(written)} were "
        f"written"
    )
    assert got == expected, (
        f"the ring must hold the sequence's last real keys {expected}; it holds {got}"
    )
    assert marker_hits == 0, (
        f"the ring holds the padding marker in {marker_hits} place(s), so the remainder "
        f"was read from rows that carry no token of this sequence"
    )
    kept = [float(request_ring[0][slot][0]) for slot in range(open_rows, pool)]
    assert kept == [0.0] * (pool - open_rows), (
        "the slots above the open pool belong to no position of this chunk and must "
        "keep what they held"
    )


def test_the_padded_write_traces_and_stores_what_the_eager_run_stored():
    """The collapsed write survives being recorded, not only being executed.

    The clamp makes several rows share one index and the write goes in place through a
    view, so a compiler has to record a write whose indices repeat. ``backend="eager"``
    records the graph and then runs it eagerly, which asks the tracing question on its own
    without any backend's further limits.

    The recorded run's own pages equal the eager run's byte for byte, and no block past the
    request's pages is written, so a trace that dropped a duplicate write or reordered the
    writes into a different result cannot pass. Graph breaks elsewhere in the stack are not
    this reading.
    """
    e2e._require_cpu_mode()
    root = e2e._fixture()["root"]
    ids = _wide_ids()
    positions = torch.tensor([WIDE_REAL - 1], dtype=torch.long)

    eager = _wide_written(root, width=WIDE_PADDED, ids=ids, positions=positions)
    torch._dynamo.reset()
    traced = _wide_written(
        root,
        width=WIDE_PADDED,
        ids=ids,
        positions=positions,
        call=torch.compile(root.forward, backend="eager", dynamic=False),
    )

    for index, (one, two) in enumerate(zip(eager, traced)):
        own = two[:WIDE_REAL_BLOCKS]
        beyond = two[WIDE_REAL_BLOCKS:]
        same = torch.equal(own, one[:WIDE_REAL_BLOCKS])
        touched = int((beyond != 0).sum())
        stored = int((own != 0).sum())
        # As above: two banks nothing wrote are equal to each other.
        assert stored > 0, (
            f"layer {index} holds nothing in the {WIDE_REAL_BLOCKS} page(s) the request "
            f"owns, so the equality below would compare two empty banks"
        )
        assert same, (
            f"layer {index}'s own pages differ between the recorded run and the eager "
            f"one; the trace did not store what the writes store"
        )
        assert touched == 0, (
            f"layer {index} wrote {touched} value(s) past the {WIDE_REAL_BLOCKS} page(s) "
            f"the request holds while it was traced"
        )


def test_the_padded_arm_selects_no_pool_beyond_the_requests_own_length(monkeypatch):
    """No real query selects a pool the request's own length does not reach.

    The write reading above compares what was stored, where a selection that reached a
    padded pool would show up only as whatever value that pool happened to hold. This reads
    the selection itself: the pool ids the indexer returns for the rows that carry real
    tokens, counted against the pools the request's own length reaches. The count says how
    much of the axis leaked rather than only that something did.

    Only the real rows are graded. Rows at or past the real length carry no token of the
    sequence and their outputs are discarded.
    """
    e2e._require_cpu_mode()
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer

    root = e2e._fixture()["root"]
    pool = int(tiny.MLA_INDEX_KPOOL)
    reachable = -(-WIDE_REAL // pool)
    seen: list[torch.Tensor] = []
    original = Glm5NextDSAIndexer.select_bounded_pools

    def recording(self, scores, seq_lens):
        """The real method, with its answer kept."""
        chosen = original(self, scores, seq_lens)
        seen.append(chosen.detach().clone())
        return chosen

    monkeypatch.setattr(Glm5NextDSAIndexer, "select_bounded_pools", recording)
    _wide_written(
        root,
        width=WIDE_PADDED,
        ids=_wide_ids(),
        positions=torch.tensor([WIDE_REAL - 1], dtype=torch.long),
    )
    assert seen, (
        "the indexer's bounded selection was never called, so no selection was read"
    )
    beyond = sum(int((chosen[:WIDE_REAL] >= reachable).sum()) for chosen in seen)
    assert beyond == 0, (
        f"{beyond} selection(s) landed on a pool at or past {reachable}, which the "
        f"request's {WIDE_REAL} real tokens never reach; a real query reading a padded "
        f"pool attends keys that belong to no row of this sequence"
    )
