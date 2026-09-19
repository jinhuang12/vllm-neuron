"""The sequence's length is the request's own, not the bucket it was padded to.

WHAT THIS FILE MEASURES. A prefill reaches the model padded up to its bucket, so the
tensors the geometry converter is handed are the bucket's width while the sequence
reaches only the request's real scheduled length. Two numbers therefore leave the input
builder: the width, which every traced operand keeps because that is what a captured
graph carries, and the real length, which is where the sequence ends. This file measures
the second one -- the array the input builder leaves, and the four places the converter
reads it: the block run, the ring cursor, the ring's end and the pools a chunk completes.

WHY EACH READING IS A SERVING FAILURE AND NOT A TIDINESS POINT. Blocks are allocated for
the REAL count, so a table taken from the padded width names row entries the request was
never given -- zero on a fresh table, which is a real page -- and the rows this step writes
are resolved through that table, so they land on another sequence's slots. A cursor left at
the bucket refuses the request's own first decode, which arrives at the real position. A
ring end past the sequence seeds slots that hold no token of it, and a pool completed on
padding members is pooled by the next completion.

WHAT THIS FILE DOES NOT MEASURE, stated so the gap is not read as coverage. The operand
WIDTHS: ``seq_lens``, ``slot_mapping`` and the block table all keep the bucket's width by
design, and the capture-shape items own that. The model's forward: the converter is what
changed and the converter is what runs here; the tiny generation and its reference live
in ``test_tiny_glm5next_e2e.py``. One request per step, which this half refuses to
exceed.

HOW TO RUN IT, both variables from the process environment:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_request_tokens_133.py
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# The landed tiny stack's dials and the sibling file's runner-shaped fixture, imported
# rather than re-implemented, as `test_tiny_glm5next_e2e.py:57-60` imports `-054a`'s.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The bucket a prefill is padded up to here, which is the landed stack's token count.
BUCKET_TOKENS = item.STACK_TOKENS

#: The request's REAL length: one whole pool short of the bucket, so the prompt is
#: shorter than what it was padded to AND still ends on a pool boundary. The boundary
#: keeps the prefill's remainder out of this file -- item 11 of the sibling file owns it
#: -- so a failure here is the position and nothing else.
REAL_TOKENS = BUCKET_TOKENS - item.MLA_INDEX_KPOOL

#: The one request these steps carry.
REQUEST = "req-133"

#: The blocks the scheduler really allocated: the real length's own pages.
REAL_BLOCKS = -(-REAL_TOKENS // item.MLA_PAGE_SIZE)

#: The two widths the WRITE item runs, both multiples of the dense seam's tile so the
#: root's forward admits each of them (``test_tiny_glm5next_forward.py:3605-3612``): a
#: chunk padded to twice the stack's tokens, and the real prompt on its own.
WIDE_PADDED = 2 * item.STACK_TOKENS
WIDE_REAL = item.STACK_TOKENS

#: The request's own pages, and the block-table WIDTH the padded chunk needs: the attention
#: half refuses a table whose pages name fewer rows than the step it is handed carries.
WIDE_REAL_BLOCKS = -(-WIDE_REAL // item.MLA_PAGE_SIZE)
WIDE_WINDOW_BLOCKS = -(-WIDE_PADDED // item.MLA_PAGE_SIZE)

#: The block-table row's WIDTH, which is the padded chunk's pages.
ROW_WIDTH = -(-BUCKET_TOKENS // item.MLA_PAGE_SIZE)


def _counts(length: int) -> np.ndarray:
    """The array the input builder leaves for a one-request step of ``length`` tokens."""
    return NeuronModelRunner._glm5next_request_token_counts(
        [REQUEST], {REQUEST: length}
    )


def _runner():
    """The runner the sibling file builds: the landed tiny root with its caches bound."""
    root = e2e._fixture()["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = e2e.E2E_MAX_SEQ_LEN
    # THE STEP'S IDENTITY AND THE ENGINE'S CONCURRENCY BOUND. The converter keys a
    # request's state by its id, so a real step served without one is refused by name,
    # and it sizes the indexer's per-sequence caches by the bound. Both values are the
    # sibling file's own, and this file's steps still carry the one request they always
    # carried.
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

    The block table is one buffer per batch slot, handed over at its full width
    (``neuron_model_runner.py:4252``), and the scheduler allocated the REAL count's
    pages, so the entries past them are whatever the slot last held -- zero on a fresh
    table. A run taken from the padded width reaches into them.
    """
    return list(range(REAL_BLOCKS)) + [0] * (ROW_WIDTH - REAL_BLOCKS)


def _step_on_row(runner, *, row, width: int, cached: int, sampling_row: int) -> dict:
    """One step whose block-table row is GIVEN, not derived from the padded width."""
    entry = e2e._entry(
        row=row,
        tokens=width,
        cached=cached,
        threshold=1,
        block_size=item.MLA_PAGE_SIZE,
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


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 1. the array the input builder leaves: batch order, one entry per request, the
# scheduler's real counts, host int32.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_request_length_array_meets_its_four_declared_properties():
    """(a) batch order, (b) one entry per request, (c) the real counts, (d) host int32.

    THE ORDER IS READ FROM THE BATCH AND NOT FROM THE MAPPING, which is what the
    reversed control measures: the consumer indexes this array by the batch's own
    position, so an array in the mapping's insertion order would name another request's
    length. The mapping carries a request the batch did not schedule, so a builder that
    walked the mapping would also fail (b).

    (c) IS SETTLED AT THE CALL SITE, and by a SOURCE READ rather than a run: both
    mappings the input builder holds are one dict lookup away from each other, and every
    other row in this file passes just as well with the padded one. The read is
    therefore about which mapping the production line names.
    """
    e2e._require_cpu_mode()
    scheduled = {"a": 7, "b": 3, "c": 5, "not-scheduled-this-step": 11}
    order = ["b", "c", "a"]

    counts = NeuronModelRunner._glm5next_request_token_counts(order, scheduled)
    reversed_counts = NeuronModelRunner._glm5next_request_token_counts(
        list(reversed(order)), scheduled
    )
    print(
        f"INC133|array|order={order}|counts={counts.tolist()}"
        f"|reversed={reversed_counts.tolist()}|dtype={counts.dtype}"
        f"|type={type(counts).__name__}"
    )
    assert counts.tolist() == [3, 5, 7], "entry i must be the batch's i-th request"
    assert reversed_counts.tolist() == [7, 5, 3], (
        "the order comes from the batch's request ids; this control reverses them and "
        "the array must reverse with them"
    )
    assert len(counts) == len(order), (
        "one entry per SCHEDULED request; the mapping holds a request this step did not "
        "schedule and it must not appear"
    )
    assert int(counts.sum()) == 15, "the entries are the counts, so they sum to the total"
    assert isinstance(counts, np.ndarray), "the array is a host array, not a tensor"
    assert counts.dtype == np.int32, f"host int32 is the declared dtype; got {counts.dtype}"

    source = inspect.getsource(NeuronModelRunner._prepare_model_input_impl)
    lines = source.splitlines()
    at = [i for i, line in enumerate(lines) if "_glm5next_request_token_counts(" in line]
    call = "\n".join(lines[at[0] : at[0] + 3]) if len(at) == 1 else ""
    print(f"INC133|call_site|hits={len(at)}|reads_padded={'padded' in call}")
    assert len(at) == 1, "the array is built once, where the input builder holds both mappings"
    assert "scheduler_output.num_scheduled_tokens" in call, (
        "the call site must read the scheduler's REAL per-request counts"
    )
    assert "num_scheduled_tokens_padded" not in call, (
        "the padded mapping is the defect this block removes; reading it here would "
        "restore the bucket count under a new name"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 2. the block run is the request's own pages, so a padded prefill is served on
# the row the request was really allocated.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_a_padded_prefill_is_served_on_the_row_the_request_was_really_allocated():
    """The table names the sequence's pages; the row's later entries are not the request's.

    THE ROW IS THE PRODUCTION SHAPE and that is the whole item: the scheduler allocates
    for the real count and the block table is handed over at its full width, so the
    entries past the allocation belong to no request. A table taken from the padded width
    names them, and every row this step writes is resolved through that table, so the
    write would land on pages another sequence holds.

    WHAT IS READ, NOW THAT NOTHING REFUSES IT. The carrier builder used to refuse a block
    run that was not one ascending run ("not one run"); the kernel gathers the pages the
    table names, so a scattered table is served and there is no refusal left to pin. The
    readings are therefore positive ones: the carrier names the request's OWN blocks, in
    the order it was given them, with `-1` in every entry the bucket pads, and every
    token's physical slot lies inside those blocks.

    THE CONTROL IS THE ROW ITSELF: if its padded entries continued the request's own run,
    a table taken from the padded width would name the same pages and this item would
    measure nothing, so the row is checked before the step is taken.
    """
    e2e._require_cpu_mode()
    row = _served_row()
    padded_run = row[:ROW_WIDTH]
    ascending = all(later - earlier == 1 for earlier, later in zip(padded_run, padded_run[1:]))
    print(
        f"INC133|row|width={ROW_WIDTH}|allocated={REAL_BLOCKS}"
        f"|padded_run_is_one_run={ascending}"
    )
    assert not ascending, (
        "the padded run must reach entries the request was never allocated, or this "
        "item is served with or without the change and measures nothing"
    )

    runner = _runner()
    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    kwargs = _step_on_row(
        runner, row=row, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1
    )
    served = len(kwargs["layer_carriers"])
    # THE TABLE AND THE SLOTS ARE READ FROM EVERY SPARSE CARRIER, as sets, so a stack whose
    # layers disagree about which pages the request holds reddens on the count of answers
    # rather than on whichever layer this item happened to look at.
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
    want_table = tuple(range(REAL_BLOCKS)) + (-1,) * (ROW_WIDTH - REAL_BLOCKS)
    names_its_own = tables == {want_table}
    print(f"INC133|served_on_the_real_row|carriers={served}"
          f"|cursor={runner._glm5next_side_cache_cursor}|distinct_tables={len(tables)}"
          f"|names_the_requests_own_blocks={names_its_own}"
          f"|allocated={REAL_BLOCKS}|row_width={ROW_WIDTH}|padded_entries="
          f"{ROW_WIDTH - REAL_BLOCKS}|slot_ranges={sorted(slot_ranges)}")
    assert served == len(runner.model.glm5next_layer_banks), (
        "the step must be served: the table names the request's own pages and pads the rest"
    )
    assert tables == {want_table}, (
        f"every sparse carrier must name the request's {REAL_BLOCKS} allocated block(s) in "
        f"the order the row gives them and pad the rest of its {ROW_WIDTH} entry(ies) with "
        f"-1; a table built from the padded width names entries this request never got"
    )
    assert slot_ranges == {(0, REAL_TOKENS - 1)}, (
        f"the physical slots must run from this request's first row to its last real one "
        f"({REAL_TOKENS - 1}), which are inside the {REAL_BLOCKS} page(s) it holds; a slot "
        f"past them was resolved through an entry the request was never given"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 3. a padded prefill leaves the ring at the request's own position, and the
# request's first decode is served.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_a_padded_prefill_leaves_the_ring_at_the_requests_position_and_the_decode_serves():
    """The prefill's tensors are the bucket's width; the ring stands at the real length.

    THE SECOND STEP IS THE ONE THAT COULD NOT BE SERVED. It arrives at the request's own
    position, which is the length the scheduler counted, and the converter refuses a
    position the ring does not stand at. Before this block the ring stood at the bucket,
    so this decode raised and the request was prefilled and abandoned.
    """
    e2e._require_cpu_mode()
    assert REAL_TOKENS < BUCKET_TOKENS, (
        "this item measures a prompt SHORTER than its bucket; equal lengths would make "
        "both readings below pass without the change"
    )
    runner = _runner()

    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1)
    cursor = runner._glm5next_side_cache_cursor
    print(
        f"INC133|prefill|bucket={BUCKET_TOKENS}|real={REAL_TOKENS}|cursor={cursor}"
    )
    assert cursor == REAL_TOKENS, (
        f"the ring must stand where the request's sequence ends ({REAL_TOKENS}); it "
        f"stands at {cursor}"
    )

    runner._glm5next_request_tokens = _counts(1)
    kwargs = _step(runner, width=1, cached=REAL_TOKENS, sampling_row=0)
    served = len(kwargs["layer_carriers"])
    print(
        f"INC133|first_decode|position={REAL_TOKENS}|carriers={served}"
        f"|cursor={runner._glm5next_side_cache_cursor}"
    )
    assert served == len(runner.model.glm5next_layer_banks), (
        "the first decode must be served: one carrier per layer of the stack"
    )
    assert runner._glm5next_side_cache_cursor == REAL_TOKENS + 1, (
        "and the decode advances the ring by its own one token"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 4. the ring's end and the pools the chunk completes stand at the request's
# position, while the operand keeps the bucket's width.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_ring_end_and_the_pool_tail_stand_at_the_requests_own_position():
    """The end position is the sequence's end, and no pool completes on padding rows.

    THE TWO READINGS ARE ONE DEFECT SEEN TWICE. The prefill leg seeds the ring's
    remainder up to the end position it is handed, and marks the rows where a pool
    completes. Taken from the padded width, both reach rows that carry no token of this
    sequence: the ring's slots past the sequence would be seeded from padding keys, and
    the pool completing inside the padding would be pooled by the next completion.

    THE WIDTH IS READ BESIDE THEM, because it must NOT move: the mapping is one entry per
    packed row and a captured graph was compiled for that width.
    """
    e2e._require_cpu_mode()
    pool = int(item.MLA_INDEX_KPOOL)
    assert REAL_TOKENS % pool == 0, (
        "this item wants the last real position to complete a pool, so the reading below "
        "is a present pool and not an absent one"
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
    print(
        f"INC133|ring_end|ends={sorted(ends)}|real={REAL_TOKENS}|widths={sorted(widths)}"
        f"|bucket={BUCKET_TOKENS}|last_real_pool={sorted(last_real)}"
        f"|past_real={sorted(past_real)}"
    )
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


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 5. a step the input builder never prepared reads its own width.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_a_step_the_input_builder_never_prepared_reads_its_own_width():
    """Warmup and capture build their own tensors, and that width IS their real length.

    THE ARRAY IS TAKEN AND NOT LEFT, which the last reading measures: a warmup step
    between two real steps must not inherit the previous request's length, so the
    converter clears the array it used. Seven of the eight call sites reach the converter
    without one.
    """
    e2e._require_cpu_mode()
    runner = _runner()

    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=BUCKET_TOKENS - 1)
    unprepared = runner._glm5next_side_cache_cursor
    print(f"INC133|unprepared_step|width={BUCKET_TOKENS}|cursor={unprepared}")
    assert unprepared == BUCKET_TOKENS, (
        "a step with no array left for it reaches the end of the tensor it built"
    )

    runner._glm5next_request_tokens = _counts(REAL_TOKENS)
    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=REAL_TOKENS - 1)
    prepared = runner._glm5next_side_cache_cursor
    _step(runner, width=BUCKET_TOKENS, cached=0, sampling_row=BUCKET_TOKENS - 1)
    after = runner._glm5next_side_cache_cursor
    print(f"INC133|array_is_taken|prepared={prepared}|next_unprepared={after}")
    assert prepared == REAL_TOKENS, "the prepared step reads the request's length"
    assert after == BUCKET_TOKENS, (
        "the step after it was never prepared, so the previous request's length must "
        "not reach it"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 6. the padded rows write the last real slot's own latent, so the bank after a
# padded prefill is the bank after the same prefill unpadded.
# ══════════════════════════════════════════════════════════════════════════════════════


def _wide_arm(*, width: int, text_config, banks, side, ids):
    """One prefill of ``width`` rows whose real length is ``WIDE_REAL``, through the root.

    The carriers come from the RUNNER's own builder, as the sibling file's item 5 drives
    them, so this arm exercises the production chain and not a hand-built mapping.
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
        softmax_scale=item.MLA_SOFTMAX_SCALE,
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
    """The one prompt both write items run, drawn from the sibling file's own seed."""
    return torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (WIDE_REAL,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    )


#: The depth the chunked paths accumulate over, which is what multiplies the dtype's own unit
#: round-off. It is the chunk width those paths use, not a number fitted to a measurement.
ACCUMULATION_DEPTH = 8


def _worst_offender(own, reference) -> tuple[float, int, float]:
    """The worst gap as a multiple of the bank's bound, how many values moved, and that bound.

    THE BOUND IS ABSOLUTE AT THE BANK'S SCALE and is derived from the bank's own dtype:
    `ACCUMULATION_DEPTH * finfo(dtype).eps * max(1, max|reference|)`. Reassociation error is
    absolute at the scale of the OPERANDS, so a bound taken relative to each value reads
    cancellation into a small output as a large relative move and fails on a value that moved
    by one ulp near one. A ratio at or under 1 means every value sits inside that bound.
    """
    wide, other = own.to(torch.float32), reference.to(torch.float32)
    scale = max(1.0, float(other.abs().max()))
    bound = ACCUMULATION_DEPTH * torch.finfo(own.dtype).eps * scale
    gap = float((wide - other).abs().max())
    return gap / bound, int((wide != other).sum()), bound


def _where_they_differ(own, reference, index: int, family: str) -> str:
    """How far apart two banks of one layer are, as a reading rather than a boolean.

    THE SIZE OF THE DIFFERENCE IS THE READING, because two answers look identical to a
    byte-equal oracle and are not the same finding: a spread at the last bits of the dtype
    is a different arithmetic order over a different row count, while a large or clustered
    difference is a value that came from a row the request does not own.

    The first dimension is the page, the one axis that means the same thing in every bank
    this fixture allocates; nothing here reads a slot, because a bank with two KV heads
    interleaves them under the page and a slot number there names no one row.
    """
    wide, other = own.to(torch.float32).flatten(), reference.to(torch.float32).flatten()
    delta = wide - other
    moved = delta.nonzero().flatten()
    scale = torch.maximum(wide.abs(), other.abs()).clamp(min=1e-30)
    pages = (own != reference).flatten(start_dim=1).any(dim=1).nonzero().flatten()
    first = [
        (int(at), round(float(wide[at]), 6), round(float(other[at]), 6))
        for at in moved[:5]
    ]
    return (
        f"INC133|write_diff|layer={index}|family={family}|dtype={own.dtype}"
        f"|shape={tuple(own.shape)}|pages={tuple(own.shape)[0]}"
        f"|pages_that_differ={int(pages.numel())}|first_pages={[int(p) for p in pages[:8]]}"
        f"|values={int(wide.numel())}|values_that_differ={int(moved.numel())}"
        f"|max_abs_diff={float(delta.abs().max()):.6g}"
        f"|max_rel_diff={float((delta.abs() / scale).max()):.6g}"
        f"|deltas_above_zero={int((delta > 0).sum())}|deltas_below_zero={int((delta < 0).sum())}"
        f"|first_values_that_moved={first}"
    )


def test_the_padded_rows_write_the_last_real_slot_and_leave_the_bank_unpadded():
    """Two runs of one prompt -- padded and not -- must leave the request's pages equal.

    WHAT THE PADDED ARM MAY NOT DO. The rows past the real length carry no token of the
    sequence, and the pages the request was allocated stop at that length, so a write
    from those rows would land on pages this request does not hold. The second reading is
    that ban stated positively: every block past the request's own pages must still be
    zero after the padded arm.

    WHY A TOLERANCE AND NOT BYTE EQUALITY. The write itself is exact: the padded rows are
    collapsed onto the LAST REAL ROW and store the value that row stores anyway, so no
    order of the repeated writes changes what is stored. What the two arms do NOT share is
    the row count the layers above reduce over, and a sum reassociated over 256 rows
    instead of 128 lands on a neighbouring representable value. A measured run moved 80 of
    16384 values by one bfloat16 ulp of their own magnitude, in both directions; a padded
    row's own data reaching this bank would move whole rows by a fraction of their size,
    which is what the bound below separates.

    HOW MANY VALUES MOVE IS A READING, NOT A THRESHOLD, and the reason is what the measured
    runs show: the share grows with DEPTH. Layer 0 moved nothing, layer 1 moved 80 of 16384
    over 3 pages, and layer 2 moved 1164 over 12 pages -- 7 percent -- while every one of
    those values stayed inside the absolute bound (worst 0.11 of it, largest gap two
    bfloat16 steps). A later query reduces over more keys, so more of its sums cross a
    blocking boundary and the moves compound through the layers above. A share ceiling
    would therefore fail on arithmetic and would have to be raised each time the stack
    grows a layer, which is a threshold measuring the wrong thing.

    WHAT STILL CATCHES A LEAK, with the share gone as an assertion. A padded row's own data
    reaching this bank moves at least a whole pool of values by a fraction of their
    magnitude, so it cannot sit inside an eight-step bound: the bound is the first detector
    and it is elementwise. The second is that nothing may be written past the pages the
    request holds, which a leak into another sequence's slots fails outright. The third
    lives in the selection item, which counts the pools a real query chose and requires
    none past the request's own reach.

    THE UNPADDED ARM IS THE REFERENCE, run on its own caches with the same ids, the same
    window and the same pooled store, so the only difference between the arms is the
    padding itself.
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

    families = [bank["family"] for bank in root.glm5next_layer_banks]
    for index, (padded, unpadded) in enumerate(zip(written["padded"], written["unpadded"])):
        own = padded[:WIDE_REAL_BLOCKS]
        beyond = padded[WIDE_REAL_BLOCKS:]
        same = torch.equal(own, unpadded[:WIDE_REAL_BLOCKS])
        touched = int((beyond != 0).sum())
        stored = int((own != 0).sum())
        print(
            f"INC133|write|layer={index}|own_pages_equal={same}"
            f"|blocks_beyond_the_request={tuple(beyond.shape)[0]}|nonzero_beyond={touched}"
            f"|values_in_its_own_pages={stored}"
        )
        print(_where_they_differ(own, unpadded[:WIDE_REAL_BLOCKS], index, families[index]))
        # THE POSITIVE CONTROL FIRST. Two banks that were never written are equal to each
        # other, so a tolerance alone would pass on a run that stored nothing at all.
        assert stored > 0, (
            f"layer {index} holds nothing in the {WIDE_REAL_BLOCKS} page(s) the request "
            f"owns, so the comparison below would compare two empty banks"
        )
        worst, moved, bound = _worst_offender(own, unpadded[:WIDE_REAL_BLOCKS])
        print(
            f"INC133|write_bound|layer={index}|worst_ratio_of_its_own_bound={worst:.6g}"
            f"|bound={bound:.6g}|dtype_eps={torch.finfo(own.dtype).eps:.6g}"
            f"|values_that_differ={moved}|share_that_differ={moved / own.numel():.6g}"
        )
        assert worst <= 1.0, (
            f"layer {index} holds a value {worst:.6g} times the bank's own bound {bound:.6g} "
            f"away from the unpadded run; reassociation over the wider row count moves the "
            f"last bits of a value, and a padded row's own data does not"
        )
        # HOW MANY VALUES MOVED IS PRINTED ABOVE AND ASSERTED NOWHERE. It grows with depth
        # because a later query reduces over more keys, so a ceiling here would fail on
        # arithmetic; the docstring names the three detectors that catch a leak instead.
        assert touched == 0, (
            f"layer {index} wrote {touched} value(s) past the {WIDE_REAL_BLOCKS} page(s) "
            f"the request holds; those slots belong to other sequences"
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 7. the ring's remainder comes from the chunk's last REAL rows, never from the
# padded ones.
# ══════════════════════════════════════════════════════════════════════════════════════

#: A real length that does NOT divide into pools, so the chunk leaves an open pool and
#: the seeding actually writes. Derived, so a change to either dial moves it.
SEED_REAL = BUCKET_TOKENS - item.MLA_INDEX_KPOOL - 2

#: The value planted in every padded row. No real row can hold it: the real rows carry
#: their own 1-based position, so the ring holding this number can only have read a row
#: that is not the sequence's. It is EXACT IN BFLOAT16, which the item checks before it
#: plants it: a value the bank's dtype rounds away is stored as something else, and then
#: the comparison against it can never be true and the reading says nothing.
PADDING_MARKER = -1024.0

#: The request slot whose ring this item seeds. The side caches hold one set PER REQUEST
#: SLOT, so the ring an indexer is handed is one slot's row of that set, which is what
#: the production path hands it (``side["tail"][state_slot]``). This item allocates one
#: slot, so its request owns the first one.
SEED_SLOT = 0


def test_the_rings_remainder_comes_from_the_chunks_last_real_rows():
    """The open pool's rows are the sequence's last ones, not the operand's last ones.

    WHAT THE RING IS FOR. A prefill pools only whole pools; the positions after its last
    complete pool are stashed in the ring, and the first decode step completes that pool
    FROM the ring. Rows read out of a padded operand's tail therefore end up pooled as
    if they were the sequence's own keys, and nothing below notices.

    THE MARKER MAKES THE FAILURE UNMISTAKABLE rather than approximate: every padded row
    carries a value no real row can hold, so a ring that holds it read a row that is not
    this sequence's. The reading prints both halves of the ring.

    THE RING AND THE KEY SHAPES ARE THE RUNNER'S OWN -- the ring comes from the runner's
    side-cache builder, so this item cannot pass on a ring shaped the production path
    would not produce.
    """
    e2e._require_cpu_mode()
    pool = int(item.MLA_INDEX_KPOOL)
    assert SEED_REAL % pool != 0, (
        "this item needs an OPEN pool; a real length that divides into pools seeds "
        "nothing and the item would pass on both trees"
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
    # RE-PINNED (D17.1). The ring set carries a leading REQUEST-SLOT axis, and an indexer
    # is handed one request's row of it, never the whole set. The original reading,
    # verbatim: "written = indexer.seed_tail(ring, ..." and "float(ring[0][slot][0])" --
    # one ring for the process, seeded and read whole. The property is the same one: the
    # remainder comes from the chunk's last real rows. This is a view, so every write
    # below reaches the set the marker count reads.
    request_ring = ring[SEED_SLOT]

    dim = int(text_config.index_head_dim)
    rows = torch.arange(1, BUCKET_TOKENS + 1, dtype=torch.float32).reshape(-1, 1)
    key = rows.expand(BUCKET_TOKENS, dim).clone().to(torch.bfloat16)
    # THE MARKER MUST SURVIVE THE DTYPE IT IS PLANTED IN. A value bfloat16 rounds away is
    # stored as something else, and the comparison below could then never be true: the
    # reading would pass on any tree, which is the same as no reading at all.
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
    want = [float(SEED_REAL - open_rows + slot + 1) for slot in range(open_rows)]
    got = [float(request_ring[0][slot][0]) for slot in range(open_rows)]
    marker_hits = int((ring.to(torch.float32) == PADDING_MARKER).sum())
    print(
        f"INC133|seeded|real={SEED_REAL}|padded={BUCKET_TOKENS}|open_rows={open_rows}"
        f"|rows_written={int(written)}|keys={got}|want={want}"
        f"|marker_hits={marker_hits}"
    )
    assert int(written) == open_rows, (
        f"the open pool holds {open_rows} row(s) of this chunk and {int(written)} were "
        f"written"
    )
    assert got == want, (
        f"the ring must hold the sequence's last real keys {want}; it holds {got}"
    )
    assert marker_hits == 0, (
        f"the ring holds the padding marker in {marker_hits} place(s), so the remainder "
        f"was read from rows that carry no token of this sequence"
    )
    kept = [float(request_ring[0][slot][0]) for slot in range(open_rows, pool)]
    print(f"INC133|seeded_untouched|slots={list(range(open_rows, pool))}|values={kept}")
    assert kept == [0.0] * (pool - open_rows), (
        "the slots above the open pool belong to no position of this chunk and must "
        "keep what they held"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 8. the collapsed write TRACES: a compiler records the duplicate indices, and the
# recorded graph stores what the eager run stored.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_padded_write_traces_and_stores_what_the_eager_run_stored():
    """The collapsed write must survive being recorded, not only being executed.

    WHY TRACING IS ITS OWN READING. The clamp makes several rows share one index and the
    write goes in place through a view, so a compiler has to record a write whose indices
    repeat. ``backend="eager"`` records the graph and then runs it eagerly, which asks the
    tracing question on its own, without any backend's further limits.

    WHAT IT GRADES. The recorded run's own pages equal the eager run's byte for byte, and
    no block past the request's pages is written, so a trace that dropped a duplicate
    write or reordered the writes into a different result cannot pass. The graph-break
    count is PRINTED and not graded: a break elsewhere in this stack is not this reading.
    """
    e2e._require_cpu_mode()
    root = e2e._fixture()["root"]
    ids = _wide_ids()
    positions = torch.tensor([WIDE_REAL - 1], dtype=torch.long)

    eager = _wide_written(root, width=WIDE_PADDED, ids=ids, positions=positions)
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    traced = _wide_written(
        root,
        width=WIDE_PADDED,
        ids=ids,
        positions=positions,
        call=torch.compile(root.forward, backend="eager", dynamic=False),
    )
    breaks = sum(counters["graph_break"].values())
    print(f"INC133|traced|graph_breaks={breaks}|banks={len(traced)}")

    for index, (one, two) in enumerate(zip(eager, traced)):
        own = two[:WIDE_REAL_BLOCKS]
        beyond = two[WIDE_REAL_BLOCKS:]
        same = torch.equal(own, one[:WIDE_REAL_BLOCKS])
        touched = int((beyond != 0).sum())
        stored = int((own != 0).sum())
        print(
            f"INC133|traced_write|layer={index}|own_pages_equal={same}"
            f"|nonzero_beyond={touched}|values_in_its_own_pages={stored}"
        )
        # THE SAME POSITIVE CONTROL AS THE EAGER ITEM: two banks nothing wrote are equal.
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


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 9. the padded arm's selection stops at the request's own pools.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_padded_arm_selects_no_pool_beyond_the_requests_own_length(monkeypatch):
    """No real query may select a pool the request's own length does not reach.

    WHAT THIS ASKS THAT THE WRITE ITEM DOES NOT. The write item compares what was STORED,
    and a selection that reached a padded pool would show up there only as whatever value
    that pool happened to hold. This reads the selection itself: the pool ids the indexer
    returns for the rows that carry real tokens, counted against the pools the request's
    own length reaches. Zero is the whole reading.

    WHY THE COUNT AND NOT A BOOLEAN. A count states how much of the axis leaked, so a
    control that removes the bound reads a number rather than a flipped flag.

    THE ROWS GRADED ARE THE REAL ONES. Rows at or past the real length carry no token of
    the sequence and their outputs are discarded, so what they select is not this item's
    subject.
    """
    e2e._require_cpu_mode()
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer

    root = e2e._fixture()["root"]
    pool = int(item.MLA_INDEX_KPOOL)
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
        "the indexer's bounded selection was never called, so this item measured no "
        "selection at all"
    )
    beyond = sum(int((chosen[:WIDE_REAL] >= reachable).sum()) for chosen in seen)
    widths = sorted({tuple(chosen.shape) for chosen in seen})
    print(
        f"INC133|selection|calls={len(seen)}|shapes={widths}|pool={pool}"
        f"|real_tokens={WIDE_REAL}|pools_the_request_reaches={reachable}"
        f"|rows_graded={WIDE_REAL}|selected_beyond_the_request={beyond}"
    )
    assert beyond == 0, (
        f"{beyond} selection(s) landed on a pool at or past {reachable}, which the "
        f"request's {WIDE_REAL} real tokens never reach; a real query reading a padded "
        f"pool attends keys that belong to no row of this sequence"
    )
