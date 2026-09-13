# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the request-keyed cache state: the runner half.

THE DECLARED ACCEPTANCE, the Tier N harness this campaign uses:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_request_keyed_state_117.py \\
      -s -rA -p no:randomly -p no:cacheprovider --timeout 60

EVERY ITEM IN THIS FILE FAILS AT THE BASE COMMIT, WITH TWO NAMED EXCEPTIONS, AND
THAT IS THE POINT. At base the converter refused a block table carrying more than
one row (``neuron_model_runner.py:5136-5141`` at base), so each two-request item
below raises that refusal instead of reading a value, and the helper-level items
raise ``AttributeError`` for methods base does not have. An item that passed at
base would be measuring nothing. THAT REFUSAL IS NOW THE SPARSE FAMILY'S: the walk
admits a batch, the linear family serves it, and the sparse carrier refuses a
second request by name because its slice is contiguous -- which is the arm A1's
fourth item reads. THE FIRST EXCEPTION is the prefill-warmup arm of A2,
which passes at base BECAUSE base served that step: it is a regression arm, it
pins the base's behaviour, and it fails on the bytes of this increment's fifth
commit, which is where review round 1 found the regression. THE SECOND EXCEPTION is
A5's mapping half: it calls only base code and this file's own tensors, so it passes
at both ends and is here as a standing property rather than as an arrival.

THE ITEMS HERE, and each names the tripwire it must fail on.

* A1, FOUR ARMS -- the SLOT half: two admitted requests get DIFFERENT recurrent
  state slots, stable within one request's life (tripwire: a table returning one
  slot for both). The CARRIER half: each request's entry is a storage view of its
  OWN bank row, the two entries are different storage, each request's position
  travels with it, and a write through one entry reaches the bank while the other
  request's row stays equal (tripwire: a carrier of copies passes every pointer
  read and fails that write). The CONVERTER arm: a two-request batch on a linear
  stack is served end to end and each request's ring records its own advance
  (tripwire: the previous commit's walk refused a second row). The SPARSE arm: on a
  hybrid stack the same batch is refused by name at the sparse carrier, which takes
  one contiguous slice of the paged latent bank.
* A2, TEN ARMS -- a finished request's slot is reused and ZEROED at hand-out; an
  over-admission refuses by name; a synthetic step takes no claim; the prefill
  WARMUP's own shape, which has no request at all, is served from slot 0 and takes
  no claim either; the per-sequence side caches carry ONE set per admitted sequence
  and not one per bank slot; a live request the scheduler skips for one step
  keeps its slot and its state; and a stack whose recurrent banks hold FEWER slots
  than the engine admits sequences refuses by name. Tripwires: a table that never
  frees cannot seat the later request, one that frees without zeroing fails the zero
  read, a synthetic step that seated itself changes the table, a converter that
  demands an identity from warmup raises where the base served it, an allocator sized
  by the block space reports the bank's slot count as its axis, a table that frees on
  absence loses the skipped request's slot and recurrence, and a capacity read that
  checks only that the bound is positive returns a slot number the banks cannot
  address. AND THE THREE ARMS OF THE COMPUTED-LENGTH RULE: a whole one-token prompt
  -- which the leg test reads as a decode -- is keyed by its own request rather than
  served from slot 0; it OPENS its own indexer ring rather than being refused for
  want of a cursor; and a step the engine did schedule cannot be classified as
  having no request at all. Tripwires: another request holds slot 0 across the first
  two, so a converter reading the leg serves that request's row and empties that
  request's ring, and the third's refusal is proved conditional in the same item by
  the request-less call it must still serve.
* A3 -- one request's pooled store and tail ring are byte-unchanged by a step of
  the other request, on both legs. Tripwire: the process-wide allocation, which
  made the two carriers one storage.
* A4 -- the per-token operands for a two-request batch equal the two per-request
  derivations concatenated, exactly. Tripwire: the batch-wide derivation from a
  single start position, which is run in the same item and must differ.
* A5, MAPPING HALF -- the slot number the KV machinery computes and the bank
  view's row index are ONE element, proved by a sentinel rather than by restating
  the formula. Tripwire: a wrong block stride or a head-interleaving view makes
  the read-back miss; the neighbouring-slot control catches a write that landed
  wider than one slot.

* A6 -- a decode carrying more tokens than it has requests refuses by name, reaches
  no seam, and leaves the slot table and the pooled stores byte-identical, with the
  admitted case built in the same item. Tripwire: the landed refusal read "one
  token", which is the batch's count and not the per-request one.
* A7 -- a padded decode row's latent slot is a sentinel and never slot 0. Tripwire:
  the padding value the KV machinery writes is ``NULL_BLOCK_ID``, which is zero and
  therefore a REAL slot; the item asserts that value is inside the addressable range
  before asserting no padded entry lands there.

STILL TO COME IN THIS FILE: A5's write-and-read-back clause, the
interleaved-vs-sequential differential and the R-3 seam readings, which belong to
this increment's kernel half. A1's carrier-VIEW half is no longer among them: it
landed with the model-side sliver and is the second item below.

WHY THE HARNESS IS RE-AUTHORED HERE rather than imported. The two landed files
with a converter harness -- ``test_kda_runner_state.py`` and
``tiny/test_tiny_glm5next_e2e.py`` -- each state that their harness keeps their
own file's single writer. Importing either would make that file a dependency of
this one and take the property away from it. The shapes below are theirs; the
code is this file's own.

CONVENTIONS. ``model_fp8`` is never imported at module level, because
``test_factory.py:318-319`` is a landed assertion that it stays out of
``sys.modules``. The runner is stood up with ``__new__`` and given only the
attributes the converter reads, so a converter that started reading something
new raises here instead of quietly finding a stand-in.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    NeuronModelRunner,
    _compute_slot_mapping_cpu,
)

# ---------------------------------------------------------------------------
# Declared values. Each is either read off the runner or declared here because
# nothing landed binds it; the difference is stated per value.
# ---------------------------------------------------------------------------

#: Two requests is the smallest batch that can show request keying at all: one
#: request cannot collide with itself.
DECLARED_REQUESTS = 2

#: The recurrent-state bank's slot count. Four rather than two so that a slot
#: number can never be confused with a block id or a request index by accident.
#: THIS IS A PAGING GEOMETRY, not a request axis: on the serving stack it is the
#: group's KV block count, which is why it is deliberately larger than the bound
#: below and why no per-sequence cache may be sized by it.
DECLARED_STATE_SLOTS = 4

#: How many sequences the modelled engine admits at once -- the scheduler's
#: ``max_num_seqs``, which the runner reads once at construction. Two, so that the
#: number DIFFERS from the bank's slot count above: an allocation sized by the wrong
#: one of the two is then visible as a shape rather than passing by coincidence.
DECLARED_MAX_NUM_SEQS = 2

#: The page. Nothing landed binds a page for a synthetic bank, so this file
#: declares one and the bank below is built to it; the converter cross-checks the
#: two against each other, which is the property A4 relies on.
DECLARED_PAGE_SIZE = 32

#: The latent bank's block count and its width. The width is the only shape the
#: sparse branch of the carrier builder reads off the bank.
DECLARED_BLOCKS = 8
DECLARED_HEAD_SIZE = 64
DECLARED_KV_HEADS = 1

#: The indexer's two widths. The side-cache allocator reads both and derives its
#: row count from them, so they are named rather than inlined.
DECLARED_INDEX_KPOOL = 8
DECLARED_INDEX_HEAD_DIM = 32

#: The leg test the converter makes is ``max_query_len > decode_token_threshold``,
#: so one means a single-token step reads as a decode.
DECLARED_DECODE_THRESHOLD = 1

#: The two requests' cached lengths. They DIFFER on purpose: equal lengths would
#: let a single-start-position derivation pass A4 by coincidence.
DECLARED_CACHED_LENGTHS = (7, 19)

#: The two requests' block rows. Neither is a prefix of the other and the second
#: is deliberately not one ascending run continuing the first, so a stack-wide
#: contiguous slice cannot serve both.
DECLARED_SPARSE_ROWS = ((0, 1), (4, 5))

#: The decode bucket A7 pads up to. Two real requests in a bucket of four leaves
#: two padded rows, which is the case the padding writer produces.
DECLARED_DECODE_BUCKET = 4

#: The prefill warmup's token count. The real value is a launch-shape bucket; the
#: only property the converter reads is that it EXCEEDS the decode threshold, which
#: is what makes the warmup step read as a prefill (``neuron_model_runner.py:4303``).
DECLARED_WARMUP_BUCKET = 4

#: The state-carrier keys, in the model's own declared spelling
#: (``model_fp8.py:4149``). The carrier is splatted into the layer, so an extra
#: key is a TypeError and a missing one is served as a default. RE-PINNED: the
#: fourth key is the sibling increment's, which landed by tip merge; the original
#: reading was the three-key set.
DECLARED_STATE_CARRIER_KEYS = {
    "conv_state",
    "recurrent_state",
    "is_prefill",
    "start_position",
}

#: The conv and recurrent state shapes per slot. Small, and their only
#: requirement is that the two differ so a mixed-up carrier is visible.
DECLARED_CONV_SHAPE = (4, 16)
DECLARED_RECURRENT_SHAPE = (2, 8, 8)


class VacuousControlError(AssertionError):
    """A control that cannot discriminate is a failure, never a pass."""


def _require_cpu_mode() -> None:
    """The environment is part of the acceptance command, never a fixture.

    The declared command sets ``VLLM_NEURON_CPU_MODE=1``. Reading it rather than
    setting it is what keeps the command the instrument: a file that set the
    variable itself would pass under a command that did not.
    """
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.fail(
            "this file's declared acceptance runs under VLLM_NEURON_CPU_MODE=1; "
            f"the environment carries {os.environ.get('VLLM_NEURON_CPU_MODE')!r}"
        )


# ---------------------------------------------------------------------------
# The world: a two-layer hybrid stack, one sparse layer and one linear layer.
# ---------------------------------------------------------------------------
def _text_config() -> SimpleNamespace:
    """Only the fields the converter reads off ``model.text_config``.

    Five of them: the two indexer widths, and the two head widths whose sum's
    inverse square root is the softmax scale the converter passes down. Anything
    else absent means a converter that grew a new read raises here.
    """
    return SimpleNamespace(
        index_kpool=DECLARED_INDEX_KPOOL,
        index_head_dim=DECLARED_INDEX_HEAD_DIM,
        qk_nope_head_dim=DECLARED_HEAD_SIZE,
        qk_rope_head_dim=0,
    )


def _banks() -> list[dict]:
    """The bank mapping ``bind_kv_cache`` leaves on the model, built its way.

    ONE SPARSE BANK AND ONE LINEAR BANK, because the stack this campaign ports is
    hybrid and the two families take different carriers. The sparse bank keeps
    both views the model keeps (``model_fp8.py:9129-9141``): the paged bank and
    its flattened sequence view, which at one KV head is the same slot order the
    runner's own slot mapping produces.
    """
    latent_bank = torch.zeros(
        (DECLARED_BLOCKS, DECLARED_KV_HEADS, DECLARED_PAGE_SIZE, DECLARED_HEAD_SIZE),
        dtype=torch.bfloat16,
    )
    sparse = {
        "name": "model.layers.0.self_attn",
        "layer_index": 0,
        "family": "self_attn",
        "latent_bank": latent_bank,
        "latent_cache": latent_bank.view(
            DECLARED_BLOCKS * DECLARED_PAGE_SIZE, DECLARED_KV_HEADS, DECLARED_HEAD_SIZE
        ),
        "blocks": DECLARED_BLOCKS,
        "block_size": DECLARED_PAGE_SIZE,
        "slots": DECLARED_BLOCKS * DECLARED_PAGE_SIZE,
        "head_size": DECLARED_HEAD_SIZE,
    }
    linear = {
        "name": "model.layers.1.attention",
        "layer_index": 1,
        "family": "linear_attn",
        "state_slots": DECLARED_STATE_SLOTS,
        "conv_state": torch.zeros(
            (DECLARED_STATE_SLOTS, *DECLARED_CONV_SHAPE), dtype=torch.bfloat16
        ),
        "recurrent_state": torch.zeros(
            (DECLARED_STATE_SLOTS, *DECLARED_RECURRENT_SHAPE), dtype=torch.bfloat16
        ),
    }
    return [sparse, linear]


def _runner(banks, *, request_ids=None) -> NeuronModelRunner:
    """A runner carrying ONLY what the converter reads.

    ``input_batch.req_ids`` is here because the request identity the slot table
    keys on is the runner's own batch-ordered list, paired with block-table rows
    by request index (``neuron_model_runner.py:1126``, ``:2108``). Building the
    object with ``__new__`` keeps every other attribute absent.

    ``request_ids`` overrides that list, and the EMPTY list is a real shape rather
    than a test contrivance: at warmup the runner's input batch holds no request
    (``initialize_kv_cache`` builds it before anything is scheduled), which is the
    state the warmup arm below drives.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=_text_config(), glm5next_layer_banks=tuple(banks)
    )
    runner.max_model_len = DECLARED_BLOCKS * DECLARED_PAGE_SIZE
    # THE CONCURRENCY BOUND IS THE SLOT AXIS, so the harness carries it: a runner
    # without it raises here rather than letting the code fall back to a geometry.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    if request_ids is None:
        request_ids = [f"req-{index}" for index in range(DECLARED_REQUESTS)]
    runner.input_batch = SimpleNamespace(req_ids=list(request_ids))
    return runner


def _entry(*, rows, tokens: int, cached, block_size: int) -> dict:
    """ONE KV-cache group's attention-metadata entry, at this step's geometry.

    THE KEYS ARE THE RUNNER'S OWN, read off the mapping it builds
    (``neuron_model_runner.py:4417-4429``). ``rows`` carries one block row PER
    REQUEST and ``cached`` one cached length per request, which is the only
    difference from the landed single-request helpers this shape comes from.

    THE GEOMETRY THE CONVERTER READS IS THE HOST-SIDE PAIR, not the device tensors
    beside it: ``host_block_table`` and ``host_num_computed_tokens`` are the runner's
    own host arrays, and the converter refuses a device tensor for either by name
    because a captured graph cannot read a value off one. The device tensors stay in
    the entry, since the entry is the runner's whole mapping and the layers do consume
    them; what changed is which of the two pairs this step's geometry comes from.
    """
    table = torch.tensor([[int(value) for value in row] for row in rows], dtype=torch.int32)
    lengths = torch.tensor([int(value) for value in cached], dtype=torch.int32)
    return {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": _physical_slots(rows=rows, cached=cached, tokens=tokens,
                                        block_size=block_size),
        "max_query_len": int(tokens),
        "block_size": int(block_size),
        "max_blocks_per_seq": int(table.shape[1]),
        "decode_token_threshold": DECLARED_DECODE_THRESHOLD,
        "cached_seq_len": lengths,
        "host_block_table": [[int(value) for value in row] for row in rows],
        "host_num_computed_tokens": [int(value) for value in cached],
        "kv_segment_size": int(table.shape[1]) * int(block_size),
    }


def _physical_slots(*, rows, cached, tokens: int, block_size: int,
                    padded_rows: int = 0) -> torch.Tensor:
    """The physical slot per token, derived the way the runner derives it.

    THE FORMULA IS THE RUNNER'S, not this file's: at one context-parallel rank it
    is ``block_number * block_size + block_offset``
    (``neuron_model_runner.py:330-334``). Padded rows carry the padding value the
    runner writes (``:3571-3574``), which is zero and therefore a real slot.
    """
    slots: list[int] = []
    for row, start in zip(rows, cached):
        for offset in range(int(tokens)):
            position = int(start) + offset
            block = int(row[position // int(block_size)])
            slots.append(block * int(block_size) + position % int(block_size))
    slots.extend([NULL_BLOCK_ID] * int(padded_rows))
    return torch.tensor(slots, dtype=torch.int32)


def _metadata(banks, *, tokens: int, sparse_rows, state_rows, cached) -> dict:
    """TWO KV-cache groups, each with its own table, keyed by every layer name.

    This is the mapping a hybrid stack produces: one entry per group written under
    every layer name of that group (``neuron_model_runner.py:4256-4257``). The two
    groups carry different rows on purpose, because a converter reading one entry
    for the whole stack would slice one family out of the other's table.
    """
    sparse = _entry(rows=sparse_rows, tokens=tokens, cached=cached,
                    block_size=DECLARED_PAGE_SIZE)
    state = _entry(rows=state_rows, tokens=tokens, cached=cached,
                   block_size=DECLARED_PAGE_SIZE)
    return {
        str(bank["name"]): (sparse if bank["family"] == "self_attn" else state)
        for bank in banks
    }


def _generic(*, tokens: int, metadata: dict) -> dict:
    """One step's generic runner kwargs at a given metadata mapping.

    The ids are zeros of the right length: the converter reads
    ``input_ids.shape[0]`` and nothing else off them.
    """
    return {
        "input_ids": torch.zeros(int(tokens), dtype=torch.long),
        "positions": torch.arange(int(tokens), dtype=torch.long),
        "attn_metadata": metadata,
        "sampling_positions": torch.tensor([int(tokens) - 1], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }


def _two_request_step(runner, banks, *, tokens: int, cached):
    """One step carrying both requests, converted by the code under test.

    Returned as the converter's own output, because every item below reads what
    the runner decided to hand the layers and never builds a carrier itself.
    """
    metadata = _metadata(
        banks,
        tokens=int(tokens),
        sparse_rows=DECLARED_SPARSE_ROWS,
        state_rows=DECLARED_SPARSE_ROWS,
        cached=cached,
    )
    converted = runner._glm5next_model_kwargs(_generic(
        tokens=int(tokens), metadata=metadata
    ))
    assert sorted(converted) == ["input_ids", "layer_carriers", "sampling_positions"], (
        f"the converter returned {sorted(converted)}"
    )
    return converted["layer_carriers"]


def _linear_banks(banks) -> list[dict]:
    """The recurrent banks, which are the ones a state slot indexes."""
    return [bank for bank in banks if bank["family"] != "self_attn"]


def _side_caches(banks):
    """The indexer's two caches, one set per sparse layer with a slot axis.

    THE AXIS HERE IS DECLARED WIDER THAN THE ENGINE'S BOUND, on purpose: items that
    build carriers directly place two requests at NON-ADJACENT slots, which a
    two-slot axis could not hold. Which axis the RUNNER chooses is a separate
    property, read from the runner's own allocator by the item that asks it.
    """
    return NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=DECLARED_INDEX_KPOOL,
        index_head_dim=DECLARED_INDEX_HEAD_DIM,
        max_seq_len=DECLARED_BLOCKS * DECLARED_PAGE_SIZE,
        request_slots=DECLARED_STATE_SLOTS,
    )


def _carriers_for(banks, side, *, slot: int, rows, cached: int, is_prefill: bool):
    """One request's carriers at a given slot, built by the code under test.

    The window's length is this request's own row count, which is the slice the
    builder handed back before the length became a required key, so the items
    below read what they read before it did.
    """
    ids = [int(value) for value in rows]
    geometries = [
        {
            "block_ids": ids,
            "state_slot": int(slot),
            "page_size": DECLARED_PAGE_SIZE,
            "window_blocks": len(ids),
        }
        for _ in banks
    ]
    return NeuronModelRunner._glm5next_layer_carriers(
        banks,
        side,
        geometries=geometries,
        is_prefill=is_prefill,
        tokens=1 if not is_prefill else 4,
        start_position=int(cached),
        softmax_scale=float(DECLARED_HEAD_SIZE) ** -0.5,
        max_seq_len=int(cached) + 4,
        index_kpool=DECLARED_INDEX_KPOOL,
    )


def _carriers_for_requests(banks, side, *, tokens: int, requests: int, is_prefill: bool):
    """The carrier builder at a declared request count, for the refusal items.

    The geometry is one entry per bank, which is what the builder still takes at
    this stage of the increment; the request COUNT is what the sharpened decode
    refusal reads, and it is passed explicitly rather than inferred from the token
    count -- inferring it is exactly the conflation the refusal used to make.

    The window's length is the row count, as above: these items must reach the
    request-count refusal, and a geometry the walk turns away first would let them
    pass on a message they never asked for.
    """
    ids = [int(value) for value in DECLARED_SPARSE_ROWS[0]]
    geometries = [
        {
            "block_ids": ids,
            "state_slot": 0,
            "page_size": DECLARED_PAGE_SIZE,
            "window_blocks": len(ids),
        }
        for _ in banks
    ]
    return NeuronModelRunner._glm5next_layer_carriers(
        banks,
        side,
        geometries=geometries,
        is_prefill=is_prefill,
        tokens=int(tokens),
        start_position=int(DECLARED_CACHED_LENGTHS[0]),
        softmax_scale=float(DECLARED_HEAD_SIZE) ** -0.5,
        max_seq_len=int(DECLARED_CACHED_LENGTHS[0]) + int(tokens),
        index_kpool=DECLARED_INDEX_KPOOL,
        requests=int(requests),
    )


# ══════════════════════════════════════════════════════════════════════════════
# A3. One request's indexer state is untouched by another request's step.
# ══════════════════════════════════════════════════════════════════════════════


def test_a3_one_requests_side_caches_are_untouched_by_the_others_step() -> None:
    """The pooled store and the tail ring belong to a request, not to the process.

    THE TRIPWIRE IS THE LANDED BEHAVIOUR ITSELF. Before this increment both caches
    were one set per layer for the whole process, so the two carriers below were the
    SAME storage and a write through one was visible through the other. This item
    fails on that arrangement twice over: the disjoint-storage assertion and the
    byte-equality read after the write.

    BOTH LEGS ARE BUILT, because the ring reaches the layer under a different
    keyword on each (``prefill_tail`` when prefilling, ``tail`` when decoding) and a
    slot applied on only one of them would leak on the other.
    """
    _require_cpu_mode()
    banks = _banks()
    side = _side_caches(banks)
    sparse = [index for index, bank in enumerate(banks) if bank["family"] == "self_attn"]
    if not sparse:
        raise VacuousControlError(
            "this item needs a sparse layer, which is the only family holding side "
            "caches; the harness built none"
        )

    for leg, ring_key in ((True, "prefill_tail"), (False, "tail")):
        mine = _carriers_for(
            banks, side, slot=0, rows=DECLARED_SPARSE_ROWS[0],
            cached=DECLARED_CACHED_LENGTHS[0], is_prefill=leg,
        )
        theirs = _carriers_for(
            banks, side, slot=1, rows=DECLARED_SPARSE_ROWS[1],
            cached=DECLARED_CACHED_LENGTHS[1], is_prefill=leg,
        )
        for index in sparse:
            a, b = mine[index], theirs[index]
            print(f"KEYED|a3|leg={'prefill' if leg else 'decode'}|layer={index}"
                  f"|pool_a={a['pool_cache'].data_ptr()}"
                  f"|pool_b={b['pool_cache'].data_ptr()}")
            assert a["pool_cache"].data_ptr() != b["pool_cache"].data_ptr(), (
                f"layer {index}: two requests were handed ONE pooled store, so the "
                f"second would pool into the first's rows"
            )
            assert a[ring_key].data_ptr() != b[ring_key].data_ptr(), (
                f"layer {index}: two requests were handed ONE tail ring under "
                f"{ring_key!r}"
            )

            # THE WRITE, AND THE READ THAT PROVES IT DID NOT TRAVEL.
            untouched = b["pool_cache"].clone()
            ring_untouched = b[ring_key].clone()
            a["pool_cache"].fill_(1.5)
            a[ring_key].fill_(-1.5)
            assert torch.equal(b["pool_cache"], untouched), (
                f"layer {index}: writing request A's pooled store changed request "
                f"B's rows"
            )
            assert torch.equal(b[ring_key], ring_untouched), (
                f"layer {index}: writing request A's ring changed request B's ring"
            )
            if not bool(a["pool_cache"].any()):
                raise VacuousControlError(
                    "the write this item relies on left request A's pooled store at "
                    "zero, so the comparison above proves nothing"
                )


# ══════════════════════════════════════════════════════════════════════════════
# A1 (slot half). Two requests never share a state slot.
# ══════════════════════════════════════════════════════════════════════════════


def test_a1_two_requests_are_assigned_distinct_state_slots() -> None:
    """The slot is the request's identity, not the block table's first block id.

    THIS IS THE SLOT HALF OF A1, and it asserts what the runner alone decides; the
    carrier-view half is the item below it.

    THE TRIPWIRE: a table that hands both requests one slot. The assertion is that
    the two slots DIFFER, so such a table fails here rather than downstream where
    one request's recurrence would silently continue the other's.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    request_ids = list(runner.input_batch.req_ids)
    assert len(request_ids) == DECLARED_REQUESTS, (
        f"this item needs {DECLARED_REQUESTS} requests to have a collision to rule "
        f"out; the harness offered {len(request_ids)}"
    )

    slots = runner._glm5next_request_slots(banks, request_ids, synthetic=False)

    print(f"KEYED|a1|requests={request_ids}|slots={slots}")
    assert len(slots) == len(request_ids)
    assert len(set(slots)) == len(slots), (
        f"the two requests were handed slots {slots}; two live requests sharing a "
        f"slot means one continues the other's recurrence"
    )
    # RE-PINNED: the range a slot must lie in. ORIGINAL READING
    # ``0 <= slot < DECLARED_STATE_SLOTS``, the bank's slot count. NEW VALUE the
    # engine's concurrency bound, which is the axis the per-sequence caches carry: a
    # slot at or above it would index those caches out of range.
    for slot in slots:
        assert 0 <= slot < DECLARED_MAX_NUM_SEQS, (
            f"slot {slot} is outside the {DECLARED_MAX_NUM_SEQS} keyed sequence slot(s)"
        )
    # The mapping is STABLE: asking again inside one request's life returns the
    # same slots, because a slot that moved would abandon the state it holds.
    again = runner._glm5next_request_slots(banks, request_ids, synthetic=False)
    assert again == slots, f"the slots moved from {slots} to {again} within one life"


def test_a1_each_requests_carrier_is_a_view_of_its_own_bank_row() -> None:
    """The carrier half of A1: two requests, two views, and the writes land in the bank.

    WHAT THE SLOT HALF ABOVE CANNOT SHOW. A table that hands out two different
    slot NUMBERS is still wrong if the carrier then hands both requests the same
    tensor, or hands out copies. So this reads the carrier the runner built: each
    request's entry must be a storage view of ITS OWN bank row, the two entries
    must be different storage, and each request's declared position must travel
    with it.

    THE TRIPWIRES, three, each on a different wrong implementation. A table
    returning one slot for both fails the different-storage read. A carrier built
    from one request's slot for the whole batch fails the per-request pointer read.
    A carrier that COPIED the rows passes both pointer reads by accident, so the
    last conjunct writes through the second request's carrier and requires the
    BANK's own row to change while the first request's row does not -- a copy
    leaves the bank untouched and fails there.

    ONE FAMILY, DELIBERATELY. The sparse family still takes one contiguous slice of
    the paged latent bank, so a second request cannot be expressed in its carrier
    at all; the arm below reads that refusal by name. This item is the linear
    family's, which is the half that becomes concurrent here.
    """
    _require_cpu_mode()
    banks = _linear_banks(_banks())
    bank = banks[0]
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=DECLARED_INDEX_KPOOL,
        index_head_dim=DECLARED_INDEX_HEAD_DIM,
        max_seq_len=DECLARED_BLOCKS * DECLARED_PAGE_SIZE,
        request_slots=DECLARED_STATE_SLOTS,
    )
    slots = [0, 2]
    if len(set(slots)) != DECLARED_REQUESTS:
        raise VacuousControlError(
            f"this item needs {DECLARED_REQUESTS} distinct slots to tell the two "
            f"requests' storage apart; it declares {slots}"
        )
    geometries = [
        {
            "block_ids": [int(value) for value in DECLARED_SPARSE_ROWS[0]],
            "state_slot": slots[0],
            "state_slots": slots,
            "page_size": DECLARED_PAGE_SIZE,
        }
        for _ in banks
    ]

    carriers = NeuronModelRunner._glm5next_layer_carriers(
        banks,
        side,
        geometries=geometries,
        is_prefill=False,
        tokens=DECLARED_REQUESTS,
        start_position=DECLARED_CACHED_LENGTHS[0],
        softmax_scale=float(DECLARED_HEAD_SIZE) ** -0.5,
        max_seq_len=max(DECLARED_CACHED_LENGTHS) + 1,
        index_kpool=DECLARED_INDEX_KPOOL,
        requests=DECLARED_REQUESTS,
        request_starts=list(DECLARED_CACHED_LENGTHS),
    )

    carrier = carriers[0]
    pointers = [value.data_ptr() for value in carrier["conv_state"]]
    print(f"KEYED|a1|keys={sorted(carrier)}|slots={slots}"
          f"|positions={carrier['start_position']}|conv_pointers={pointers}")
    assert set(carrier) == DECLARED_STATE_CARRIER_KEYS, (
        f"the linear carrier holds {sorted(carrier)}, not "
        f"{sorted(DECLARED_STATE_CARRIER_KEYS)}; the layer takes these as keywords, "
        f"so an extra key raises and a missing one is served as a default"
    )
    assert len(carrier["conv_state"]) == DECLARED_REQUESTS, (
        f"the carrier holds {len(carrier['conv_state'])} conv entry(ies) for "
        f"{DECLARED_REQUESTS} requests"
    )
    # THE POSITIONS ARE ONE int32 TENSOR WITH A ROW PER REQUEST, compared as a tensor:
    # a python int at this boundary is baked into the graph it was captured with, and a
    # tuple of them is that defect once per request.
    assert carrier["start_position"].dtype == torch.int32, (
        f"the carrier's positions are {carrier['start_position'].dtype}, not int32"
    )
    assert torch.equal(
        carrier["start_position"],
        torch.tensor(DECLARED_CACHED_LENGTHS, dtype=torch.int32),
    ), (
        f"the carrier carries positions {carrier['start_position'].tolist()} rather "
        f"than each request's own {list(DECLARED_CACHED_LENGTHS)}; one position for the "
        f"batch would enter the second request's recurrence at the first one's point"
    )
    for index, slot in enumerate(slots):
        assert carrier["conv_state"][index].data_ptr() == (
            bank["conv_state"][slot].data_ptr()
        ), f"request {index}'s conv entry is not slot {slot}'s own row"
        assert carrier["recurrent_state"][index].data_ptr() == (
            bank["recurrent_state"][slot].data_ptr()
        ), f"request {index}'s recurrent entry is not slot {slot}'s own row"
    assert len(set(pointers)) == DECLARED_REQUESTS, (
        f"the two requests' conv entries share storage {pointers}; one request's "
        f"advance would then be the other's entering state"
    )

    # ---- THE VIEW READ, which a copy cannot pass: write through the carrier.
    untouched = bank["recurrent_state"][slots[0]].clone()
    carrier["recurrent_state"][1].fill_(3.0)
    landed = float(bank["recurrent_state"][slots[1]].abs().max())
    print(f"KEYED|a1|wrote_through_request_1|bank_row_max={landed}")
    assert landed == 3.0, (
        "a write through the second request's carrier did not reach its bank row, so "
        "the carrier is a copy and the layer's in-place advance would be discarded"
    )
    assert torch.equal(bank["recurrent_state"][slots[0]], untouched), (
        "the write reached the FIRST request's row as well; the two requests' states "
        "must be disjoint or one continues the other's sequence"
    )


def test_a1_a_two_request_linear_batch_is_served_through_the_converter() -> None:
    """The whole runner path, not the builder: two requests in one batch, end to end.

    WHAT THE BUILDER-LEVEL ITEM ABOVE CANNOT REACH. The converter is what reads the
    block tables, pairs identity with paging, classifies each request's position and
    records what each one advanced to. So this drives it twice on a LINEAR stack --
    an opening prefill, then a decode carrying one token per request -- and reads the
    carrier and the recorded positions afterwards.

    THE TRIPWIRE IS THE PREVIOUS COMMIT. Its walk refused any block table with more
    than one row, so this item raises there instead of reading a carrier; the refusal
    now belongs to the sparse family, which is the arm below.

    THE TWO REQUESTS OPEN AT THE SAME LENGTH, and that is a limit of the CONVERTER
    path rather than a choice: opening two sequences at different lengths needs one
    request fresh beside one continuing, which is the mixed batch this block refuses
    by design. The per-request POSITIONS at different lengths are read at the builder
    in the item above, where they can be declared directly.
    """
    _require_cpu_mode()
    banks = _linear_banks(_banks())
    runner = _runner(banks)
    request_ids = list(runner.input_batch.req_ids)
    opening = DECLARED_CACHED_LENGTHS[0]

    _two_request_step(
        runner, banks, tokens=opening * DECLARED_REQUESTS, cached=(0, 0)
    )
    carriers = _two_request_step(
        runner, banks, tokens=DECLARED_REQUESTS, cached=(opening,) * DECLARED_REQUESTS
    )

    carrier = carriers[0]
    table = runner._glm5next_request_slot_table
    slots = [table[request_id] for request_id in request_ids]
    positions = runner._glm5next_side_cache_positions
    print(f"KEYED|a1|converter_batch|slots={slots}|carrier_positions="
          f"{carrier['start_position'].tolist()}|recorded="
          f"{[positions[slot] for slot in slots]}")
    assert len(set(slots)) == DECLARED_REQUESTS, (
        f"the converter seated both requests at {slots}"
    )
    assert torch.equal(
        carrier["start_position"],
        torch.tensor([opening] * DECLARED_REQUESTS, dtype=torch.int32),
    ), (
        f"the carrier carries positions {carrier['start_position'].tolist()} for a "
        f"decode at {opening}"
    )
    for index, slot in enumerate(slots):
        assert carrier["conv_state"][index].data_ptr() == (
            banks[0]["conv_state"][slot].data_ptr()
        ), f"request {index}'s conv entry is not the slot the converter assigned it"
    for slot in slots:
        assert int(positions[slot]) == opening + 1, (
            f"slot {slot}'s ring records {positions[slot]} after a one-token decode "
            f"at {opening}; each request advances by ITS OWN tokens, and recording "
            f"the batch's total would refuse that request's next step"
        )


def test_a1_the_sparse_family_refuses_a_second_request_by_name() -> None:
    """The sparse carrier is one contiguous slice, so it says so instead of guessing.

    THE TRIPWIRE: a builder that silently served the first request's slice for a
    two-request batch would hand both requests one sequence's latents. The refusal
    names the paged gather as what lifts it, so the boundary is readable at the
    failure rather than only in the plan.
    """
    _require_cpu_mode()
    banks = _banks()
    side = _side_caches(banks)
    geometries = [
        {
            "block_ids": [int(value) for value in DECLARED_SPARSE_ROWS[0]],
            "state_slot": 0,
            "state_slots": [0, 2],
            "page_size": DECLARED_PAGE_SIZE,
        }
        for _ in banks
    ]

    with pytest.raises(ValueError, match="ONE contiguous slice"):
        NeuronModelRunner._glm5next_layer_carriers(
            banks,
            side,
            geometries=geometries,
            is_prefill=False,
            tokens=DECLARED_REQUESTS,
            start_position=DECLARED_CACHED_LENGTHS[0],
            softmax_scale=float(DECLARED_HEAD_SIZE) ** -0.5,
            max_seq_len=max(DECLARED_CACHED_LENGTHS) + 1,
            index_kpool=DECLARED_INDEX_KPOOL,
            requests=DECLARED_REQUESTS,
            request_starts=list(DECLARED_CACHED_LENGTHS),
        )


def test_a1_the_converter_hands_a_two_request_batch_to_the_sparse_refusal() -> None:
    """On a HYBRID stack the batch is admitted, walked, and refused where the limit is.

    THE POINT OF THE ARM. The refusal used to stand in the bank walk, where it read
    every family's table and said the whole forward threads one sequence. That is no
    longer true: the linear family is concurrent. So the batch now travels through the
    walk, the identity pairing and the position arms, and meets the refusal at the
    SPARSE carrier, which names the contiguous latent slice and the increment that
    lifts it.

    THE TRIPWIRE: a builder that served the first request's slice for the whole batch
    would hand both requests one sequence's latents, silently.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)

    with pytest.raises(ValueError, match="ONE contiguous slice"):
        _two_request_step(
            runner,
            banks,
            tokens=DECLARED_CACHED_LENGTHS[0] * DECLARED_REQUESTS,
            cached=(0, 0),
        )


# ══════════════════════════════════════════════════════════════════════════════
# A2. A finished request's slot is freed, and its next owner gets it zeroed.
# ══════════════════════════════════════════════════════════════════════════════


def test_a2_a_finished_requests_slot_is_reused_and_zeroed_on_hand_out() -> None:
    """Freed, reused, and ZEROED at the moment ownership changes.

    THE TWO TRIPWIRES, BOTH ASSERTED RATHER THAN DESCRIBED. A table that never
    frees cannot seat the later request at all, so the reuse assertion fails on a
    raised refusal. A table that frees WITHOUT zeroing hands the new owner the last
    one's recurrence, which the zero read fails on -- and the read is only
    meaningful because this item dirties the rows first, which it asserts it did.

    RE-PINNED: WHAT MAKES THE FIRST REQUEST FINISHED. The runner is now TOLD, with
    the engine's own finished set, and this item tells it. The reading this replaces,
    verbatim: "A LATER BATCH WITHOUT req-0 IS req-0 FINISHING. Nothing else is told."
    Absence from a batch is not finishing -- the item below drives that case -- so the
    old form would now leave the slot held and this item would fail on the reuse read.
    The property under test did not move: a finished request's slot is reused, and it
    is zeroed at hand-out.

    RE-PINNED AGAIN, AND THE TWO HALVES PART HERE. The INDEXER's two caches are still
    emptied at hand-out, because the runner allocates them and no reader of theirs
    takes a position. The RECURRENT BANKS are not: they are the engine's own cache
    tensors, every one a view of a single allocation, and an eager write on such a
    buffer is refused by the runtime wherever it sits. Their freshness is served where
    the state is READ -- an opening prefill selects a zero state, which
    `test_kda_prefill_segments_116.py`'s fresh-leg pair measures live, in both
    directions and on the write-back too. So what this item reads of the banks is that
    the hand-out left the previous owner's bytes ALONE.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    side = _side_caches(banks)
    rings = [entry for entry in side if entry]
    if not rings:
        raise VacuousControlError(
            "no bank in this stack carries indexer side caches, so the second half of "
            "this item -- that they are zeroed at the same moment -- measures nothing"
        )
    first = ["req-0"]

    seated = runner._glm5next_request_slots(
        banks, first, synthetic=False, side_caches=side
    )
    slot = seated[0]

    # DIRTY THE SLOT, and prove it is dirty. Without this the zero read below
    # would pass against a slot that was never written. BOTH the recurrent state and
    # the indexer's two caches are dirtied, because a HALF-fresh slot -- fresh
    # recurrence beside a stale pool -- is the defect the joint zeroing closes.
    for bank in _linear_banks(banks):
        bank["conv_state"][slot].fill_(3.0)
        bank["recurrent_state"][slot].fill_(-2.0)
    # WHAT THE PREVIOUS OWNER LEFT, kept to compare against rather than re-typed as a
    # literal below: the banks must come out of the hand-out holding exactly this.
    planted = [
        (bank["conv_state"][slot].clone(), bank["recurrent_state"][slot].clone())
        for bank in _linear_banks(banks)
    ]
    for entry in rings:
        entry["pool_cache"][slot].fill_(5.0)
        entry["tail"][slot].fill_(-7.0)
    dirtied = [
        bool(bank["conv_state"][slot].any()) and bool(bank["recurrent_state"][slot].any())
        for bank in _linear_banks(banks)
    ] + [
        bool(entry["pool_cache"][slot].any()) and bool(entry["tail"][slot].any())
        for entry in rings
    ]
    print(f"KEYED|a2|slot={slot}|dirtied={dirtied}")
    if not all(dirtied):
        raise VacuousControlError(
            "this item needs the freed slot to hold non-zero state before hand-out, "
            f"and the banks read {dirtied}"
        )

    # req-0 FINISHES, and the engine's finished set is what says so.
    runner._glm5next_note_finished_requests(["req-0"])
    later = runner._glm5next_request_slots(
        banks, ["req-2"], synthetic=False, side_caches=side
    )

    print(f"KEYED|a2|reused={later}|freed_slot={slot}")
    assert later == [slot], (
        f"the finished request's slot {slot} was not handed to the next request, "
        f"which got {later}; a table that never frees leaks its slots"
    )
    # RE-PINNED: THE RECURRENT BANKS ARE NOT WRITTEN AT HAND-OUT, and this reads that
    # they are not. They are the engine's own cache tensors, every one a view of a
    # single allocation, so an eager write on them is refused by the runtime; the
    # freshness is served where the state is READ instead, and the layer item that
    # measures it is `test_kda_prefill_segments_116.py`'s fresh-leg pair. What this
    # item still owns is that the slot changes hands without a write. The original
    # readings, verbatim: "assert not bank["conv_state"][slot].any(), ... still holds
    # the previous request's values on hand-out" and the same for "recurrent_state".
    for bank, (was_conv, was_recurrent) in zip(_linear_banks(banks), planted):
        assert torch.equal(bank["conv_state"][slot], was_conv), (
            f"bank {bank['name']}'s conv state at slot {slot} was written at hand-out; "
            f"the banks are the engine's buffers and no eager write may reach them"
        )
        assert torch.equal(bank["recurrent_state"][slot], was_recurrent), (
            f"bank {bank['name']}'s recurrent state at slot {slot} was written at "
            f"hand-out; the banks are the engine's buffers and no eager write may "
            f"reach them"
        )
    # THE INDEXER'S TWO CACHES ARE ZEROED AT THE SAME MOMENT. A slot handed over with
    # a fresh recurrence and the last owner's pooled keys would complete its next
    # pool from another request's members, with nothing shaped wrongly.
    for entry in rings:
        assert not entry["pool_cache"][slot].any(), (
            f"the pooled store at slot {slot} still holds the previous owner's rows "
            f"on hand-out, while its recurrent state was cleared"
        )
        assert not entry["tail"][slot].any(), (
            f"the tail ring at slot {slot} still holds the previous owner's rows on "
            f"hand-out, while its recurrent state was cleared"
        )


def test_a2_more_live_requests_than_slots_refuses_by_name() -> None:
    """The keyed axis is a fixed size, so an over-admission refuses instead of colliding.

    Called at the helper rather than through the converter on purpose: an
    over-admission needs MORE live requests than the axis keys, and the
    converter's own batch is bounded by the pinned launch shape, so the case is
    only reachable here. The refusal is matched on its own message, not on the
    exception type alone.

    RE-PINNED: the count that bounds it. ORIGINAL READING ``DECLARED_STATE_SLOTS + 1``,
    the bank's slot count plus one. NEW VALUE ``DECLARED_MAX_NUM_SEQS + 1``: the axis a
    request occupies is the engine's concurrency bound, so one request past THAT is what
    over-admission means. The bank still holds more slots than the bound, which is why
    the old count would no longer reach this refusal first.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    too_many = [f"req-{index}" for index in range(DECLARED_MAX_NUM_SEQS + 1)]

    print(f"KEYED|a2|requested={len(too_many)}|keyed={DECLARED_MAX_NUM_SEQS}"
          f"|bank_slots={DECLARED_STATE_SLOTS}")
    with pytest.raises(ValueError, match="has no free slot"):
        runner._glm5next_request_slots(banks, too_many, synthetic=False)


def test_a2_banks_holding_fewer_slots_than_the_bound_refuse_by_name() -> None:
    """Banks holding fewer slots than the engine admits sequences refuse by name.

    ONE slot number addresses the recurrent banks and the indexer's per-sequence
    caches together, so a stack whose banks hold fewer slots than the engine's
    concurrent-sequence bound has no slot at all for the last sequence. Without this
    refusal the capacity read returns the bound, the table hands out a slot number the
    banks cannot address, and the last sequence overwrites another sequence's state
    mid-serve. The case is reached by raising the runner's own bound past the banks'
    slot count, because the banks' geometry is what a real stack ships.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    banked = runner._glm5next_state_slot_count(banks)
    if banked <= 0:
        raise VacuousControlError(
            "this item raises the bound past the banks' own slot count, and the "
            "harness's banks report no recurrent state slot at all"
        )
    runner.max_num_reqs = banked + 1

    print(f"KEYED|a2|banked={banked}|bound={runner.max_num_reqs}")
    with pytest.raises(ValueError, match="no slot to hand the last sequence"):
        runner._glm5next_request_slot_capacity(banks)


def test_a2_a_stack_with_no_recurrent_bank_is_admitted_at_the_bound() -> None:
    """A stack holding no recurrent bank has nothing to provision, so it must be served.

    The refusal above compares the recurrent banks' slot count with the engine's bound.
    A stack of sparse layers alone reports NO recurrent slot, and that zero is an
    absence rather than an under-provisioned bank: there is no state to hand a slot, and
    the bound still sizes the per-sequence caches every sparse layer holds. A refusal on
    the zero takes the whole DSA-only stack out of service, which is the shape the tiny
    fixture ships and the shape a served DSA-only model has.
    """
    _require_cpu_mode()
    sparse_only = [bank for bank in _banks() if bank["family"] == "self_attn"]
    runner = _runner(sparse_only)
    banked = runner._glm5next_state_slot_count(sparse_only)
    if banked:
        raise VacuousControlError(
            f"this item drives a stack with no recurrent bank, and the harness's "
            f"sparse-only stack reports {banked} recurrent state slot(s)"
        )

    print(f"KEYED|a2|banked={banked}|bound={DECLARED_MAX_NUM_SEQS}")
    assert (
        runner._glm5next_request_slot_capacity(sparse_only) == DECLARED_MAX_NUM_SEQS
    ), "a stack with no recurrent bank was not admitted at the engine's own bound"


def test_a2_the_side_cache_slot_axis_is_the_engines_concurrency_bound() -> None:
    """The per-sequence caches are sized by ``max_num_seqs``, never by the block space.

    WHY THIS ITEM EXISTS. Both indexer caches hold ONE sequence's state, so their
    leading axis is how many sequences may hold state at once. The recurrent banks'
    leading dimension is a paging geometry -- one entry per KV block of the group -- and
    on the serving stack it is thousands of entries: a per-sequence cache sized by it
    allocates gigabytes per layer outside the engine's KV budget, at the first warmup
    step, on a rank whose memory is already committed.

    THE TRIPWIRE, AND WHY IT IS NOT A COINCIDENCE. The harness declares the two numbers
    DIFFERENT -- two sequences against four bank slots -- so an allocator reading either
    one is visible in the shape. The item asserts the axis equals the bound AND that it
    is not the bank's count, and it refuses to run if the harness ever makes the two
    equal, because then neither assertion could fail.

    THE SIZE IS READ, NOT RESTATED. The row count beside the slot axis is the
    indexer's own minimum from ``max_model_len``, which this item does not re-derive; it
    reads only which axis the slots ride on.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    if DECLARED_MAX_NUM_SEQS == DECLARED_STATE_SLOTS:
        raise VacuousControlError(
            "this item tells the concurrency bound from the bank's slot count by their "
            f"shapes, and the harness declares both as {DECLARED_MAX_NUM_SEQS}"
        )

    live = runner._glm5next_live_side_caches(banks)

    rings = [entry for entry in live if entry]
    if not rings:
        raise VacuousControlError(
            "no bank in this stack carries indexer side caches, so this item has no "
            "slot axis to read"
        )
    axes = [
        (int(entry["pool_cache"].shape[0]), int(entry["tail"].shape[0]))
        for entry in rings
    ]
    print(f"KEYED|a2|side_cache_axes={axes}|bound={DECLARED_MAX_NUM_SEQS}"
          f"|bank_slots={DECLARED_STATE_SLOTS}")
    for pool_axis, tail_axis in axes:
        assert pool_axis == DECLARED_MAX_NUM_SEQS, (
            f"the pooled store carries {pool_axis} slot(s) while the engine admits "
            f"{DECLARED_MAX_NUM_SEQS} concurrent sequence(s)"
        )
        assert tail_axis == DECLARED_MAX_NUM_SEQS, (
            f"the tail ring carries {tail_axis} slot(s) while the engine admits "
            f"{DECLARED_MAX_NUM_SEQS} concurrent sequence(s)"
        )
        assert pool_axis != DECLARED_STATE_SLOTS, (
            f"the pooled store carries one set per bank slot ({pool_axis}), which is "
            f"the block space and not a sequence count"
        )


def test_a2_a_request_skipped_for_one_step_keeps_its_slot_and_state() -> None:
    """Absence from a step's batch is not finishing, and the state must survive it.

    THE CASE. This runner removes a live request from its persistent batch when the
    scheduler gives it no tokens in a step, and that request keeps its cached state and
    comes back. A table that freed on absence would hand its slot away, zeroed, while
    the request was still alive; the request would then return to a slot holding
    somebody else's recurrence, or to a refusal with its own recurrence gone.

    THE TRIPWIRE, IN TWO HALVES. The skipped request must come back to the SAME slot
    with its planted state intact -- a free-on-absence table fails on both -- and the
    second half proves the item is not simply asserting that nothing is ever freed: the
    same request, once the engine calls it finished, releases the slot.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    both = list(runner.input_batch.req_ids)
    assert len(both) == DECLARED_REQUESTS

    seated = runner._glm5next_request_slots(banks, both, synthetic=False)
    skipped, kept = both[0], both[1]
    skipped_slot = seated[both.index(skipped)]
    for bank in _linear_banks(banks):
        bank["recurrent_state"][skipped_slot].fill_(4.0)
    planted = [
        bool(bank["recurrent_state"][skipped_slot].any()) for bank in _linear_banks(banks)
    ]
    if not all(planted):
        raise VacuousControlError(
            "the skipped request's state must be non-zero before the step that skips "
            f"it, and the banks read {planted}"
        )

    # THE STEP THAT SKIPS IT. Only the other request is scheduled, and the engine says
    # nobody finished.
    runner._glm5next_request_slots(banks, [kept], synthetic=False)
    returned = runner._glm5next_request_slots(banks, both, synthetic=False)

    survived = [
        bool(bank["recurrent_state"][skipped_slot].any()) for bank in _linear_banks(banks)
    ]
    print(f"KEYED|a2|skipped={skipped}|slot={skipped_slot}|returned={returned}"
          f"|state_survived={survived}")
    assert returned[both.index(skipped)] == skipped_slot, (
        f"request {skipped!r} was skipped for one step and came back to slot "
        f"{returned[both.index(skipped)]} instead of its own {skipped_slot}"
    )
    assert all(survived), (
        f"request {skipped!r}'s recurrent state was cleared by a step that merely did "
        f"not schedule it; the banks read {survived}"
    )

    # AND THE SLOT DOES FREE WHEN THE ENGINE SAYS SO, which is what keeps the half
    # above from passing on a table that never frees anything.
    runner._glm5next_note_finished_requests([skipped])
    runner._glm5next_request_slots(banks, [kept], synthetic=False)
    table = dict(runner._glm5next_request_slot_table)
    print(f"KEYED|a2|table_after_finish={table}")
    assert skipped not in table, (
        f"request {skipped!r} was finished by the engine and still owns slot "
        f"{table.get(skipped)}"
    )


def test_a2_a_synthetic_step_takes_no_claim() -> None:
    """Warmup is served without an identity and leaves the table untouched.

    THE TRIPWIRE: a synthetic step that seated itself would evict a live request
    on a busy bank, so this asserts the table is byte-identical across it AND that
    the step was still served with a usable slot.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    runner._glm5next_request_slots(banks, ["req-0"], synthetic=False)
    before = dict(runner._glm5next_request_slot_table)

    served = runner._glm5next_request_slots(banks, [None], synthetic=True)

    after = dict(runner._glm5next_request_slot_table)
    print(f"KEYED|a2|synthetic_served={served}|table_before={before}|after={after}")
    assert served == [0], f"a synthetic step was served slots {served}"
    assert after == before, (
        f"a synthetic step changed the slot table from {before} to {after}; warmup "
        f"is not a sequence step and must take no claim"
    )


def test_a2_the_prefill_warmups_own_shape_is_served_and_takes_no_claim() -> None:
    """The prefill warmup has no request, and it must still be served.

    THIS IS A REGRESSION ARM, and it is the one item in this file that PASSES at the
    base commit -- deliberately, because what it pins is the base's own behaviour.
    Review round 1 found the regression it exists to catch: the prefill warmup
    arrives as a PREFILL at position 0 with an input batch that names no request
    (``_build_prefill_synthetic_inputs`` builds ``cached_seq_len = 0``, one request
    and a whole bucket of tokens against ``decode_token_threshold = 1``), and the
    position rule covered only a decode at 0, so the converter asked for an identity
    that warmup cannot have and raised. The first bucket of ``warmup_prefill`` is
    unconditional, so a server would have died before READY.

    THE SHAPE IS DRIVEN THROUGH THE CONVERTER, not asserted about: this calls
    ``_glm5next_model_kwargs`` with the warmup shape, so the classification, the slot
    hand-out and the carrier build all run.

    THE TRIPWIRE, and it is a value rather than a hope: the step must be served from
    slot 0 -- read as a VIEW of the bank's slot 0, by storage -- and the slot table
    must still be empty afterwards. A converter that seated warmup would leave a key
    behind, and warmup between two real steps would then evict a live request.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks, request_ids=[])
    metadata = _metadata(
        banks,
        tokens=DECLARED_WARMUP_BUCKET,
        sparse_rows=(DECLARED_SPARSE_ROWS[0],),
        state_rows=(DECLARED_SPARSE_ROWS[0],),
        cached=(0,),
    )

    converted = runner._glm5next_model_kwargs(
        _generic(tokens=DECLARED_WARMUP_BUCKET, metadata=metadata)
    )

    carriers = converted["layer_carriers"]
    table = getattr(runner, "_glm5next_request_slot_table", None) or {}
    linear = _linear_banks(banks)[0]
    served = [
        index
        for index in range(DECLARED_STATE_SLOTS)
        if carriers[1]["conv_state"][0].data_ptr()
        == linear["conv_state"][index].data_ptr()
    ]
    print(f"KEYED|a2|warmup_tokens={DECLARED_WARMUP_BUCKET}|carriers={len(carriers)}"
          f"|served_slot={served}|table={table}")
    assert len(carriers) == len(banks), (
        f"the prefill warmup was served {len(carriers)} carrier(s) for "
        f"{len(banks)} bank(s)"
    )
    assert served == [0], (
        f"the prefill warmup's linear carrier is not slot 0's row; it matched "
        f"{served}, and a warmup served from a live request's slot would run over "
        f"that request's state"
    )
    assert table == {}, (
        f"the prefill warmup left {table} in the slot table; it is not a sequence "
        f"step and must take no claim, or a warmup would evict a live request"
    )


def _one_token_step(runner, banks):
    """One request's WHOLE one-token prompt, converted by the code under test.

    THE SHAPE IS THE ONE THE LEG TEST MISREADS. One token against a decode
    threshold of one is not a prefill by that test, and the request has computed
    nothing, so this is the step where the leg and the computed length disagree.
    """
    metadata = _metadata(
        banks,
        tokens=1,
        sparse_rows=(DECLARED_SPARSE_ROWS[0],),
        state_rows=(DECLARED_SPARSE_ROWS[0],),
        cached=(0,),
    )
    return runner._glm5next_model_kwargs(_generic(tokens=1, metadata=metadata))


def _held_and_opening(banks):
    """A runner whose batch names one opening request, with another already seated.

    THE SIDE CACHES ARE ALLOCATED FIRST, and the reason is the allocator's own: it
    discards the slot table along with the rows the table referred to, so a claim
    taken before the first allocation would be thrown away by the converter's own
    call. Allocating here means the converter finds this set and keeps the claim.
    """
    opening, held = "req-opening", "req-held"
    runner = _runner(banks, request_ids=[opening])
    live = runner._glm5next_live_side_caches(banks)
    held_slot = runner._glm5next_request_slots(banks, [held], synthetic=False)[0]
    if DECLARED_MAX_NUM_SEQS < 2:
        raise VacuousControlError(
            f"these items tell a request's own slot from slot 0 by their numbers, and "
            f"the harness admits {DECLARED_MAX_NUM_SEQS} concurrent sequence(s)"
        )
    if held_slot != 0:
        raise VacuousControlError(
            f"the already-seated request holds slot {held_slot}; these items read "
            f"whether the opening request was served from SLOT 0 instead of its own, "
            f"so slot 0 has to be the one that is already taken"
        )
    return runner, live, opening, held, held_slot


def test_a2_a_one_token_prompt_is_keyed_by_its_own_request_not_slot_zero() -> None:
    """A prompt of one token is an opening sequence, not a step without a request.

    WHY THIS IS THE STEP THAT DECIDES IT. Which requests a step serves used to be
    read from the LEG as well as the position: a step at position 0 that was not a
    prefill by the leg test was served as though the engine had scheduled nothing --
    from slot 0, with no claim taken. A whole one-token prompt is exactly that step,
    because one token does not exceed a decode threshold of one, and so is a request
    resumed after preemption. Both are real requests, and both were handed the state
    of whichever request holds slot 0.

    THE TRIPWIRE IS A SLOT NUMBER. Another request holds slot 0 before the step, so
    a converter that classified this one by its leg serves it from that request's row
    and leaves no claim of its own. This item reads the row the carriers actually
    bind, by storage, and the table afterwards.
    """
    _require_cpu_mode()
    banks = _banks()
    runner, _, opening, held, held_slot = _held_and_opening(banks)

    carriers = _one_token_step(runner, banks)["layer_carriers"]

    table = dict(runner._glm5next_request_slot_table)
    linear = _linear_banks(banks)[0]
    carrier = next(entry for entry in carriers if "conv_state" in entry)
    served = [
        index
        for index in range(DECLARED_STATE_SLOTS)
        if carrier["conv_state"][0].data_ptr()
        == linear["conv_state"][index].data_ptr()
    ]
    print(f"KEYED|a2|one_token_prompt_table={table}|served_slot={served}"
          f"|held_slot={held_slot}")
    assert table.get(opening) is not None, (
        f"the one-token prompt took no slot at all; the table holds {table}, so the "
        f"step was served as though the engine had scheduled nothing"
    )
    assert table[opening] != held_slot, (
        f"the one-token prompt was given slot {table[opening]}, which request "
        f"{held!r} already holds"
    )
    assert served == [table[opening]], (
        f"the one-token prompt's carrier binds slot(s) {served} and its own slot is "
        f"{table[opening]}; a step served from another slot reads and writes that "
        f"request's state"
    )
    assert table[held] == held_slot, (
        f"request {held!r} held slot {held_slot} before the step and holds "
        f"{table.get(held)} after it"
    )


def test_a2_a_one_token_prompt_opens_its_own_ring_instead_of_being_refused() -> None:
    """The opening arm is the position's, so this step opens a sequence here.

    WHY THIS ITEM EXISTS BESIDE THE ONE ABOVE. Keying the step by its request is
    only half of serving it: the indexer's ring is opened for a sequence that starts
    at position 0, and that arm read the LEG too. A one-token prompt reaching the
    other arm is refused for want of a cursor -- ``holds no sequence cursor`` -- so a
    legal request would have been turned away by name rather than served.

    THE TWO READINGS. This request's ring row must be emptied and its cursor must
    record the one token it consumed, while the ring row of the request holding slot
    0 keeps the value planted in it -- so an arm that emptied the whole set, or the
    wrong row, fails here.
    """
    _require_cpu_mode()
    banks = _banks()
    runner, live, opening, held, held_slot = _held_and_opening(banks)
    #: Any non-zero value stands for what the previous owner stashed; this one is
    #: exact in every dtype a ring can carry.
    planted_ring = 3.0
    rings = [entry for entry in live if entry and "tail" in entry]
    if not rings:
        raise VacuousControlError(
            "no bank in this stack carries an indexer ring, so this item has no "
            "opening to read"
        )
    for entry in rings:
        entry["tail"].fill_(planted_ring)

    carriers = _one_token_step(runner, banks)["layer_carriers"]
    assert len(carriers) == len(banks)

    table = dict(runner._glm5next_request_slot_table)
    own_slot = int(table[opening])
    emptied = [
        float(entry["tail"][own_slot].abs().max()) for entry in rings
    ]
    kept = [float(entry["tail"][held_slot].abs().max()) for entry in rings]
    stood_at = runner._glm5next_side_cache_positions.get(own_slot)
    print(f"KEYED|a2|one_token_prompt_opened_slot={own_slot}|ring_max_own={emptied}"
          f"|ring_max_held={kept}|cursor={stood_at}|planted={planted_ring}")
    assert all(value == 0.0 for value in emptied), (
        f"the one-token prompt's own ring row still holds {emptied}; a sequence "
        f"starting at position 0 must not inherit what the last owner stashed"
    )
    assert all(value == planted_ring for value in kept), (
        f"request {held!r}'s ring row reads {kept} rather than the planted "
        f"{planted_ring}; the opening emptied a row that is not its own"
    )
    assert stood_at == 1, (
        f"the one-token prompt's ring stands at {stood_at} after consuming its one "
        f"token; a step that opened no sequence records no cursor at all"
    )


def test_a2_a_step_the_engine_scheduled_cannot_be_served_without_its_identity() -> None:
    """The classification that has no legal caller, refused by name.

    NO CALLER REACHES THIS. A step is classified as having no request BY the absence
    of request ids, so the two can no longer disagree. The refusal stands for what it
    would cost to be wrong: a scheduled request served without its identity reads and
    writes slot 0 while its own slot stands untouched, which destroys the state of
    whoever holds slot 0 and answers this request out of it. The rule that read the
    LEG did exactly that to a one-token prompt, so this is the shape a later edit
    would bring back.

    THE CONTROL IS IN THE SAME ITEM, because a refusal that fired on every call
    would break warmup instead of protecting it: with no request in the batch the
    same call is served, which is the step warmup actually builds.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    scheduled = list(runner.input_batch.req_ids)
    if not scheduled:
        raise VacuousControlError(
            "this item needs a batch that names a request, and the harness built one "
            "that names none"
        )

    with pytest.raises(ValueError) as caught:
        runner._glm5next_request_identities(synthetic=True)

    message = str(caught.value)
    served = _runner(banks, request_ids=[])._glm5next_request_identities(
        synthetic=True
    )
    print(f"KEYED|a2|scheduled={scheduled}|refusal={message}|no_batch_served={served}")
    assert "classified as having no request" in message, (
        f"the call raised, but not the refusal this item names: {message}"
    )
    assert scheduled[0] in message, (
        f"the refusal names no request of the batch it refused: {message}"
    )
    assert served == [None], (
        f"a step whose batch names no request was refused too, and that is the step "
        f"warmup builds; the call returned {served}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# A4. The batch's per-token operands are the per-request derivations, in order.
# ══════════════════════════════════════════════════════════════════════════════


def test_a4_the_batch_operands_are_the_per_request_derivations_concatenated() -> None:
    """Two requests at DIFFERENT cached lengths, and each token gets its own.

    THE TRIPWIRE IS THE LANDED DERIVATION. Before this increment both operands came
    from ONE start position for the whole batch, so the second request's tokens
    carried the first request's positions. This item asserts the batch form equals
    the two per-request derivations concatenated AND that the single-start form
    differs from it, so a converter that kept the old derivation fails here.

    THE TWO CACHED LENGTHS DIFFER ON PURPOSE. At equal lengths the old and new
    derivations agree and this item would pass on either.
    """
    _require_cpu_mode()
    first, second = DECLARED_CACHED_LENGTHS
    if first == second:
        raise VacuousControlError(
            f"this item needs two different cached lengths to separate a per-request "
            f"derivation from a batch-wide one; both are {first}"
        )
    requests = [(1, first), (1, second)]
    device = torch.device("cpu")

    seq_lens = NeuronModelRunner._glm5next_batch_row_seq_lens(requests, device=device)
    slots = NeuronModelRunner._glm5next_batch_pool_slot_mapping(
        requests, index_kpool=DECLARED_INDEX_KPOOL, device=device
    )

    want_seq_lens = torch.cat([
        NeuronModelRunner._glm5next_row_seq_lens(
            tokens=tokens, start_position=start, device=device
        )
        for tokens, start in requests
    ])
    want_slots = torch.cat([
        NeuronModelRunner._glm5next_pool_slot_mapping(
            tokens=tokens, start_position=start,
            index_kpool=DECLARED_INDEX_KPOOL, device=device,
        )
        for tokens, start in requests
    ])
    print(f"KEYED|a4|requests={requests}|seq_lens={seq_lens.tolist()}"
          f"|pool_slots={slots.tolist()}")
    assert torch.equal(seq_lens, want_seq_lens), (
        f"the batch seq_lens {seq_lens.tolist()} are not the per-request "
        f"derivations concatenated {want_seq_lens.tolist()}"
    )
    assert torch.equal(slots, want_slots), (
        f"the batch pool slots {slots.tolist()} are not the per-request "
        f"derivations concatenated {want_slots.tolist()}"
    )

    # ---- THE OLD DERIVATION, RUN, so the item is proved to separate the two.
    total = sum(tokens for tokens, _ in requests)
    stale_seq_lens = NeuronModelRunner._glm5next_row_seq_lens(
        tokens=total, start_position=first, device=device
    )
    print(f"KEYED|a4|stale_batch_wide={stale_seq_lens.tolist()}")
    assert not torch.equal(seq_lens, stale_seq_lens), (
        f"a single-start-position derivation produced the same seq_lens "
        f"{stale_seq_lens.tolist()} as the per-request one, so this item is not "
        f"measuring the per-request derivation at all"
    )


# ══════════════════════════════════════════════════════════════════════════════
# A5 (mapping half). The runner's slot number and the bank's view are one address.
# ══════════════════════════════════════════════════════════════════════════════


def test_a5_the_runners_slot_mapping_addresses_the_banks_own_view() -> None:
    """The physical slot the KV machinery computes IS the bank view's row index.

    WHY THIS IS THE LOAD-BEARING PREMISE. This increment stops deriving the latent
    write's address from a contiguous block run and consumes the runner's own slot
    mapping instead. That is only correct if the number the mapping produces
    addresses the same element the bank's flattened sequence view does. The two are
    derived independently -- one in the KV machinery, one in the model's mapper --
    so the equality is measured here rather than assumed.

    THE PROOF IS A SENTINEL, NOT A RESTATED FORMULA. Comparing the two arithmetic
    expressions would only show that this file can copy a formula. Instead a value
    is written through the paged bank at ``[block, 0, offset]`` and read back
    through the flattened view at the mapping's slot; if they are not one element
    the read misses.

    THE TRIPWIRE: an off-by-one block stride or a view that interleaves heads makes
    the read-back differ, and the untouched-slot control catches a write that
    landed everywhere.
    """
    _require_cpu_mode()
    banks = _banks()
    sparse = next(bank for bank in banks if bank["family"] == "self_attn")
    page = DECLARED_PAGE_SIZE
    rows = DECLARED_SPARSE_ROWS
    starts = DECLARED_CACHED_LENGTHS

    # The KV machinery's own derivation, called rather than reproduced.
    table = np.array([[int(v) for v in row] for row in rows], dtype=np.int32)
    positions = np.array([int(start) for start in starts], dtype=np.int64)
    req_indices = np.arange(len(starts), dtype=np.int64)
    produced = np.zeros(len(starts), dtype=np.int64)
    _compute_slot_mapping_cpu(table, produced, positions, req_indices, page)

    print(f"KEYED|a5|rows={rows}|starts={starts}|slot_mapping={produced.tolist()}")
    for index, (row, start) in enumerate(zip(rows, starts)):
        block = int(row[int(start) // page])
        offset = int(start) % page
        slot = int(produced[index])

        sentinel = float(index + 1) * 0.5
        sparse["latent_bank"][block, 0, offset, :].fill_(sentinel)
        read_back = sparse["latent_cache"][slot, 0, :]
        assert bool((read_back == sentinel).all()), (
            f"request {index}: the KV machinery's slot {slot} and the bank view's "
            f"row for block {block} offset {offset} are not one element"
        )
        # THE CONTROL: the neighbouring slot must NOT have taken the write.
        neighbour = (slot + 1) % int(sparse["latent_cache"].shape[0])
        assert not bool((sparse["latent_cache"][neighbour, 0, :] == sentinel).all()), (
            f"request {index}: slot {neighbour} also holds the sentinel, so the "
            f"write landed wider than one slot and the read proves nothing"
        )


# ══════════════════════════════════════════════════════════════════════════════
# A6. A decode carries one token PER REQUEST, and more than that still refuses.
# ══════════════════════════════════════════════════════════════════════════════


def test_a6_a_decode_carrying_more_tokens_than_requests_refuses_by_name() -> None:
    """Speculative decoding's verify step is still out of scope, now stated exactly.

    WHAT THE SHARPENING FIXED. The landed refusal read "one token", which conflated
    one token PER REQUEST -- what a decode step is -- with one token in the batch,
    which only holds when the batch has one request. So the honest refusal is a step
    carrying MORE tokens than it has requests.

    TWO READINGS, because a refusal must be told apart from a silent skip. The
    message is matched on its own text, and the seam dispatch counters are read
    afterwards and must be exactly zero -- a refused step reaches no kernel. The
    slot table and the side caches are also compared byte-for-byte across the
    refused call: a refusal must leave no trace.

    THE ADMITTED CASE IS ASSERTED IN THE SAME ITEM, so this cannot pass by refusing
    everything: one token per request for two requests is built without raising.
    """
    _require_cpu_mode()
    from vllm_neuron.functional.attention.mla_sparse import (
        mla_sparse_dispatch_counters,
        mla_sparse_row_tiled_dispatch_counters,
        reset_mla_sparse_dispatch_counters,
        reset_mla_sparse_row_tiled_dispatch_counters,
    )

    banks = _banks()
    side = _side_caches(banks)
    runner = _runner(banks)
    runner._glm5next_request_slots(banks, list(runner.input_batch.req_ids),
                                  synthetic=False, side_caches=side)
    table_before = dict(runner._glm5next_request_slot_table)
    pools_before = [
        entry["pool_cache"].clone() for entry in side if entry
    ]

    reset_mla_sparse_dispatch_counters()
    reset_mla_sparse_row_tiled_dispatch_counters()

    over_by_one = DECLARED_REQUESTS + 1
    if over_by_one <= DECLARED_REQUESTS:
        raise VacuousControlError(
            "this item needs more tokens than requests to have anything to refuse"
        )
    print(f"KEYED|a6|requests={DECLARED_REQUESTS}|tokens={over_by_one}")
    with pytest.raises(ValueError, match="one token per request"):
        _carriers_for_requests(
            banks, side, tokens=over_by_one, requests=DECLARED_REQUESTS,
            is_prefill=False,
        )

    base = mla_sparse_dispatch_counters()
    row_tiled = mla_sparse_row_tiled_dispatch_counters()
    print(f"KEYED|a6|seam_counters={base}|row_tiled={row_tiled}")
    assert base == (0, 0), f"a refused step reached the seam: {base}"
    assert row_tiled == (0, 0), f"a refused step reached the row-tiled seam: {row_tiled}"
    assert dict(runner._glm5next_request_slot_table) == table_before, (
        "the refused step changed the slot table; a refusal must leave no trace"
    )
    for entry, before in zip([e for e in side if e], pools_before):
        assert torch.equal(entry["pool_cache"], before), (
            "the refused step wrote into a pooled store"
        )

    # ---- THE ADMITTED CASE, so the refusal is not simply refusing everything.
    admitted = _carriers_for_requests(
        banks, side, tokens=DECLARED_REQUESTS, requests=DECLARED_REQUESTS,
        is_prefill=False,
    )
    print(f"KEYED|a6|admitted_carriers={len(admitted)}")
    assert len(admitted) == len(banks)


# ══════════════════════════════════════════════════════════════════════════════
# A7. A padded decode row writes NOWHERE, because zero is a real slot.
# ══════════════════════════════════════════════════════════════════════════════


def test_a7_padded_decode_rows_carry_a_sentinel_and_never_slot_zero() -> None:
    """The KV machinery's padding value is an ADDRESS, so it is masked here.

    THE FACT THIS RESTS ON, read off the runner: ``NULL_BLOCK_ID`` is 0 and the
    decode padder writes it into padded rows. Zero is block 0 offset 0 -- a real,
    addressable latent slot -- so a write consuming the mapping unmasked lands a
    padded row's latent in a slot a live request can own.

    THE TRIPWIRE IS DEMONSTRATED, NOT DESCRIBED. This item asserts that the
    unmasked padding value IS inside the bank's addressable range (which is what
    makes it dangerous) and that every padded entry of the masked mapping is
    OUTSIDE it. Dropping the mask therefore fails the second assertion, and an
    implementation that masked the real rows too fails the first block below.

    THE RUNNER HALF ONLY. Observing the write itself belongs to the half of this
    increment that owns the writer; what is settled here is the address the writer
    is handed.
    """
    _require_cpu_mode()
    banks = _banks()
    sparse = next(bank for bank in banks if bank["family"] == "self_attn")
    addressable = int(sparse["latent_cache"].shape[0])
    padded_rows = DECLARED_DECODE_BUCKET - DECLARED_REQUESTS
    if padded_rows <= 0:
        raise VacuousControlError(
            f"this item needs a decode bucket wider than the request count to have a "
            f"padded row at all; bucket {DECLARED_DECODE_BUCKET} against "
            f"{DECLARED_REQUESTS} request(s)"
        )

    mapping = NeuronModelRunner._glm5next_latent_slot_mapping(
        rows=DECLARED_SPARSE_ROWS,
        starts=DECLARED_CACHED_LENGTHS,
        tokens=1,
        block_size=DECLARED_PAGE_SIZE,
        padded_rows=padded_rows,
        device=torch.device("cpu"),
    )

    print(f"KEYED|a7|mapping={mapping.tolist()}|padded_rows={padded_rows}"
          f"|null_block_id={NULL_BLOCK_ID}|addressable={addressable}")
    assert int(mapping.shape[0]) == DECLARED_REQUESTS + padded_rows

    # ---- The real rows address their own request's block, inside the bank.
    for index, (row, start) in enumerate(zip(DECLARED_SPARSE_ROWS, DECLARED_CACHED_LENGTHS)):
        block = int(row[int(start) // DECLARED_PAGE_SIZE])
        want = block * DECLARED_PAGE_SIZE + int(start) % DECLARED_PAGE_SIZE
        assert int(mapping[index]) == want, (
            f"request {index}'s token was mapped to slot {int(mapping[index])} rather "
            f"than block {block} offset {int(start) % DECLARED_PAGE_SIZE}"
        )
        assert 0 <= int(mapping[index]) < addressable

    # ---- WHY THE PADDING VALUE IS DANGEROUS, asserted rather than asserted about.
    assert 0 <= NULL_BLOCK_ID < addressable, (
        f"this item's whole premise is that the padding value {NULL_BLOCK_ID} is a "
        f"real slot; the bank holds {addressable} slot(s), so it is not, and the "
        f"mask this item measures would be unnecessary"
    )

    # ---- Every padded row is outside the bank, so it cannot address anything.
    for offset in range(padded_rows):
        value = int(mapping[DECLARED_REQUESTS + offset])
        assert value == PAD_SLOT_ID, (
            f"padded row {offset} carries {value}; the mask writes {PAD_SLOT_ID}"
        )
        assert not 0 <= value < addressable, (
            f"padded row {offset} carries {value}, which is a real slot of this bank"
        )
        assert value != NULL_BLOCK_ID, (
            f"padded row {offset} carries the KV machinery's own padding value, so "
            f"the mask was not applied and a padded latent would land in slot "
            f"{NULL_BLOCK_ID}"
        )
