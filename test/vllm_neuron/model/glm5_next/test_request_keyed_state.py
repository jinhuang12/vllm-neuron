# SPDX-License-Identifier: Apache-2.0
"""The request-keyed cache state: the runner half.

A request's conv and recurrent state live at one slot number, and a step reads and
writes the slots its own requests hold.
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
# nothing in the tree binds it; the difference is stated per value.
# ---------------------------------------------------------------------------

#: Two requests is the smallest batch that can show request keying at all: one
#: request cannot collide with itself.
DECLARED_REQUESTS = 2

#: The recurrent-state bank's slot count. Four rather than two so that a slot
#: number can never be confused with a block id or a request index by accident.
#: this is a paging geometry, not a request axis: on the serving stack it is the
#: group's KV block count, which is why it is deliberately larger than the bound
#: below and why no per-sequence cache may be sized by it.
DECLARED_STATE_SLOTS = 4

#: How many sequences the modelled engine admits at once -- the scheduler's
#: ``max_num_seqs``, which the runner reads once at construction. Two, so that the
#: number differs from the bank's slot count above: an allocation sized by the wrong
#: one of the two is then visible as a shape rather than passing by coincidence.
DECLARED_MAX_NUM_SEQS = 2

#: The page. Nothing binds a page for a synthetic bank, so this file
#: declares one and the bank below is built to it; the converter cross-checks the
#: two against each other, which is the property the per-token arm relies on.
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

#: The two requests' cached lengths. They differ on purpose: equal lengths would
#: let a single-start-position derivation pass by coincidence.
DECLARED_CACHED_LENGTHS = (7, 19)

#: The two requests' block rows. Neither is a prefix of the other and the second
#: is deliberately not one ascending run continuing the first, so no single slice of
#: the bank can serve both; each sparse carrier must keep its own request's pages.
DECLARED_SPARSE_ROWS = ((0, 1), (4, 5))

#: A row whose pages are neither adjacent nor ascending, which is the shape the
#: contiguity refusal used to turn away and the carrier now serves. Both entries are
#: pages this file's bank holds, so the rows they address are real.
DECLARED_SCATTERED_ROW = (4, 1)

#: A step that straddles the two entries of that row, so the order the carrier names
#: them in is readable off the rows it hands the layer: it opens two tokens before the
#: page boundary and ends two past it.
DECLARED_SCATTERED_START = DECLARED_PAGE_SIZE - 2
DECLARED_SCATTERED_TOKENS = 4

#: The decode bucket the padded-row test pads up to. Two real requests in a bucket of four leaves
#: two padded rows, which is the case the padding writer produces.
DECLARED_DECODE_BUCKET = 4

#: The prefill warmup's token count. The real value is a launch-shape bucket; the
#: only property the converter reads is that it exceeds the decode threshold, which
#: is what makes the warmup step read as a prefill (``neuron_model_runner.py``).
DECLARED_WARMUP_BUCKET = 4

#: The state-carrier keys, in the model's own declared spelling
DECLARED_STATE_CARRIER_KEYS = {
    "conv_state",
    "recurrent_state",
    "is_prefill",
    "start_position",
    "real_tokens",
    "row_mask",
}

#: The conv and recurrent state shapes per slot. Small, and their only
#: requirement is that the two differ so a mixed-up carrier is visible.
DECLARED_CONV_SHAPE = (4, 16)
DECLARED_RECURRENT_SHAPE = (2, 8, 8)


class VacuousControlError(AssertionError):
    """A control that cannot discriminate is a failure, never a pass."""


def _require_cpu_mode() -> None:
    """The environment is part of the acceptance command, never a fixture. """
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.fail(
            "this file's declared acceptance runs under VLLM_NEURON_CPU_MODE=1; "
            f"the environment carries {os.environ.get('VLLM_NEURON_CPU_MODE')!r}"
        )


# ---------------------------------------------------------------------------
# The world: a two-layer hybrid stack, one sparse layer and one linear layer.
# ---------------------------------------------------------------------------
def _text_config() -> SimpleNamespace:
    """Only the fields the converter reads off ``model.text_config``. """
    return SimpleNamespace(
        index_kpool=DECLARED_INDEX_KPOOL,
        index_head_dim=DECLARED_INDEX_HEAD_DIM,
        qk_nope_head_dim=DECLARED_HEAD_SIZE,
        qk_rope_head_dim=0,
    )


def _banks() -> list[dict]:
    """The bank mapping ``bind_kv_cache`` leaves on the model, built its way. """
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
    """A runner carrying only what the converter reads. """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=_text_config(), glm5next_layer_banks=tuple(banks)
    )
    runner.max_model_len = DECLARED_BLOCKS * DECLARED_PAGE_SIZE
    # The concurrency bound is the slot axis, so the harness carries it: a runner
    # without it raises here rather than letting the code fall back to a geometry.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    if request_ids is None:
        request_ids = [f"req-{index}" for index in range(DECLARED_REQUESTS)]
    runner.input_batch = SimpleNamespace(req_ids=list(request_ids))
    return runner


def _entry(*, rows, tokens: int, cached, block_size: int) -> dict:
    """one KV-cache group's attention-metadata entry, at this step's geometry. """
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
    """The physical slot per token, derived the way the runner derives it. """
    slots: list[int] = []
    for row, start in zip(rows, cached):
        for offset in range(int(tokens)):
            position = int(start) + offset
            block = int(row[position // int(block_size)])
            slots.append(block * int(block_size) + position % int(block_size))
    slots.extend([NULL_BLOCK_ID] * int(padded_rows))
    return torch.tensor(slots, dtype=torch.int32)


def _metadata(banks, *, tokens: int, sparse_rows, state_rows, cached) -> dict:
    """two KV-cache groups, each with its own table, keyed by every layer name. """
    sparse = _entry(rows=sparse_rows, tokens=tokens, cached=cached,
                    block_size=DECLARED_PAGE_SIZE)
    state = _entry(rows=state_rows, tokens=tokens, cached=cached,
                   block_size=DECLARED_PAGE_SIZE)
    return {
        str(bank["name"]): (sparse if bank["family"] == "self_attn" else state)
        for bank in banks
    }


def _generic(*, tokens: int, metadata: dict) -> dict:
    """One step's generic runner kwargs at a given metadata mapping. """
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
    """One step carrying both requests, converted by the code under test. """
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
    assert sorted(converted) == [
        "expert_parallel_rank", "input_ids", "layer_carriers", "moe_group",
        "sampling_positions", "tp_degree",
    ], f"the converter returned {sorted(converted)}"
    return converted["layer_carriers"]


def _linear_banks(banks) -> list[dict]:
    """The recurrent banks, which are the ones a state slot indexes."""
    return [bank for bank in banks if bank["family"] != "self_attn"]


def _side_caches(banks):
    """The indexer's two caches, one set per sparse layer with a slot axis. """
    return NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=DECLARED_INDEX_KPOOL,
        index_head_dim=DECLARED_INDEX_HEAD_DIM,
        max_seq_len=DECLARED_BLOCKS * DECLARED_PAGE_SIZE,
        request_slots=DECLARED_STATE_SLOTS,
    )


def _carriers_for(banks, side, *, slot: int, rows, cached: int, is_prefill: bool):
    """One request's carriers at a given slot, built by the code under test. """
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
    """The carrier builder at a declared request count, for the refusal tests. """
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
# One request's indexer state is untouched by another request's step.
# ══════════════════════════════════════════════════════════════════════════════


def test_a3_one_requests_side_caches_are_untouched_by_the_others_step() -> None:
    """The pooled store and the tail ring belong to a request, not to the process. """
    _require_cpu_mode()
    banks = _banks()
    side = _side_caches(banks)
    sparse = [index for index, bank in enumerate(banks) if bank["family"] == "self_attn"]
    if not sparse:
        raise VacuousControlError(
            "this test needs a sparse layer, which is the only family holding side "
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
            assert a["pool_cache"].data_ptr() != b["pool_cache"].data_ptr(), (
                f"layer {index}: two requests were handed ONE pooled store, so the "
                f"second would pool into the first's rows"
            )
            assert a[ring_key].data_ptr() != b[ring_key].data_ptr(), (
                f"layer {index}: two requests were handed ONE tail ring under "
                f"{ring_key!r}"
            )

            # The write, and the read that proves it did not travel.
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
                    "the write this test relies on left request A's pooled store at "
                    "zero, so the comparison above proves nothing"
                )


# ══════════════════════════════════════════════════════════════════════════════
# The slot half. Two requests never share a state slot.
# ══════════════════════════════════════════════════════════════════════════════


def test_a1_two_requests_are_assigned_distinct_state_slots() -> None:
    """The slot is the request's identity, not the block table's first block id. """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    request_ids = list(runner.input_batch.req_ids)
    assert len(request_ids) == DECLARED_REQUESTS, (
        f"this test needs {DECLARED_REQUESTS} requests to have a collision to rule "
        f"out; the harness offered {len(request_ids)}"
    )

    slots = runner._glm5next_request_slots(banks, request_ids, synthetic=False)

    assert len(slots) == len(request_ids)
    assert len(set(slots)) == len(slots), (
        f"the two requests were handed slots {slots}; two live requests sharing a "
        f"slot means one continues the other's recurrence"
    )
    # The range a slot must lie in is the engine's concurrency bound, not the bank's slot
    # count, because the concurrency bound is the axis the per-sequence caches carry: a
    # slot at or above it would index those caches out of range.
    for slot in slots:
        assert 0 <= slot < DECLARED_MAX_NUM_SEQS, (
            f"slot {slot} is outside the {DECLARED_MAX_NUM_SEQS} keyed sequence slot(s)"
        )
    # The mapping is stable: asking again inside one request's life returns the
    # same slots, because a slot that moved would abandon the state it holds.
    again = runner._glm5next_request_slots(banks, request_ids, synthetic=False)
    assert again == slots, f"the slots moved from {slots} to {again} within one life"


def test_a1_each_requests_carrier_is_a_view_of_its_own_bank_row() -> None:
    """The carrier half: two requests, two views, and the writes land in the bank.
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
            f"this test needs {DECLARED_REQUESTS} distinct slots to tell the two "
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
    assert set(carrier) == DECLARED_STATE_CARRIER_KEYS, (
        f"the linear carrier holds {sorted(carrier)}, not "
        f"{sorted(DECLARED_STATE_CARRIER_KEYS)}; the layer takes these as keywords, "
        f"so an extra key raises and a missing one is served as a default"
    )
    assert len(carrier["conv_state"]) == DECLARED_REQUESTS, (
        f"the carrier holds {len(carrier['conv_state'])} conv entry(ies) for "
        f"{DECLARED_REQUESTS} requests"
    )
    # The positions are one int32 tensor with a row per request, compared as a tensor:
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

    # ---- the view read, which a copy cannot pass: write through the carrier.
    untouched = bank["recurrent_state"][slots[0]].clone()
    carrier["recurrent_state"][1].fill_(3.0)
    written = float(bank["recurrent_state"][slots[1]].abs().max())
    assert written == 3.0, (
        "a write through the second request's carrier did not reach its bank row, so "
        "the carrier is a copy and the layer's in-place advance would be discarded"
    )
    assert torch.equal(bank["recurrent_state"][slots[0]], untouched), (
        "the write reached the FIRST request's row as well; the two requests' states "
        "must be disjoint or one continues the other's sequence"
    )


def test_a1_a_two_request_linear_batch_is_served_through_the_converter() -> None:
    """The whole runner path, not the builder: two requests in one batch, end to end.
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


def test_a1_the_sparse_family_preserves_each_requests_pages_and_state() -> None:
    """Concurrent decode keeps each request's table, position, and state view."""
    _require_cpu_mode()
    banks = _banks()
    side = _side_caches(banks)
    rows = ((4, 1), (6, 2))
    starts = (DECLARED_PAGE_SIZE + 1, 19)
    slots = (0, 2)
    geometries = [
        {
            "block_ids": list(rows[0]),
            "block_id_rows": [list(row) for row in rows],
            "state_slot": slots[0],
            "state_slots": list(slots),
            "page_size": DECLARED_PAGE_SIZE,
            "window_blocks": 2,
        }
        for _ in banks
    ]
    carriers = NeuronModelRunner._glm5next_layer_carriers(
        banks, side, geometries=geometries, is_prefill=False,
        tokens=DECLARED_REQUESTS, start_position=starts[0],
        softmax_scale=float(DECLARED_HEAD_SIZE) ** -0.5,
        max_seq_len=max(starts) + 1, index_kpool=DECLARED_INDEX_KPOOL,
        requests=DECLARED_REQUESTS, request_starts=starts,
    )
    sparse = carriers[0]
    assert sparse["latent_cache"] is banks[0]["latent_cache"]
    for index, (slot, start, row) in enumerate(zip(slots, starts, rows)):
        assert sparse["block_table_row"][index].flatten().tolist() == list(row)
        assert sparse["latent_slots"][index].tolist() == [
            row[start // DECLARED_PAGE_SIZE] * DECLARED_PAGE_SIZE
            + start % DECLARED_PAGE_SIZE
        ]
        assert sparse["seq_lens"][index].tolist() == [start + 1]
        assert sparse["position"][index].item() == start
        assert sparse["pool_cache"][index].data_ptr() == side[0]["pool_cache"][slot].data_ptr()
        assert sparse["tail"][index].data_ptr() == side[0]["tail"][slot].data_ptr()


def test_a1_the_converter_preserves_all_sparse_decode_rows() -> None:
    """The real converter retains the second request's page and write address."""
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    metadata = _metadata(
        banks, tokens=1, sparse_rows=DECLARED_SPARSE_ROWS,
        state_rows=DECLARED_SPARSE_ROWS, cached=(0, 0),
    )
    converted = runner._glm5next_model_kwargs(_generic(tokens=2, metadata=metadata))
    sparse = converted["layer_carriers"][0]
    for index, row in enumerate(DECLARED_SPARSE_ROWS):
        assert sparse["block_table_row"][index].flatten().tolist() == [row[0], -1]
        assert sparse["latent_slots"][index].tolist() == [row[0] * DECLARED_PAGE_SIZE]
        assert sparse["seq_lens"][index].tolist() == [1]
    assert sparse["pool_cache"][0].data_ptr() != sparse["pool_cache"][1].data_ptr()
    assert sparse["tail"][0].data_ptr() != sparse["tail"][1].data_ptr()


# ══════════════════════════════════════════════════════════════════════════════
# A finished request's slot is freed, and its next owner gets it zeroed.
# ══════════════════════════════════════════════════════════════════════════════


def test_a2_a_finished_requests_slot_is_reused_and_zeroed_on_hand_out() -> None:
    """Freed, reused, and zeroed at the moment ownership changes. """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    side = _side_caches(banks)
    rings = [entry for entry in side if entry]
    if not rings:
        raise VacuousControlError(
            "no bank in this stack carries indexer side caches, so the second half of "
            "this test -- that they are zeroed at the same moment -- measures nothing"
        )
    first = ["req-0"]

    seated = runner._glm5next_request_slots(
        banks, first, synthetic=False, side_caches=side
    )
    slot = seated[0]

    # Dirty the slot, and prove it is dirty. Without this the zero read below
    # would pass against a slot that was never written. Both the recurrent state and
    # the indexer's two caches are dirtied, because a half-fresh slot -- fresh
    # recurrence beside a stale pool -- is the defect the joint zeroing closes.
    for bank in _linear_banks(banks):
        bank["conv_state"][slot].fill_(3.0)
        bank["recurrent_state"][slot].fill_(-2.0)
    # What the previous owner left, kept to compare against rather than re-typed as a
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
    if not all(dirtied):
        raise VacuousControlError(
            "this test needs the freed slot to hold non-zero state before hand-out, "
            f"and the banks read {dirtied}"
        )

    # Req-0 finishes, and the engine's finished set is what says so.
    runner._glm5next_note_finished_requests(["req-0"])
    later = runner._glm5next_request_slots(
        banks, ["req-2"], synthetic=False, side_caches=side
    )

    assert later == [slot], (
        f"the finished request's slot {slot} was not handed to the next request, "
        f"which got {later}; a table that never frees leaks its slots"
    )
    # The recurrent banks are not written at hand-out, and this reads that they are not.
    # They are the engine's own cache tensors, every one a view of a single allocation, so
    # an eager write on them is refused by the runtime; the freshness is served where the
    # state is read instead, and the layer test that measures it is the fresh-leg pair in
    # `test_kda_prefill_segments.py`. What this test owns is that the slot changes hands
    # without a write, so a bank still holding the previous request's values is expected.
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
    # The indexer's two caches are zeroed at the same moment. A slot handed over with
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
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    too_many = [f"req-{index}" for index in range(DECLARED_MAX_NUM_SEQS + 1)]

    with pytest.raises(ValueError, match="has no free slot"):
        runner._glm5next_request_slots(banks, too_many, synthetic=False)


def test_a2_banks_holding_fewer_slots_than_the_bound_refuse_by_name() -> None:
    """Banks holding fewer slots than the engine admits sequences refuse by name. """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    banked = runner._glm5next_state_slot_count(banks)
    if banked <= 0:
        raise VacuousControlError(
            "this test raises the bound past the banks' own slot count, and the "
            "harness's banks report no recurrent state slot at all"
        )
    runner.max_num_reqs = banked + 1

    with pytest.raises(ValueError, match="no slot to hand the last sequence"):
        runner._glm5next_request_slot_capacity(banks)


def test_a2_a_stack_with_no_recurrent_bank_is_admitted_at_the_bound() -> None:
    """A stack holding no recurrent bank has nothing to provision, so it must be served.
    """
    _require_cpu_mode()
    sparse_only = [bank for bank in _banks() if bank["family"] == "self_attn"]
    runner = _runner(sparse_only)
    banked = runner._glm5next_state_slot_count(sparse_only)
    if banked:
        raise VacuousControlError(
            f"this test drives a stack with no recurrent bank, and the harness's "
            f"sparse-only stack reports {banked} recurrent state slot(s)"
        )

    assert (
        runner._glm5next_request_slot_capacity(sparse_only) == DECLARED_MAX_NUM_SEQS
    ), "a stack with no recurrent bank was not admitted at the engine's own bound"


def test_a2_the_side_cache_slot_axis_is_the_engines_concurrency_bound() -> None:
    """The per-sequence caches are sized by ``max_num_seqs``, never by the block space.
    """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    if DECLARED_MAX_NUM_SEQS == DECLARED_STATE_SLOTS:
        raise VacuousControlError(
            "this test tells the concurrency bound from the bank's slot count by their "
            f"shapes, and the harness declares both as {DECLARED_MAX_NUM_SEQS}"
        )

    live = runner._glm5next_live_side_caches(banks)

    rings = [entry for entry in live if entry]
    if not rings:
        raise VacuousControlError(
            "no bank in this stack carries indexer side caches, so this test has no "
            "slot axis to read"
        )
    axes = [
        (int(entry["pool_cache"].shape[0]), int(entry["tail"].shape[0]))
        for entry in rings
    ]
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
    """Absence from a step's batch is not finishing, and the state must survive it. """
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

    # The step that skips it. Only the other request is scheduled, and the engine says
    # nobody finished.
    runner._glm5next_request_slots(banks, [kept], synthetic=False)
    returned = runner._glm5next_request_slots(banks, both, synthetic=False)

    survived = [
        bool(bank["recurrent_state"][skipped_slot].any()) for bank in _linear_banks(banks)
    ]
    assert returned[both.index(skipped)] == skipped_slot, (
        f"request {skipped!r} was skipped for one step and came back to slot "
        f"{returned[both.index(skipped)]} instead of its own {skipped_slot}"
    )
    assert all(survived), (
        f"request {skipped!r}'s recurrent state was cleared by a step that merely did "
        f"not schedule it; the banks read {survived}"
    )

    # And the slot does free when the engine says so, which is what keeps the half
    # above from passing on a table that never frees anything.
    runner._glm5next_note_finished_requests([skipped])
    runner._glm5next_request_slots(banks, [kept], synthetic=False)
    table = dict(runner._glm5next_request_slot_table)
    assert skipped not in table, (
        f"request {skipped!r} was finished by the engine and still owns slot "
        f"{table.get(skipped)}"
    )


def test_a2_a_synthetic_step_refuses_live_owners_without_changing_claims() -> None:
    """Startup warmup cannot borrow the state of a live request."""
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks, request_ids=[])
    runner._glm5next_request_slots(banks, ["req-0"], synthetic=False)
    before = dict(runner._glm5next_request_slot_table)
    with pytest.raises(ValueError, match="startup without live requests"):
        runner._glm5next_request_slots(banks, [None], synthetic=True)
    assert runner._glm5next_request_slot_table == before


def test_a2_the_prefill_warmups_own_shape_is_served_and_takes_no_claim() -> None:
    """The prefill warmup has no request, and it must still be served. """
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
    """One request's whole one-token prompt, converted by the code under test. """
    metadata = _metadata(
        banks,
        tokens=1,
        sparse_rows=(DECLARED_SPARSE_ROWS[0],),
        state_rows=(DECLARED_SPARSE_ROWS[0],),
        cached=(0,),
    )
    return runner._glm5next_model_kwargs(_generic(tokens=1, metadata=metadata))


def _held_and_opening(banks):
    """A runner whose batch names one opening request, with another already seated. """
    opening, held = "req-opening", "req-held"
    runner = _runner(banks, request_ids=[opening])
    live = runner._glm5next_live_side_caches(banks)
    held_slot = runner._glm5next_request_slots(banks, [held], synthetic=False)[0]
    if DECLARED_MAX_NUM_SEQS < 2:
        raise VacuousControlError(
            f"these tests tell a request's own slot from slot 0 by their numbers, and "
            f"the harness admits {DECLARED_MAX_NUM_SEQS} concurrent sequence(s)"
        )
    if held_slot != 0:
        raise VacuousControlError(
            f"the already-seated request holds slot {held_slot}; these tests read "
            f"whether the opening request was served from SLOT 0 instead of its own, "
            f"so slot 0 has to be the one that is already taken"
        )
    return runner, live, opening, held, held_slot


def test_a2_a_one_token_prompt_is_keyed_by_its_own_request_not_slot_zero() -> None:
    """A prompt of one token is an opening sequence, not a step without a request. """
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
    """The opening arm is the position's, so this step opens a sequence here. """
    _require_cpu_mode()
    banks = _banks()
    runner, live, opening, held, held_slot = _held_and_opening(banks)
    #: Any non-zero value stands for what the previous owner stashed; this one is
    #: exact in every dtype a ring can carry.
    planted_ring = 3.0
    rings = [entry for entry in live if entry and "tail" in entry]
    if not rings:
        raise VacuousControlError(
            "no bank in this stack carries an indexer ring, so this test has no "
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
    """The classification that has no legal caller, refused by name. """
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    scheduled = list(runner.input_batch.req_ids)
    if not scheduled:
        raise VacuousControlError(
            "this test needs a batch that names a request, and the harness built one "
            "that names none"
        )

    with pytest.raises(ValueError) as caught:
        runner._glm5next_request_identities(synthetic=True)

    message = str(caught.value)
    served = _runner(banks, request_ids=[])._glm5next_request_identities(
        synthetic=True
    )
    assert "classified as having no request" in message, (
        f"the call raised, but not the refusal this test names: {message}"
    )
    assert scheduled[0] in message, (
        f"the refusal names no request of the batch it refused: {message}"
    )
    assert served == [None], (
        f"a step whose batch names no request was refused too, and that is the step "
        f"warmup builds; the call returned {served}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# The batch's per-token operands are the per-request derivations, in order.
# ══════════════════════════════════════════════════════════════════════════════


def test_a4_the_batch_operands_are_the_per_request_derivations_concatenated() -> None:
    """Two requests at different cached lengths, and each token gets its own. """
    _require_cpu_mode()
    first, second = DECLARED_CACHED_LENGTHS
    if first == second:
        raise VacuousControlError(
            f"this test needs two different cached lengths to separate a per-request "
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
    assert torch.equal(seq_lens, want_seq_lens), (
        f"the batch seq_lens {seq_lens.tolist()} are not the per-request "
        f"derivations concatenated {want_seq_lens.tolist()}"
    )
    assert torch.equal(slots, want_slots), (
        f"the batch pool slots {slots.tolist()} are not the per-request "
        f"derivations concatenated {want_slots.tolist()}"
    )

    # ---- the old derivation, run, so the test is proved to separate the two.
    total = sum(tokens for tokens, _ in requests)
    stale_seq_lens = NeuronModelRunner._glm5next_row_seq_lens(
        tokens=total, start_position=first, device=device
    )
    assert not torch.equal(seq_lens, stale_seq_lens), (
        f"a single-start-position derivation produced the same seq_lens "
        f"{stale_seq_lens.tolist()} as the per-request one, so this test is not "
        f"measuring the per-request derivation at all"
    )


# ══════════════════════════════════════════════════════════════════════════════
# The mapping half. The runner's slot number and the bank's view are one address.
# ══════════════════════════════════════════════════════════════════════════════


def test_a5_the_runners_slot_mapping_addresses_the_banks_own_view() -> None:
    """The physical slot the KV machinery computes is the bank view's row index. """
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
        # The control: the neighbouring slot must not have taken the write.
        neighbour = (slot + 1) % int(sparse["latent_cache"].shape[0])
        assert not bool((sparse["latent_cache"][neighbour, 0, :] == sentinel).all()), (
            f"request {index}: slot {neighbour} also holds the sentinel, so the "
            f"write landed wider than one slot and the read proves nothing"
        )


# ══════════════════════════════════════════════════════════════════════════════
# A decode carries one token per request, and more than that still refuses.
# ══════════════════════════════════════════════════════════════════════════════


def test_a6_a_decode_carrying_more_tokens_than_requests_refuses_by_name() -> None:
    """Speculative decoding's verify step is still out of scope, now stated exactly. """
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
            "this test needs more tokens than requests to have anything to refuse"
        )
    with pytest.raises(ValueError, match="one token per request"):
        _carriers_for_requests(
            banks, side, tokens=over_by_one, requests=DECLARED_REQUESTS,
            is_prefill=False,
        )

    base = mla_sparse_dispatch_counters()
    row_tiled = mla_sparse_row_tiled_dispatch_counters()
    assert base == (0, 0), f"a refused step reached the seam: {base}"
    assert row_tiled == (0, 0), f"a refused step reached the row-tiled seam: {row_tiled}"
    assert dict(runner._glm5next_request_slot_table) == table_before, (
        "the refused step changed the slot table; a refusal must leave no trace"
    )
    for entry, before in zip([e for e in side if e], pools_before):
        assert torch.equal(entry["pool_cache"], before), (
            "the refused step wrote into a pooled store"
        )

    # ---- the admitted case, so the refusal is not simply refusing everything.
    admitted = _carriers_for_requests(
        banks, side, tokens=DECLARED_REQUESTS, requests=DECLARED_REQUESTS,
        is_prefill=False,
    )
    assert len(admitted) == len(banks)


# ══════════════════════════════════════════════════════════════════════════════
# A padded decode row writes nowhere, because zero is a real slot.
# ══════════════════════════════════════════════════════════════════════════════


def test_a7_padded_decode_rows_carry_a_sentinel_and_never_slot_zero() -> None:
    """The KV machinery's padding value is an address, so it is masked here. """
    _require_cpu_mode()
    banks = _banks()
    sparse = next(bank for bank in banks if bank["family"] == "self_attn")
    addressable = int(sparse["latent_cache"].shape[0])
    padded_rows = DECLARED_DECODE_BUCKET - DECLARED_REQUESTS
    if padded_rows <= 0:
        raise VacuousControlError(
            f"this test needs a decode bucket wider than the request count to have a "
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

    # ---- why the padding value is dangerous, asserted rather than asserted about.
    assert 0 <= NULL_BLOCK_ID < addressable, (
        f"this test's whole premise is that the padding value {NULL_BLOCK_ID} is a "
        f"real slot; the bank holds {addressable} slot(s), so it is not, and the "
        f"mask this test measures would be unnecessary"
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


# ══════════════════════════════════════════════════════════════════════════════
# A scattered block table is served, and its pages are named in order.
# ══════════════════════════════════════════════════════════════════════════════


def test_a8_a_scattered_block_table_is_served_and_named_in_order() -> None:
    """The refusal this replaces is stated as the fact that replaced it. """
    _require_cpu_mode()
    banks = _banks()
    side = _side_caches(banks)
    ids = [int(value) for value in DECLARED_SCATTERED_ROW]
    if sorted(ids) == ids:
        raise VacuousControlError(
            f"this test needs a row that is not one ascending run, which is what the "
            f"old refusal turned away; it declares {ids}"
        )
    # One padded entry, so the bucket's width is wider than this request's pages and the
    # padding value is read rather than assumed absent.
    padded = len(ids) + 1
    geometries = [
        {
            "block_ids": ids,
            "state_slot": 0,
            "page_size": DECLARED_PAGE_SIZE,
            "window_blocks": padded,
        }
        for _ in banks
    ]

    carriers = NeuronModelRunner._glm5next_layer_carriers(
        banks,
        side,
        geometries=geometries,
        is_prefill=True,
        tokens=DECLARED_SCATTERED_TOKENS,
        start_position=DECLARED_SCATTERED_START,
        softmax_scale=float(DECLARED_HEAD_SIZE) ** -0.5,
        max_seq_len=DECLARED_SCATTERED_START + DECLARED_SCATTERED_TOKENS,
        index_kpool=DECLARED_INDEX_KPOOL,
    )

    index = next(
        position for position, bank in enumerate(banks) if bank["family"] == "self_attn"
    )
    carrier = carriers[index]
    bank = banks[index]["latent_cache"]
    want = _physical_slots(
        rows=[ids],
        cached=[DECLARED_SCATTERED_START],
        tokens=DECLARED_SCATTERED_TOKENS,
        block_size=DECLARED_PAGE_SIZE,
    ).tolist()
    if_sorted = _physical_slots(
        rows=[sorted(ids)],
        cached=[DECLARED_SCATTERED_START],
        tokens=DECLARED_SCATTERED_TOKENS,
        block_size=DECLARED_PAGE_SIZE,
    ).tolist()

    # ---- the bank, whole: the pages are named beside it, so nothing is cut out of it.
    assert carrier["latent_cache"].data_ptr() == bank.data_ptr(), (
        "the carrier's latent cache does not start at the bank's first slot, so it is a "
        "slice of the bank and the table beside it addresses the wrong rows"
    )
    assert int(carrier["latent_cache"].shape[0]) == DECLARED_BLOCKS * DECLARED_PAGE_SIZE, (
        f"the carrier holds {int(carrier['latent_cache'].shape[0])} slot(s) where the "
        f"bank holds {DECLARED_BLOCKS * DECLARED_PAGE_SIZE}"
    )

    # ---- the table: this request's pages in order, padded to the bucket's width.
    assert carrier["block_table_row"].dtype == torch.int32
    assert tuple(carrier["block_table_row"].shape) == (padded, 1), (
        f"the table is {tuple(carrier['block_table_row'].shape)} where the bucket's "
        f"width is {padded} page(s) of one column"
    )
    assert carrier["block_table_row"].flatten().tolist() == ids + [-1], (
        f"the table names {carrier['block_table_row'].flatten().tolist()} for a request "
        f"holding {ids}; the pages come in the order given and the bucket's remaining "
        f"entry is -1, which is the value the layer's own refusal reads"
    )

    # ---- the rows: each token's physical bank row, through the page it falls in.
    assert carrier["latent_slots"].dtype == torch.int64
    assert carrier["latent_slots"].tolist() == want, (
        f"the carrier hands the layer rows {carrier['latent_slots'].tolist()} where this "
        f"step's tokens live at {want}; a write at another row lands in a page this "
        f"request was never given"
    )
    if want == if_sorted:
        raise VacuousControlError(
            f"a sorted row addresses the same rows {if_sorted}, so this test does not "
            f"measure that the table's order was kept"
        )


def test_concurrent_sparse_prefill_refuses_before_state_changes() -> None:
    """The decode bridge does not admit prefill or reset an existing ring."""
    _require_cpu_mode()
    banks = _banks()
    runner = _runner(banks)
    side = runner._glm5next_live_side_caches(banks)
    side[0]["tail"].fill_(7)
    before = side[0]["tail"].clone()
    with pytest.raises(ValueError, match="concurrent sparse prefill"):
        _two_request_step(runner, banks, tokens=14, cached=(0, 0))
    assert torch.equal(side[0]["tail"], before)
