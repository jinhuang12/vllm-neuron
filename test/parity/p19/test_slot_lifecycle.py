# SPDX-License-Identifier: Apache-2.0
"""The request slot lifecycle of GLM-5.3-Flash: arrival, completion, cancellation and slot reuse.

The runner keeps each request's KDA recurrent state and DSA indexer side caches in a
request slot, keyed by the engine's request id. Each test feeds vLLM's
``SchedulerOutput`` through the runner's state update and its slot table, and reads
the slot table, the indexer cursor and the side caches after the event.

A client disconnect is not a separate case here: the API server aborts the request,
so it reaches the runner as the same finished id as a cancellation.
"""

from __future__ import annotations

import torch

from p19_world import PROMPT, cursor, dirty, output, sparse_entries, step, synthetic_step, table, world


def test_an_arriving_request_takes_the_lowest_free_slot_emptied_and_opens_at_position_zero(monkeypatch) -> None:
    """A new request gets the lowest free slot, with its pooled store and ring emptied."""
    w = world(monkeypatch, max_num_seqs=2)
    # The side caches live for the process; a slot can hold whatever its last owner left.
    dirty(w, 0, 5.0)
    dirty(w, 1, 7.0)

    held = step(w, output(new={"a": PROMPT}))

    assert held == {"a": 0}
    assert cursor(w, 0) == PROMPT
    for entry in sparse_entries(w):
        assert not entry["pool_cache"][0].any(), "the new owner's pooled store was handed over dirty"
        assert not entry["tail"][0].any(), "the new owner's ring was handed over dirty"
        assert torch.all(entry["pool_cache"][1] == 7.0), "a slot nobody took was written"


def test_a_running_request_keeps_its_slot_and_continues_at_its_cursor(monkeypatch) -> None:
    """Each decode step of a request is served from the slot it was handed, at its own cursor."""
    w = world(monkeypatch, max_num_seqs=2)
    step(w, output(new={"a": PROMPT}))

    held = step(w, output(running={"a": PROMPT}))

    assert held == {"a": 0}
    assert cursor(w, 0) == PROMPT + 1


def test_a_completed_request_releases_its_slot_cursor_and_kda_row_at_the_next_real_step(monkeypatch) -> None:
    """A finished id frees the slot when the next real step runs; the next owner gets it emptied."""
    w = world(monkeypatch, max_num_seqs=2)
    step(w, output(new={"done": PROMPT}))
    step(w, output(new={"kept": PROMPT}, running={"done": PROMPT}))
    # The verify step's state hook records the accepted draft count per slot.
    w.runner._glm5next_checkpoint_rows[0] = 2
    w.runner._glm5next_checkpoint_rows[1] = 1
    dirty(w, 0, 5.0)
    dirty(w, 1, 7.0)

    # vLLM reports the finished id once, in the next output; this one schedules nothing
    # else, so the runner returns before building a step and the table still names it.
    assert step(w, output(finished={"done"})) == {}
    assert table(w) == {"done": 0, "kept": 1}

    held = step(w, output(new={"next": PROMPT}, running={"kept": PROMPT}))

    assert held == {"kept": 1, "next": 0}
    assert table(w) == {"kept": 1, "next": 0}
    assert 0 not in w.runner._glm5next_checkpoint_rows
    assert cursor(w, 0) == PROMPT
    for entry in sparse_entries(w):
        assert not entry["pool_cache"][0].any()
        # The ring is emptied at hand-out and the opening prefill keeps it empty here.
        assert not entry["tail"][0].any()
        assert torch.all(entry["pool_cache"][1] == 7.0), "the live neighbour's pooled store changed"
        assert torch.all(entry["tail"][1] == 7.0), "the live neighbour's ring changed"
    assert cursor(w, 1) == PROMPT + 1
    assert w.runner._glm5next_checkpoint_rows[1] == 1


def test_a_request_cancelled_between_prefill_segments_frees_its_slot_for_the_same_step(monkeypatch) -> None:
    """An abort is a finished id; a request scheduled in the same output takes the freed slot."""
    w = world(monkeypatch, max_num_seqs=1)
    # The first 1024 tokens of a 2000-token prompt.
    step(w, output(new={"cancelled": (1024, 2000)}))
    assert cursor(w, 0) == 1024

    held = step(w, output(new={"next": PROMPT}, finished={"cancelled"}))

    assert held == {"next": 0}
    assert table(w) == {"next": 0}
    assert cursor(w, 0) == PROMPT


def test_a_request_resubmitted_under_a_finished_id_is_handed_a_fresh_slot(monkeypatch) -> None:
    """An id that is finished and new in one output is two requests: the second never continues the first."""
    w = world(monkeypatch, max_num_seqs=2)
    step(w, output(new={"same": PROMPT}))
    dirty(w, 0, 5.0)

    held = step(w, output(new={"same": 200}, finished={"same"}))

    assert held == {"same": 0}
    assert cursor(w, 0) == 200
    for entry in sparse_entries(w):
        # Emptied at hand-out: a continuation would have kept the pooled store.
        assert not entry["pool_cache"][0].any()


def test_a_live_request_left_out_of_a_step_keeps_its_slot_and_its_state(monkeypatch) -> None:
    """Absence from a step is not a release: the request leaves the persistent batch and keeps its slot."""
    w = world(monkeypatch, max_num_seqs=2)
    step(w, output(new={"decoding": PROMPT}))
    step(w, output(running={"decoding": PROMPT}))
    dirty(w, 0, 3.0)

    # A prefill step that holds the decode back (the Neuron scheduler's prefill/decode
    # separation) names only the new request.
    held = step(w, output(new={"prefill": PROMPT}))

    assert held == {"prefill": 1}
    assert table(w) == {"decoding": 0, "prefill": 1}
    assert cursor(w, 0) == PROMPT + 1
    for entry in sparse_entries(w):
        assert torch.all(entry["pool_cache"][0] == 3.0)
        assert torch.all(entry["tail"][0] == 3.0)

    # The next decode step continues it at its cursor.
    assert step(w, output(running={"decoding": PROMPT + 1, "prefill": PROMPT})) == {
        "decoding": 0,
        "prefill": 1,
    }


def test_a_synthetic_step_neither_reads_nor_releases_the_slot_table(monkeypatch) -> None:
    """A warmup or idle dummy step between two real steps leaves a pending release for the real one."""
    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"done": PROMPT}))
    step(w, output(finished={"done"}))

    assert synthetic_step(w) == 0
    assert table(w) == {"done": 0}

    assert step(w, output(new={"next": PROMPT})) == {"next": 0}
    assert table(w) == {"next": 0}


def test_the_hand_out_drains_the_pending_step_before_it_empties_a_freed_slot(monkeypatch) -> None:
    """Under async scheduling the last owner's step can still be in flight; it is read back first."""
    from vllm_neuron.vllm.worker import glm5next_state_banks

    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"done": PROMPT}))
    order: list[str] = []
    real_empty = glm5next_state_banks.empty_slot

    def recorded_empty(cache, slot):
        order.append(f"empty slot {slot}")
        return real_empty(cache, slot)

    monkeypatch.setattr(glm5next_state_banks, "empty_slot", recorded_empty)
    w.runner.async_execution_buffer = {
        "async_output": type("Pending", (), {"get_output": lambda self: order.append("drain")})()
    }

    step(w, output(new={"next": PROMPT}, finished={"done"}))

    # Hand-out: drain, then the pooled store and the ring; then the opening prefill's
    # ring empty at position 0.
    assert order[:3] == ["drain", "empty slot 0", "empty slot 0"]
