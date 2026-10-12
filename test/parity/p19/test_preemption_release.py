# SPDX-License-Identifier: Apache-2.0
"""Preemption and the request slot: a preempted request gives its slot up.

vLLM bounds only its running requests by ``max_num_seqs``. A preempted request goes
back to the waiting queue with its computed length reset to 0, and its slot must not
stay held while vLLM admits other requests in its place. The scheduler reports it in
``SchedulerOutput.preempted_req_ids`` of the step that preempted it; the runner frees
the slot from that report, as it does for a finished id. A resumed request is handed
a slot again and opens its sequence at position 0.
"""

from __future__ import annotations

import pytest
import torch

from p19_world import PROMPT, cursor, dirty, output, sparse_entries, step, table, world


def test_a_new_request_takes_the_slot_a_preempted_request_gave_up(monkeypatch) -> None:
    """At the slot bound, a request admitted while another is preempted is served, not refused."""
    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"preempted": PROMPT}))
    # vLLM preempts the only running request when its next block cannot be
    # allocated; that step schedules no token.
    assert step(w, output(preempted={"preempted"})) == {}

    # FCFS schedules a skipped-waiting request (for example one whose grammar just
    # compiled) before the preempted request at the head of the waiting queue.
    held = step(w, output(new={"arrived": PROMPT}))

    assert held == {"arrived": 0}
    assert table(w) == {"arrived": 0}
    assert cursor(w, 0) == PROMPT


def test_a_preemption_frees_one_slot_and_leaves_the_running_neighbour_alone(monkeypatch) -> None:
    """The release frees only the preempted request's slot; the request that kept running keeps its state."""
    w = world(monkeypatch, max_num_seqs=2)
    step(w, output(new={"preempted": PROMPT}))
    step(w, output(new={"running": PROMPT}, running={"preempted": PROMPT}))
    dirty(w, 0, 5.0)
    dirty(w, 1, 7.0)

    # The running request takes the preempted one's blocks for its next token.
    step(w, output(running={"running": PROMPT}, preempted={"preempted"}))
    held = step(w, output(new={"arrived": PROMPT}, running={"running": PROMPT + 1}))

    assert held == {"running": 1, "arrived": 0}
    assert cursor(w, 1) == PROMPT + 2
    for entry in sparse_entries(w):
        assert not entry["pool_cache"][0].any(), "the freed slot was handed over dirty"
        assert torch.all(entry["pool_cache"][1] == 7.0), "the running request's pooled store changed"
        assert torch.all(entry["tail"][1] == 7.0), "the running request's ring changed"


def test_a_preempted_request_resumes_on_a_slot_handed_out_afresh(monkeypatch) -> None:
    """A resumed request is handed a slot like a new one and opens at position 0.

    The release runs at the next real step, as for a finished id; here that is the
    step that resumes the request, so the request is released and handed a slot in
    the same step.
    """
    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"preempted": PROMPT}))
    w.runner._glm5next_checkpoint_rows[0] = 2
    dirty(w, 0, 5.0)
    assert step(w, output(preempted={"preempted"})) == {}

    held = step(w, output(resumed={"preempted": PROMPT}))

    assert held == {"preempted": 0}
    assert cursor(w, 0) == PROMPT
    assert 0 not in w.runner._glm5next_checkpoint_rows
    for entry in sparse_entries(w):
        assert not entry["pool_cache"][0].any(), "the resumed request read its last pass's pooled store"
        assert not entry["tail"][0].any(), "the resumed request read its last pass's ring"


def test_a_preempted_request_aborted_before_it_resumes_is_released_once(monkeypatch) -> None:
    """A preempted id later reported finished has nothing left to release, and no step fails on it."""
    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"preempted": PROMPT}))
    step(w, output(preempted={"preempted"}))

    held = step(w, output(new={"arrived": PROMPT}, finished={"preempted"}))

    assert held == {"arrived": 0}
    assert table(w) == {"arrived": 0}


def test_a_scheduler_output_without_preempted_ids_releases_nothing(monkeypatch) -> None:
    """``preempted_req_ids`` defaults to ``None`` in ``SchedulerOutput``; that is read as no preemption."""
    w = world(monkeypatch, max_num_seqs=2)
    step(w, output(new={"kept": PROMPT}))

    held = step(w, output(new={"arrived": PROMPT}, preempted=None))

    assert held == {"arrived": 1}
    assert table(w) == {"kept": 0, "arrived": 1}


def test_a_preemption_the_scheduler_does_not_report_keeps_its_slot(monkeypatch) -> None:
    """``reset_prefix_cache(reset_running_requests=True)`` preempts outside ``schedule()``; no id reaches the runner.

    The request only leaves the persistent batch, which is not a release, and resumes
    on the slot it holds. At the slot bound, a request admitted ahead of it is still
    refused by name: the release covers preemptions the scheduler reports.
    """
    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"forced": PROMPT}))

    assert step(w, output()) == {}
    assert table(w) == {"forced": 0}
    with pytest.raises(ValueError, match="has no free slot"):
        step(w, output(new={"arrived": PROMPT}))


def test_a_request_resumed_after_an_unreported_preemption_reopens_its_own_slot(monkeypatch) -> None:
    """Without a report, the resumed request keeps its slot and its prefill at position 0 empties the ring."""
    w = world(monkeypatch, max_num_seqs=1)
    step(w, output(new={"forced": PROMPT}))
    dirty(w, 0, 3.0)
    step(w, output())

    held = step(w, output(resumed={"forced": PROMPT}))

    assert held == {"forced": 0}
    assert cursor(w, 0) == PROMPT
    for entry in sparse_entries(w):
        assert not entry["tail"][0].any(), "the reopened ring still holds the last pass's rows"
        # The pooled store is left for the new prefill to overwrite pool by pool.
        assert torch.all(entry["pool_cache"][0] == 3.0)
