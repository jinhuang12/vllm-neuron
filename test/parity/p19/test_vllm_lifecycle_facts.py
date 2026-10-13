# SPDX-License-Identifier: Apache-2.0
"""What vLLM's scheduler does on preemption, read off its own code.

The slot release on preemption is safe only because vLLM restarts a preempted request
from position 0: it never resumes from the request's old per-slot state. And the
release is needed because FCFS can schedule another request before the preempted one.
These tests call vLLM's scheduler methods on a scheduler that holds only the members
those methods read.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus


def _scheduler() -> Scheduler:
    """An FCFS scheduler with empty queues, and no KV or encoder cache to free."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.finished_req_ids = set()
    scheduler.defer_block_free = False
    scheduler.kv_cache_manager = SimpleNamespace(free=lambda request: None)
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda request: None)
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    return scheduler


def _running_request(request_id: str, computed: int) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=[1] * computed,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = computed
    return request


def test_vllm_restarts_a_preempted_request_from_position_zero_at_the_head_of_the_queue() -> None:
    """Preemption resets the computed length to 0 and does not finish the request."""
    scheduler = _scheduler()
    waiting = _running_request("waiting", 0)
    waiting.status = RequestStatus.WAITING
    scheduler.waiting.add_request(waiting)
    preempted = _running_request("preempted", 300)

    scheduler._preempt_request(preempted, timestamp=0.0)

    assert preempted.status == RequestStatus.PREEMPTED
    assert preempted.num_computed_tokens == 0
    assert scheduler.waiting.peek_request() is preempted
    assert "preempted" not in scheduler.finished_req_ids


def test_fcfs_schedules_a_skipped_request_before_a_preempted_one() -> None:
    """A skipped-waiting request goes first, ahead of the preempted request at the waiting queue's head."""
    scheduler = _scheduler()
    preempted = _running_request("preempted", 300)
    scheduler._preempt_request(preempted, timestamp=0.0)
    skipped = _running_request("skipped", 0)
    skipped.status = RequestStatus.WAITING
    scheduler.skipped_waiting.add_request(skipped)

    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.skipped_waiting

    # Control: without a skipped request, the preempted one is next.
    scheduler.skipped_waiting.pop_request()
    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.waiting
    assert scheduler.waiting.peek_request() is preempted
