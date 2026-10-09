# SPDX-License-Identifier: Apache-2.0
"""A side-cache slot hand-out drains the pending async step before its first device read.

The gate run dsa8k-pc (2026-10-08 10:41, ``reports/prefill-cores-hang.md`` §worker-59) hung
because two host threads waited on one in-flight execution: the async-output thread draining
an aborted request's intermediate prefill chunk (``AsyncNeuronModelRunnerOutput.get_output``,
``neuron_model_runner.py`` line 206) and the submit thread's one-element ordering read in
``glm5next_state_banks.empty_slot`` when that request's slot was handed to the next request.
The runtime keeps one completion handle per execution (NRT ``NMGR:tpb_xu_base_set_comp_efd
... Completion handle already set for sequence ..., overwriting previous FD 1501 with 1504``),
so the first waiter never woke and the engine's ``sample_tokens`` RPC timed out. The
composition-change materialize skips an all-partial pending output on purpose, so the
hand-out is the place that must synchronise: through ``get_output``'s lock, never with a
second device wait.

Read here on the runner shell of ``test_glm5next_slot_reset.py``, with a stand-in runtime
that records the waiters of every in-flight execution and a stand-in pending step whose
``get_output`` behaves as the real one's lock does (the caller returns when the output
thread's wait has returned, that is when the execution is complete):

1. a hand-out while a step is pending leaves every execution with at most one waiter, and
   the drain happens before the first side-cache access of the new slot;
2. a step of continuing requests only (no hand-out) does not drain: the intermediate chunks
   of a segmented prefill keep their off-submit-path drain;
3. without a pending step (synchronous scheduling, or the first step) nothing is drained
   and the slot is emptied exactly as before;
4. a pending step the output thread has already read back costs the hand-out one idempotent
   call and registers no waiter;
5. against the real ``AsyncNeuronModelRunnerOutput`` holding an all-partial step whose future
   is a tensor that counts its read-backs: two hand-outs in one step and the output thread's
   own call read the future back once, and leave the step's sampled ids as the empty lists
   the all-partial drain writes.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_slot_drain.py
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput

from vllm_neuron.vllm.worker import glm5next_state_banks
from vllm_neuron.vllm.worker.neuron_model_runner import (
    AsyncNeuronModelRunnerOutput,
    NeuronModelRunner,
)

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The runtime registers one completion handle per in-flight execution; a second waiter
#: overwrites the first ("Completion handle already set for sequence ..., overwriting").
HANDLES_PER_SEQUENCE = 1
#: A step's future is read back from the device once; every later ``get_output`` sees the
#: materialized lists and returns them.
READBACKS_PER_STEP = 1
SLOTS = 4
LINEAR, SPARSE = 1, 2
POOL = 4
WIDTH = 8
MAX_SEQ_LEN = 64
DTYPE = torch.bfloat16
#: The in-flight execution of the scenario: the aborted request's chunk at 3072 (dsa8k-pc
#: rank-0 sequence 0x1000000000b4d).
IN_FLIGHT = 0xB4D


class _Runtime:
    """The waiters of every in-flight execution, as the runtime's completion handles see them."""

    def __init__(self) -> None:
        self.waiters: dict[int, list[str]] = {}
        self.complete: set[int] = set()

    def wait(self, sequence: int, who: str) -> None:
        if sequence in self.complete:
            return
        self.waiters.setdefault(sequence, []).append(who)

    def finish(self, sequence: int) -> None:
        self.complete.add(sequence)

    def most_waiters(self) -> int:
        return max((len(w) for w in self.waiters.values()), default=0)


class _PendingStep:
    """Stands in for ``AsyncNeuronModelRunnerOutput`` whose drain is in progress.

    The output thread is already inside ``get_output`` waiting on ``sequence`` (hang dump:
    ``WorkerAsyncOutputCopy`` at ``get_output``:206). A call from another thread takes the
    lock, so it returns when that wait has returned: the execution is complete by then.
    """

    def __init__(self, runtime: _Runtime, sequence: int, *, drained: bool = False) -> None:
        self.runtime, self.sequence, self.calls = runtime, sequence, 0
        if drained:
            runtime.finish(sequence)
        else:
            runtime.wait(sequence, "WorkerAsyncOutputCopy")

    def get_output(self):
        self.calls += 1
        self.runtime.finish(self.sequence)
        return None


class _Future(torch.Tensor):
    """A sampled-token future that counts its device-to-host read-backs (``.cpu()``)."""

    @staticmethod
    def __new__(cls, data: torch.Tensor, log: list):
        future = torch.Tensor._make_subclass(cls, data)
        future.log = log
        return future

    def cpu(self, *args, **kwargs):
        self.log.append("cpu")
        return torch.Tensor.cpu(self, *args, **kwargs)


def _banks() -> list[dict]:
    banks = [{"name": "linear.0", "family": "linear_attn", "state_slots": SLOTS}]
    for index in range(SPARSE):
        banks.append({
            "name": f"sparse.{index}", "family": "self_attn", "block_size": POOL,
            "latent_cache": torch.zeros((4, 1, POOL, WIDTH), dtype=DTYPE),
        })
    return banks


def _runner(pending):
    banks = _banks()
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.max_num_reqs = SLOTS
    runner.max_model_len = MAX_SEQ_LEN
    side = NeuronModelRunner._glm5next_side_caches(
        banks, index_kpool=POOL, index_head_dim=WIDTH, max_seq_len=MAX_SEQ_LEN,
        request_slots=SLOTS + glm5next_state_banks.SCRATCH_SLOTS,
    )
    # A previous owner's leftovers in every slot, so an emptied slot is visible.
    for entry in side:
        for key in ("pool_cache", "tail"):
            if key in entry:
                entry[key].fill_(1.0)
    runner._glm5next_side_cache_set = side
    runner._glm5next_request_slot_table = {}
    runner._glm5next_side_cache_positions = {}
    runner.use_async_scheduling = pending is not None
    runner.async_execution_buffer = {} if pending is None else {"async_output": pending}
    return runner, banks, side


def _side_cache_reads(monkeypatch, runtime: _Runtime, in_flight: int | None, events: list):
    """``empty_slot``'s ordering read as the runtime sees it: a wait on the execution that is
    still writing the bank, if any, then the real emptying."""
    real = glm5next_state_banks.empty_slot

    def read_then_empty(bank, slot):
        if in_flight is not None:
            runtime.wait(in_flight, "MainThread")
        events.append(("empty_slot", int(slot)))
        real(bank, slot)

    monkeypatch.setattr(glm5next_state_banks, "empty_slot", read_then_empty)


def _assert_slot_fresh(side, slot: int) -> None:
    for entry in side:
        for key in ("pool_cache", "tail"):
            if key in entry:
                assert torch.equal(entry[key][slot], torch.zeros_like(entry[key][slot])), key


def test_a_hand_out_during_a_pending_step_leaves_one_waiter_per_execution(monkeypatch):
    runtime = _Runtime()
    pending = _PendingStep(runtime, IN_FLIGHT)
    runner, banks, side = _runner(pending)
    events: list = []
    _side_cache_reads(monkeypatch, runtime, IN_FLIGHT, events)
    # The aborted request held slot 0; the engine reported it finished while its chunk runs.
    runner._glm5next_request_slot_table["aborted"] = 0
    runner._glm5next_note_finished_requests(["aborted"])

    slots = runner._glm5next_request_slots(banks, ["new"], synthetic=False, side_caches=side)

    assert slots == [0], "the new request takes the lowest free slot, the aborted one's"
    assert runtime.most_waiters() <= HANDLES_PER_SEQUENCE, (
        f"two threads waited on execution {IN_FLIGHT:#x}: {runtime.waiters[IN_FLIGHT]}; the "
        "runtime overwrites the first handle and that thread never wakes"
    )
    assert pending.calls >= 1, "the hand-out did not drain the pending step through get_output"
    assert IN_FLIGHT in runtime.complete, (
        "the drain must complete the step before the slot is read"
    )
    assert events == [("empty_slot", 0)] * (2 * SPARSE), events
    _assert_slot_fresh(side, 0)


def test_continuing_requests_do_not_drain_the_pending_step(monkeypatch):
    runtime = _Runtime()
    pending = _PendingStep(runtime, IN_FLIGHT)
    runner, banks, side = _runner(pending)
    events: list = []
    _side_cache_reads(monkeypatch, runtime, IN_FLIGHT, events)
    runner._glm5next_request_slot_table.update({"a": 0, "b": 1})

    slots = runner._glm5next_request_slots(banks, ["a", "b"], synthetic=False, side_caches=side)

    assert slots == [0, 1]
    assert pending.calls == 0, (
        "a step without a hand-out must leave the chunk's drain to the output thread"
    )
    assert events == []
    assert runtime.waiters == {IN_FLIGHT: ["WorkerAsyncOutputCopy"]}


def test_without_a_pending_step_the_hand_out_only_empties_the_slot(monkeypatch):
    runtime = _Runtime()
    runner, banks, side = _runner(None)
    events: list = []
    _side_cache_reads(monkeypatch, runtime, None, events)

    slots = runner._glm5next_request_slots(banks, ["new"], synthetic=False, side_caches=side)

    assert slots == [0]
    assert events == [("empty_slot", 0)] * (2 * SPARSE)
    assert runtime.waiters == {}
    _assert_slot_fresh(side, 0)


def test_a_pending_step_already_read_back_costs_one_idempotent_call(monkeypatch):
    runtime = _Runtime()
    pending = _PendingStep(runtime, IN_FLIGHT, drained=True)
    runner, banks, side = _runner(pending)
    events: list = []
    _side_cache_reads(monkeypatch, runtime, IN_FLIGHT, events)

    slots = runner._glm5next_request_slots(banks, ["new"], synthetic=False, side_caches=side)

    assert slots == [0]
    assert pending.calls == 1
    assert runtime.waiters == {}, "a completed execution takes no waiter"
    assert events == [("empty_slot", 0)] * (2 * SPARSE)


def test_the_drain_through_the_real_async_output_reads_the_future_back_once(monkeypatch):
    runtime = _Runtime()
    events: list = []
    readbacks: list = []
    runner, banks, side = _runner(None)
    _side_cache_reads(monkeypatch, runtime, None, events)
    # The aborted request's intermediate chunk: every request of the step is partial, so
    # ``get_output`` drains the future and writes empty lists (the all-partial branch).
    pending = AsyncNeuronModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["aborted"], req_id_to_index={"aborted": 0},
            sampled_token_ids=_Future(torch.zeros(1, dtype=torch.int64), readbacks),
        ),
        model_runner=runner,
        partial_prefill_req_ids={"aborted"},
    )
    assert pending.is_all_partial_prefill()
    runner.use_async_scheduling = True
    runner.async_execution_buffer = {"async_output": pending}
    runner._glm5next_request_slot_table["aborted"] = 0
    runner._glm5next_note_finished_requests(["aborted"])

    slots = runner._glm5next_request_slots(
        banks, ["new-a", "new-b"], synthetic=False, side_caches=side
    )
    assert not torch.is_tensor(pending.model_runner_output.sampled_token_ids), (
        "the hand-out did not drain the pending step through get_output"
    )
    output_thread_view = pending.get_output()   # the output thread's own call, later

    assert slots == [0, 1], "two hand-outs in one step, the aborted request's slot first"
    assert readbacks == ["cpu"] * READBACKS_PER_STEP, (
        f"the future was read back {len(readbacks)} times over two hand-outs and the output "
        f"thread's call; every call after the first must find the materialized lists"
    )
    assert output_thread_view is pending.model_runner_output
    assert output_thread_view.sampled_token_ids == [[] for _ in output_thread_view.req_ids]
    assert events == [("empty_slot", 0)] * (2 * SPARSE) + [("empty_slot", 1)] * (2 * SPARSE)
