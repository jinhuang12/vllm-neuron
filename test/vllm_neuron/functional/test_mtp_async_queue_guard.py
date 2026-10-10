# SPDX-License-Identifier: Apache-2.0
"""``functional/mtp/async_step.py``: the async drafter never raises on a full execution queue.

The runtime queues each rank's NEFF executions and frees a slot when its completion thread
has processed the completion; a launch that finds the queue full returns ``NRT_QUEUE_FULL``
(status 7), which the executor raises as ``RuntimeError("Failed to schedule neff execution.
status=7 message=Execution Queue Full")``. Two guards hold the drafter to "degrade, never
raise":

* **the degradation** -- a launch refused with exactly that status and message is served by
  the torch route on the host for that step (the operands read back once, the fields moved
  back once), with values the synchronous drafter builds, and counted once
  (``queue_full_fallbacks``); every other ``RuntimeError`` is re-raised unchanged;
* **the bound** -- :class:`DeviceLaunch` keeps at most ``launch_depth(execution_queue)`` of
  its launches unprocessed per rank (derived from the runtime's queue depth the worker
  sets, ``NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS``) and waits on the oldest before issuing
  another, so a rank whose completion thread stalls stops enqueuing instead of filling the
  queue; a read that returns at once retires a processed launch silently, a read that
  blocks is a wait, counted (``bound_waits``) and logged once; a queue too small for the
  drafter's steady state is refused by name.

The device is stood in for on the host as ``test_mtp_async_launch`` does: every operand
counts as off-host, a launch object runs the eager launch functions (the simulator answers)
and raises the runtime's error on the chosen call; the bound is driven by a launch whose
first output blocks in ``.cpu()`` until the test releases it, the way a future of an
unprocessed execution blocks.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/test_mtp_async_queue_guard.py
"""

from __future__ import annotations

import ast
import logging
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.functional.mtp import async_step
from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.functional import test_mtp_async_step as step_tests
from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as spec
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.worker import test_mtp_async_identity as identity
from test.vllm_neuron.worker import test_mtp_async_launch as launch_tests
from test.vllm_neuron.worker import test_mtp_proposer as proposer_tests

#: The executor's text for ``NRT_QUEUE_FULL`` (``libtorch_neuronx_lite`` ``backend.py``
#: ``executor.execute``), as its ``RuntimeError`` carries it.
QUEUE_FULL = "Failed to schedule neff execution. status=7 message=Execution Queue Full"
PAGE = step_tests.PAGE
T = step_tests.T


class _RefusingLaunch:
    """A :class:`DeviceLaunch` stand-in: the eager launch functions answer every call (the
    simulator runs the kernels) except call ``fail_at`` (1-based over take and correct
    together), which raises ``error`` as the executor would."""

    def __init__(self, fail_at: int, error: Exception) -> None:
        self.fail_at = fail_at
        self.error = error
        self.calls = {"take": 0, "correct": 0}
        self.issued = 0

    def _refuse_or_run(self, name: str, run):
        self.calls[name] += 1
        self.issued += 1
        if self.issued == self.fail_at:
            raise self.error
        return run()

    def take(self, accepted, drafts):
        return self._refuse_or_run("take", lambda: async_step._take_launch(accepted, drafts))

    def correct(self, prev_rows, starts, table, **constants):
        return self._refuse_or_run(
            "correct", lambda: async_step._correct_launch(prev_rows, starts, table, **constants)
        )


def _take_operands(seed: int):
    accepted, _kept = step_tests._accepted(seed, 3, T)
    return accepted, step_tests._drafts(seed + 1, 3)


def _correct_case():
    tables = step_tests._tables(51, 3, 4)
    column = step_tests._column(tables, 4, 1)
    prev = torch.tensor([0, 3, 1], dtype=torch.int32)
    optimistic = torch.tensor([PAGE + 3, 2 * PAGE, 9], dtype=torch.int32)
    return tables, column, prev, optimistic


def _assert_same_fields(got, want, where: str) -> None:
    for name, expected in zip(want._fields, want):
        value = getattr(got, name)
        if expected is None:
            assert value is None, (where, name)
            continue
        assert value.dtype == expected.dtype, (where, name, value.dtype, expected.dtype)
        assert value.shape == expected.shape, (where, name, value.shape, expected.shape)
        assert value.device == expected.device, (where, name, value.device, expected.device)
        assert value.is_contiguous() and expected.is_contiguous(), (where, name, value.stride(), expected.stride())
        assert value.tolist() == expected.tolist(), (where, name)


# ── the degradation ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_a_full_queue_at_a_take_serves_the_step_from_the_host_and_counts_once(monkeypatch, caplog, fail_at):
    """Three takes, the ``fail_at``-th refused with the queue-full error: no raise, every take
    equals the torch route bit for bit, the two others launched, the refused one counted and
    logged once as a warning that names the full queue and the launch."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = _RefusingLaunch(fail_at, RuntimeError(QUEUE_FULL))
    async_step.reset_dispatch_counters()
    with caplog.at_level(logging.WARNING, logger=async_step.__name__):
        for seed in (11, 12, 13):
            accepted, drafts = _take_operands(seed)
            got = async_step.mtp_async_take(accepted, drafts, launch=launch)
            _assert_same_fields(got, async_step.mtp_async_take_torch(accepted, drafts), f"take seed {seed}")
    assert launch.calls == {"take": 3, "correct": 0}
    assert async_step.queue_full_fallbacks() == 1
    assert async_step.dispatch_counters() == (2, 1), async_step.dispatch_counters()
    warnings = [record for record in caplog.records if record.name == async_step.__name__]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    assert warnings[0].levelno == logging.WARNING
    assert "full execution queue" in warnings[0].getMessage() and "take" in warnings[0].getMessage()


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_a_full_queue_at_a_correction_serves_the_step_from_the_host_with_the_sync_translators_operands(
    monkeypatch, fail_at,
):
    """Three corrections, the ``fail_at``-th refused: no raise; the refused step's fields are
    the synchronous translator's host operands at the true positions (what the kernel is held
    to) and the torch route's, slots int64; counted once."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = _RefusingLaunch(fail_at, RuntimeError(QUEUE_FULL))
    tables, column, prev, optimistic = _correct_case()
    want = step_tests._host_operands(
        tables, optimistic.tolist(), width=T, padding=1, prev_width=T, prev_rows=prev.tolist()
    )
    async_step.reset_dispatch_counters()
    for _ in range(3):
        got = async_step.mtp_async_correct(
            prev, optimistic, column, prev_width=T, width=T, page_size=PAGE, padding=1, launch=launch
        )
        for name, expected in want.items():
            value = getattr(got, name)
            assert value.dtype == expected.dtype, (name, value.dtype)
            assert value.tolist() == expected.tolist(), name
        assert got.latent_slots.dtype == torch.int64
        _assert_same_fields(
            got,
            async_step.mtp_async_correct_torch(
                prev, optimistic, column, prev_width=T, width=T, page_size=PAGE, padding=1
            ),
            "correct",
        )
    assert launch.calls == {"take": 0, "correct": 3}
    assert async_step.queue_full_fallbacks() == 1
    assert async_step.dispatch_counters() == (2, 1), async_step.dispatch_counters()


def test_a_full_queue_without_a_page_table_serves_the_step_from_the_host_with_no_slots(monkeypatch):
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = _RefusingLaunch(1, RuntimeError(QUEUE_FULL))
    _tables, _column, prev, optimistic = _correct_case()
    async_step.reset_dispatch_counters()
    got = async_step.mtp_async_correct(
        prev, optimistic, None, prev_width=T, width=T, page_size=PAGE, padding=1, launch=launch
    )
    assert got.latent_slots is None
    _assert_same_fields(
        got,
        async_step.mtp_async_correct_torch(prev, optimistic, None, prev_width=T, width=T, page_size=PAGE, padding=1),
        "no table",
    )
    assert async_step.queue_full_fallbacks() == 1


@pytest.mark.parametrize(
    "message",
    [
        "Failed to schedule neff execution. status=5 message=Execution Queue Full",
        "Failed to schedule neff execution. status=7 message=Timeout",
        "Failed to schedule neff execution. status=70 message=Execution Queue Full",
        "Failed to schedule neff execution. status=17 message=Execution Queue Full",
        "nrt_tensor_read failed",
    ],
)
@pytest.mark.parametrize("site", ["take", "correct"])
def test_every_other_runtime_error_at_a_launch_is_re_raised_unchanged(monkeypatch, site, message):
    """The match is status 7 AND the queue-full message, as whole tokens: another status, another
    message, or a status that merely starts with 7 re-raises the executor's own exception
    object, and nothing is counted."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    error = RuntimeError(message)
    launch = _RefusingLaunch(1, error)
    async_step.reset_dispatch_counters()
    with pytest.raises(RuntimeError) as raised:
        if site == "take":
            async_step.mtp_async_take(*_take_operands(21), launch=launch)
        else:
            _tables, column, prev, optimistic = _correct_case()
            async_step.mtp_async_correct(
                prev, optimistic, column, prev_width=T, width=T, page_size=PAGE, padding=1, launch=launch
            )
    assert raised.value is error
    assert async_step.queue_full_fallbacks() == 0
    assert async_step.dispatch_counters() == (0, 0)


def test_the_queue_full_match_is_the_status_and_the_message_as_whole_tokens():
    assert async_step._is_queue_full(RuntimeError(QUEUE_FULL))
    assert async_step._is_queue_full(RuntimeError("status=7 message=Execution Queue Full (unit 3b01)"))
    for text in (
        "status=7",
        "Execution Queue Full",
        "status=70 message=Execution Queue Full",
        "status=7.0 message=Execution Queue Full",
        "status=17 message=Execution Queue Full",
        "status=5 message=Execution Queue Full",
        "status=7 message=Execution Queue",
    ):
        assert not async_step._is_queue_full(RuntimeError(text)), text


def test_a_refused_launch_is_not_a_kernel_launch_that_returned(monkeypatch):
    """The kernel counter counts launches that returned; a refused one is counted as the
    degradation (and as the torch route that served it), never as a kernel."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    async_step.reset_dispatch_counters()
    async_step.mtp_async_take(*_take_operands(31), launch=_RefusingLaunch(1, RuntimeError(QUEUE_FULL)))
    assert async_step.dispatch_counters() == (0, 1)
    assert async_step.queue_full_fallbacks() == 1
    async_step.reset_dispatch_counters()
    assert async_step.queue_full_fallbacks() == 0


# ── the bound ─────────────────────────────────────────────────────────────────────────


class _Pending(torch.Tensor):
    """A launch's output as the host sees it while the runtime has not processed the
    execution: a host read blocks until the release. Tensor operations return plain
    tensors (``__torch_function__`` disabled), as a NEFF output fed to the next launch does."""

    __torch_function__ = torch._C._disabled_torch_function_impl
    release: threading.Event
    reads: list

    @staticmethod
    def of(tensor: torch.Tensor, release: threading.Event, reads: list) -> "_Pending":
        pending = tensor.as_subclass(_Pending)
        pending.release = release
        pending.reads = reads
        return pending

    def cpu(self):
        self.reads.append(time.monotonic())
        assert self.release.wait(timeout=30.0), "the release never came"
        return torch.Tensor.cpu(self.as_subclass(torch.Tensor))


def _stalled_graph(real, release: threading.Event, issued: list, reads: list):
    """The compiled launch function with every first output pending until ``release``."""

    def graph(*args, **kwargs):
        fields = real(*args, **kwargs)
        issued.append(len(issued) + 1)
        return (_Pending.of(fields[0], release, reads), *fields[1:])

    return graph


def test_the_launch_holds_at_most_the_depth_of_unprocessed_launches(step_tests_fresh_dynamo, monkeypatch, caplog):
    """With the runtime processing nothing, a thread issuing ``depth + 2`` takes gets exactly
    ``depth`` of them enqueued and blocks in the oldest's read; releasing the runtime lets the
    rest through and the launch holds no more than ``depth``. The one blocked read is the one
    wait: counted once and logged once with the launches outstanding and the time waited; the
    read after the release returns at once and is neither."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    queue = _workers_queue_setting()
    launch = async_step.DeviceLaunch("eager", execution_queue=queue)
    release, issued, reads = threading.Event(), [], []
    launch._take = _stalled_graph(async_step._take_launch, release, issued, reads)
    accepted, drafts = _take_operands(41)
    depth = launch.depth
    assert depth == async_step.launch_depth(queue)
    results: list = []
    async_step.reset_dispatch_counters()
    caplog.set_level(logging.WARNING, logger=async_step.__name__)

    def issue():
        for _ in range(depth + 2):
            results.append(async_step.mtp_async_take(accepted, drafts, launch=launch))

    worker = threading.Thread(target=issue, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not reads:
        time.sleep(0.01)
    time.sleep(0.2)  # a bound that is off by one issues the next launch in this window
    assert worker.is_alive(), "the issuing thread did not block on the oldest launch"
    assert len(issued) == depth, (len(issued), depth)
    assert launch.outstanding == depth
    assert len(reads) == 1, reads
    release.set()
    worker.join(timeout=30.0)
    assert not worker.is_alive()
    assert len(issued) == depth + 2
    assert len(results) == depth + 2
    assert launch.outstanding <= depth
    assert launch.calls == {"take": depth + 2, "correct": 0}
    assert len(reads) == 2, reads  # one read per launch past the depth
    waits = [record.getMessage() for record in caplog.records if record.name == async_step.__name__]
    assert len(waits) == 1 and async_step.bound_waits() == 1, waits
    assert f"{depth} launches outstanding" in waits[0] and "the bound waited" in waits[0], waits
    waited_ms = float(waits[0].split("waited ")[1].split(" ms")[0])
    assert waited_ms >= 200.0 > async_step.WAIT_THRESHOLD_MS, waited_ms


LAUNCHES_OF_A_HEALTHY_RUN = 120


def test_a_processed_launch_is_let_through_without_a_wait(step_tests_fresh_dynamo, monkeypatch, caplog):
    """A healthy host: every launch is processed before the next arrives. Over
    ``LAUNCHES_OF_A_HEALTHY_RUN`` takes the bound reads one processed output per launch past
    the depth, retires it at once, counts no wait and logs nothing."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = async_step.DeviceLaunch("eager", execution_queue=_workers_queue_setting())
    release, issued, reads = threading.Event(), [], []
    release.set()
    launch._take = _stalled_graph(async_step._take_launch, release, issued, reads)
    accepted, drafts = _take_operands(42)
    depth = launch.depth
    async_step.reset_dispatch_counters()
    caplog.set_level(logging.WARNING, logger=async_step.__name__)
    for _ in range(LAUNCHES_OF_A_HEALTHY_RUN):
        async_step.mtp_async_take(accepted, drafts, launch=launch)
    assert len(issued) == LAUNCHES_OF_A_HEALTHY_RUN
    assert len(reads) == LAUNCHES_OF_A_HEALTHY_RUN - depth
    assert launch.outstanding == depth
    assert async_step.bound_waits() == 0
    assert [record.getMessage() for record in caplog.records if record.name == async_step.__name__] == []


def test_a_launch_without_an_execution_queue_has_no_bound(step_tests_fresh_dynamo, monkeypatch):
    """A launch built without a queue (a host backend: launches run eagerly, nothing is
    queued) has no depth and never reads an output back: twenty takes whose outputs would
    block in a read all issue."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = async_step.DeviceLaunch("eager")
    assert launch.depth is None
    release, issued, reads = threading.Event(), [], []
    launch._take = _stalled_graph(async_step._take_launch, release, issued, reads)
    accepted, drafts = _take_operands(43)
    for _ in range(20):
        async_step.mtp_async_take(accepted, drafts, launch=launch)
    assert len(issued) == 20 and reads == [] and launch.outstanding == 0


@pytest.fixture
def step_tests_fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _workers_queue_setting() -> int:
    """The execution-queue depth the worker sets for the runtime
    (``os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", <n>)``)."""
    tree = ast.parse(pathlib.Path(neuron_worker.__file__).read_text())
    values = [
        int(node.args[1].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setdefault"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS"
    ]
    assert len(values) == 1, f"the worker sets the runtime's queue depth at {len(values)} site(s)"
    return values[0]


def _executions_held_by_enumeration(launches: int) -> int:
    """The most executions a rank has enqueued but not seen processed while ``launches`` of
    the drafter's launches are unprocessed, by walking the enqueue order of a served
    sequence -- the prefill's take, then per step the correction, the forward, the take --
    over every window of ``launches`` launches: the launches, the forwards between them, and
    the forward already enqueued when the launch that waits is a take."""
    order = ["take"] + ["correct", "forward", "take"] * 8
    launch_sites = [i for i, kind in enumerate(order) if kind != "forward"]
    held = []
    for first in range(len(launch_sites) - launches):
        waiting = launch_sites[first + launches]
        # executions from the oldest unprocessed launch up to, not including, the waiting launch
        held.append(waiting - launch_sites[first])
    return max(held)


@pytest.mark.parametrize("launches", range(1, 13))
def test_the_executions_a_rank_holds_for_its_unprocessed_launches(launches):
    assert async_step.DECODE_STEP_ENQUEUES == ("correct", "forward", "take")
    assert async_step.executions_held(launches) == _executions_held_by_enumeration(launches)


def test_the_healthy_hosts_outstanding_launches_derive_from_the_step_and_the_engines_batches(monkeypatch):
    """A healthy host holds the previous step's take (read only by the next launch) plus the
    launches of every batch the engine keeps in flight; the drafter's launches per step come
    from the step's enqueue order, the batches from the async drafter's engine config
    (``max_concurrent_batches`` under async scheduling)."""
    launches_per_step = sum(1 for enqueue in async_step.DECODE_STEP_ENQUEUES if enqueue != "forward")
    assert async_step.LAUNCHES_PER_STEP == launches_per_step == 2
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    config = identity._async_config(identity.K)
    assert config.scheduler_config.async_scheduling is True
    assert async_step.ENGINE_BATCHES_IN_FLIGHT == config.max_concurrent_batches == 2
    assert async_step.STEADY_STATE_LAUNCHES == 1 + launches_per_step * async_step.ENGINE_BATCHES_IN_FLIGHT == 5


@pytest.mark.parametrize("queue", range(2, 65))
def test_the_launch_depth_is_the_largest_whose_executions_fit_half_the_queue(queue):
    """The depth is the most unprocessed launches whose executions (with the forwards
    between them) fit half the execution queue; one more would not. The other half stays free
    for executions this count does not include."""
    depth = async_step.launch_depth(queue)
    assert depth >= 0
    assert async_step.executions_held(depth) <= queue // 2 < async_step.executions_held(depth + 1), (queue, depth)


def test_the_launch_depth_at_the_workers_queue_and_at_the_runtimes_default():
    """The worker's queue (read from its source) admits the drafter with a depth above the
    launches a healthy host holds; the runtime's own default of 8 derives a depth below it."""
    queue = _workers_queue_setting()
    assert async_step.launch_depth(queue) == 10 and queue == 32, (queue, async_step.launch_depth(queue))
    assert async_step.launch_depth(queue) >= async_step.STEADY_STATE_LAUNCHES
    assert async_step.launch_depth(8) == 2 < async_step.STEADY_STATE_LAUNCHES
    assert async_step.DeviceLaunch("eager", execution_queue=queue).depth == 10


def test_the_minimum_execution_queue_is_the_smallest_that_never_waits_on_a_healthy_host():
    """``MINIMUM_EXECUTION_QUEUE`` is the least queue whose depth reaches the launches a
    healthy host holds (take N, the correction and take of N+1 and N+2 under the engine's two
    batches): one below it derives a smaller depth."""
    minimum = async_step.MINIMUM_EXECUTION_QUEUE
    assert async_step.launch_depth(minimum) == async_step.STEADY_STATE_LAUNCHES == 5
    assert async_step.launch_depth(minimum - 1) < async_step.STEADY_STATE_LAUNCHES
    assert async_step.DeviceLaunch("eager", execution_queue=minimum).depth == async_step.STEADY_STATE_LAUNCHES


@pytest.mark.parametrize("below", [0, 1, 7])
def test_a_queue_too_small_for_the_drafters_steady_state_is_refused_by_name(below):
    """A launch under a queue whose depth is below the launches a healthy host holds would
    wait on every step; it is refused with the setting's name, its value and the minimum."""
    queue = async_step.MINIMUM_EXECUTION_QUEUE - 1 - below
    assert queue >= 1
    with pytest.raises(async_step.MtpAsyncStepError) as raised:
        async_step.DeviceLaunch("eager", execution_queue=queue)
    text = str(raised.value)
    assert async_step.EXECUTION_QUEUE_ENV in text and f"={queue}" in text, text
    assert str(async_step.MINIMUM_EXECUTION_QUEUE) in text, text
    assert async_step.EXECUTION_QUEUE_ENV == "NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS"


class _RecordingLaunch:
    """Stands in for :class:`DeviceLaunch` at the runner's builder: records the arguments."""

    built: list = []

    def __init__(self, backend, options=None, *, execution_queue=None) -> None:
        self.backend, self.options, self.execution_queue = backend, options, execution_queue
        _RecordingLaunch.built.append(self)


def test_the_runner_hands_the_launch_the_runtimes_execution_queue_or_refuses_by_name(monkeypatch):
    """The runner's launch builder names the execution queue the worker set for the runtime
    (``NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS``) to the launch it builds; with the setting
    absent it refuses by name instead of building an unbounded launch."""
    monkeypatch.setattr(async_step, "DeviceLaunch", _RecordingLaunch)
    _RecordingLaunch.built.clear()
    runner = SimpleNamespace(
        device=torch.device("meta"), _glm5next_async_launch_instance=None, compile_options={"option": 1}
    )
    monkeypatch.setenv(async_step.EXECUTION_QUEUE_ENV, "24")
    launch = NeuronModelRunner._glm5next_async_launch(runner)
    assert isinstance(launch, _RecordingLaunch) and launch.execution_queue == 24
    assert launch.options == {"option": 1} and runner._glm5next_async_launch_instance is launch
    assert NeuronModelRunner._glm5next_async_launch(runner) is launch and len(_RecordingLaunch.built) == 1
    runner._glm5next_async_launch_instance = None
    monkeypatch.delenv(async_step.EXECUTION_QUEUE_ENV)
    with pytest.raises(async_step.MtpAsyncStepError, match=async_step.EXECUTION_QUEUE_ENV):
        NeuronModelRunner._glm5next_async_launch(runner)
    assert len(_RecordingLaunch.built) == 1


# ── the degradation inside the served runner ──────────────────────────────────────────


class _RefusingDeviceLaunch(async_step.DeviceLaunch):
    """The runner's :class:`DeviceLaunch` on the eager backend, bounded by the worker's queue,
    whose ``fail_at``-th launch (1-based over take and correct together, in the order the
    runner issues them) raises ``error`` from the compiled launch call -- where the executor
    raises it -- so the base class's admission and ``calls`` run as they do in the product
    (``calls`` counts the refused launch too: the count precedes the call); ``issued`` counts
    every launch call made."""

    def __init__(self, fail_at: int, error: Exception) -> None:
        super().__init__("eager", execution_queue=_workers_queue_setting())
        self.fail_at = fail_at
        self.error = error
        self.issued = 0
        self._take = self._refusing(self._take)
        self._correct = self._refusing(self._correct)

    def _refusing(self, graph):
        def launch(*args, **kwargs):
            self.issued += 1
            if self.issued == self.fail_at:
                raise self.error
            return graph(*args, **kwargs)

        return launch


#: The runner's launches in order: the prefill's take at ``sample_tokens``, then per decode
#: step the correction at ``execute_model`` and the take at ``sample_tokens``.
LAUNCH_SITES = {"prefill take": 1, "first correction": 2, "first decode take": 3, "second correction": 4}
TOKENS = launch_tests.TOKENS


def _synchronous_drafters_ids(tmp_path, monkeypatch, prompt: list[int]) -> list[int]:
    """The control of ``test_mtp_async_identity``: the synchronous drafter (the shipped
    opt-in path, async scheduling off) served ``prompt`` for ``TOKENS`` tokens."""
    monkeypatch.delenv(proposer_tests.ASYNC_KNOB, raising=False)
    monkeypatch.setenv(spec.KNOB, str(identity.K))
    config = spec._config(identity.K)
    assert config.scheduler_config.async_scheduling is False
    (tmp_path / "sync").mkdir()
    with fr._parallel_state(tmp_path / "sync", config):
        root, _ = spec._root()
        runner = spec._runner(config, root)
        assert runner.drafter.async_steps is False
        ids, _steps = spec._generate(runner, "sync", prompt, tokens=TOKENS, finished=set())
    monkeypatch.delenv(spec.KNOB)
    return ids


@pytest.mark.forked
@pytest.mark.parametrize("site", list(LAUNCH_SITES))
def test_a_full_queue_at_one_launch_of_a_served_sequence_leaves_the_synchronous_drafters_ids(
    tmp_path, monkeypatch, site
):
    """The real runner under the async drafter, the device stood in for as
    ``test_mtp_async_launch`` does, with the launch at ``site`` refused by a full queue: no
    raise, the ids are the synchronous drafter's, every step stayed an async step (the runner
    counted no synchronous fallback), the refused launch is the one queue-full event and the
    one torch-route call, every other launch returned from the device launch (the kernel
    count), none ran eagerly, and the launch never held more than its depth."""
    config, prompt = launch_tests._async_runner(tmp_path, monkeypatch)
    control = _synchronous_drafters_ids(tmp_path, monkeypatch, prompt)
    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    launch = _RefusingDeviceLaunch(LAUNCH_SITES[site], RuntimeError(QUEUE_FULL))
    seen = launch_tests._emulated_device(monkeypatch, lambda: launch)
    async_step.reset_dispatch_counters()
    with fr._parallel_state(tmp_path / "async", config):
        root, _ = spec._root()
        runner = spec._runner(config, root)
        assert runner.use_async_scheduling and runner.drafter.async_steps is True
        ids, steps, async_steps, fallbacks = identity._generate_async(
            runner, "async-queue-full", prompt, tokens=TOKENS, finished=set()
        )
    assert ids == control, (site, ids, control)
    assert fallbacks == 0 and async_steps == steps, (site, steps, async_steps, fallbacks)
    launches = 1 + 2 * steps
    assert LAUNCH_SITES[site] <= launches, (site, launches)
    assert launch.issued == launches and launch.calls == {"take": steps + 1, "correct": steps}, (launch.issued, launch.calls)
    assert launch.depth == async_step.launch_depth(_workers_queue_setting()) and launch.outstanding <= launch.depth
    assert seen["take"] == {"eager": 0} and seen["correct"] == {"eager": 0}
    assert async_step.queue_full_fallbacks() == 1
    assert async_step.dispatch_counters() == (launches - 1, 1), async_step.dispatch_counters()
