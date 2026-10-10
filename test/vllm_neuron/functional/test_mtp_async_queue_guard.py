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
* **the bound** -- :class:`DeviceLaunch` keeps at most ``LAUNCH_DEPTH`` of its launches
  unprocessed per rank and waits on the oldest before issuing another, so a rank whose
  completion thread stalls stops enqueuing instead of filling the queue.

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
import pathlib
import threading
import time

import pytest
import torch

from vllm_neuron.functional.mtp import async_step
from vllm_neuron.vllm.worker import neuron_worker

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
        assert value.tolist() == expected.tolist(), (where, name)


# ── the degradation ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_a_full_queue_at_a_take_serves_the_step_from_the_host_and_counts_once(monkeypatch, fail_at):
    """Three takes, the ``fail_at``-th refused with the queue-full error: no raise, every take
    equals the torch route bit for bit, the two others launched, the refused one counted."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = _RefusingLaunch(fail_at, RuntimeError(QUEUE_FULL))
    async_step.reset_dispatch_counters()
    for seed in (11, 12, 13):
        accepted, drafts = _take_operands(seed)
        got = async_step.mtp_async_take(accepted, drafts, launch=launch)
        _assert_same_fields(got, async_step.mtp_async_take_torch(accepted, drafts), f"take seed {seed}")
    assert launch.calls == {"take": 3, "correct": 0}
    assert async_step.queue_full_fallbacks() == 1
    assert async_step.dispatch_counters() == (2, 1), async_step.dispatch_counters()


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


def test_the_launch_holds_at_most_launch_depth_unprocessed_launches(step_tests_fresh_dynamo, monkeypatch):
    """With the runtime processing nothing, a thread issuing ``LAUNCH_DEPTH + 2`` takes gets
    exactly ``LAUNCH_DEPTH`` of them enqueued and blocks in the oldest's read; releasing the
    runtime lets the rest through, and the launch holds no more than ``LAUNCH_DEPTH``."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = async_step.DeviceLaunch("eager")
    release, issued, reads = threading.Event(), [], []
    launch._take = _stalled_graph(async_step._take_launch, release, issued, reads)
    accepted, drafts = _take_operands(41)
    depth = async_step.LAUNCH_DEPTH
    results: list = []

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


def test_a_processed_launch_is_let_through_without_a_wait(step_tests_fresh_dynamo, monkeypatch):
    """Outputs the runtime has processed read back at once: ``LAUNCH_DEPTH + 2`` takes in a
    row complete with one read of the oldest per launch past the depth."""
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = async_step.DeviceLaunch("eager")
    release, issued, reads = threading.Event(), [], []
    release.set()
    launch._take = _stalled_graph(async_step._take_launch, release, issued, reads)
    accepted, drafts = _take_operands(42)
    for _ in range(async_step.LAUNCH_DEPTH + 2):
        async_step.mtp_async_take(accepted, drafts, launch=launch)
    assert len(issued) == async_step.LAUNCH_DEPTH + 2
    assert len(reads) == 2
    assert launch.outstanding == async_step.LAUNCH_DEPTH


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


def test_the_launch_depth_fits_the_workers_execution_queue_with_a_factor_two_margin():
    """The drafter's launches are two of every three executions of a decode step, so D
    unprocessed launches hold at most ``D + D // 2 + 1`` queue slots; that must fit half the
    queue the worker configures. The depth exceeds the five launches a healthy host holds
    (take N, correct and take of N+1 and N+2 under the engine's two batches), so the bound
    never waits on a healthy host, and is a whole number of launch pairs."""
    queue = _workers_queue_setting()
    depth = async_step.LAUNCH_DEPTH
    assert depth + depth // 2 + 1 <= queue // 2, (depth, queue)
    assert depth > 5
    assert depth % 2 == 0


# ── the degradation inside the served runner ──────────────────────────────────────────


class _RefusingDeviceLaunch(async_step.DeviceLaunch):
    """The runner's :class:`DeviceLaunch` on the eager backend whose ``fail_at``-th launch
    (1-based over take and correct together, in the order the runner issues them) raises
    ``error`` in place of launching; ``issued`` counts every launch asked for, ``calls`` (the
    base class's) only those that launched."""

    def __init__(self, fail_at: int, error: Exception) -> None:
        super().__init__("eager")
        self.fail_at = fail_at
        self.error = error
        self.issued = 0

    def _refuse_the_chosen_launch(self) -> None:
        self.issued += 1
        if self.issued == self.fail_at:
            raise self.error

    def take(self, accepted, drafts):
        self._refuse_the_chosen_launch()
        return super().take(accepted, drafts)

    def correct(self, prev_rows, starts, table, **constants):
        self._refuse_the_chosen_launch()
        return super().correct(prev_rows, starts, table, **constants)


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
    one torch-route call, every other launch reached the device launch, none ran eagerly."""
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
    assert launch.issued == launches and sum(launch.calls.values()) == launches - 1, (launch.issued, launch.calls)
    assert seen["take"] == {"eager": 0} and seen["correct"] == {"eager": 0}
    assert async_step.queue_full_fallbacks() == 1
    assert async_step.dispatch_counters() == (launches - 1, 1), async_step.dispatch_counters()
