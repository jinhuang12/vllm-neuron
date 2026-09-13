# SPDX-License-Identifier: Apache-2.0
"""Acceptance: the warmup model call runs on every rank at once, and graph capture does not.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/worker/test_warmup_all_rank_execution.py \\
      -rA -s -p no:cacheprovider --timeout=5400

A warmup call executes the model, and the model's forward carries collectives over the
whole tensor-parallel group. So the ranks a wave leaves out wait in the status exchange
while the ranks inside the wave wait for them in the collective, and neither side can
move. What a wave CAN bound is a step that talks to no other rank.

Six items, ONE test each, no ``parametrize``:

* T01 -- the all-rank runner completes ``work`` that holds an all-rank collective.
* T02 -- the wave runner completes that same ``work`` on NO rank.
* T03 -- prefill graph capture never calls the model.
* T04 -- the prefill warmup call does call it, which is what makes T03's zero a reading.
* T05 -- prefill warmup drives the all-rank runner and not the wave runner.
* T06 -- decode warmup does the same.
* T07 -- every rank logs its resident size on both sides of the executing call.

T01 and T02 are one measurement in two arms and both print the ranks that finished, so
the transcript carries the numbers rather than only a verdict. Run pytest with ``-s``.
The collective in T02 is given a timeout, because an unbounded wait is the defect itself.
"""

import re
import threading
import types

from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker, run_warmup_on_all_ranks

# The model runner and the runtime dep it imports are loaded inside the two items that need
# them, never at module scope: importing ``libtorch_neuronx_lite`` presets
# NEURON_RT_ROOT_COMM_ID for the whole process, and the worker defers it for that reason.

WORLD = 4
COLLECTIVE_TIMEOUT = 2.0
EXCHANGE_TIMEOUT = 20.0

_LOCAL = threading.local()


class _Group:
    """A TP group that reports the world size and the calling thread's rank."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size

    @property
    def rank_in_group(self) -> int:
        return _LOCAL.rank


class _Exchange:
    """A real cross-thread integer sum: every rank arrives, or the waiters time out."""

    def __init__(self, world_size: int, timeout: float) -> None:
        self._world = world_size
        self._timeout = timeout
        self._cond = threading.Condition()
        self._turn = 0
        self._arrived = 0
        self._pending = 0
        self._results: dict[int, int] = {}

    def __call__(self, value: int = 0) -> int:
        with self._cond:
            turn = self._turn
            self._arrived += 1
            self._pending += value
            if self._arrived == self._world:
                self._results[turn] = self._pending
                self._arrived = 0
                self._pending = 0
                self._turn += 1
                self._cond.notify_all()
            elif not self._cond.wait_for(
                lambda: turn in self._results, timeout=self._timeout
            ):
                raise TimeoutError(f"exchange {turn}: not every rank arrived")
            return self._results[turn]


def _run_group(monkeypatch, runner):
    """Run ``runner`` on WORLD threads; return (ranks that finished work, errors by rank)."""
    exchange = _Exchange(WORLD, EXCHANGE_TIMEOUT)
    # The collective a device forward carries: every rank of the group must enter it.
    collective = threading.Barrier(WORLD)
    finished: list[int] = []
    errors: dict[int, BaseException] = {}

    monkeypatch.setattr(neuron_worker, "get_tp_group", lambda: _Group(WORLD))
    monkeypatch.setattr(neuron_worker, "tp_sum_int", exchange)
    monkeypatch.setattr(neuron_worker, "tp_barrier", exchange)
    monkeypatch.setenv("VLLM_NEURON_WARMUP_WAVE_SIZE", "1")

    def work():
        collective.wait(timeout=COLLECTIVE_TIMEOUT)
        finished.append(_LOCAL.rank)

    def body(rank):
        _LOCAL.rank = rank
        try:
            runner("prefill", "2048/kv2048", work)
        except BaseException as exc:  # every rank's own end is part of the reading
            errors[rank] = exc

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=EXCHANGE_TIMEOUT + COLLECTIVE_TIMEOUT)
    assert not [thread for thread in threads if thread.is_alive()]
    return sorted(finished), errors


class _Rows:
    """A logger that keeps every row it renders, so a test can read what a rank wrote."""

    def __init__(self) -> None:
        self.rows: list[str] = []
        self._lock = threading.Lock()

    def info(self, message, *args) -> None:
        with self._lock:
            self.rows.append(message % args if args else message)

    def debug(self, *args, **kwargs) -> None:
        pass


def _capture_stub(model_calls, capture_calls):
    """A model runner holding only what prefill capture and prefill warmup read."""
    from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete

    def model(**kwargs):
        model_calls.append(kwargs)
        return object()

    def capture_backend_model(**kwargs):
        capture_calls.append(kwargs)
        raise CaptureComplete()

    return types.SimpleNamespace(
        model=model,
        capture_backend_model=capture_backend_model,
        drafter=None,
        kv_cache_config=object(),
        use_async_scheduling=False,
        _tensor_replacer=None,
        _build_prefill_synthetic_inputs=lambda bucket, kv, device=None: {
            "attn_metadata": object()
        },
        _glm5next_model_kwargs=lambda kwargs: kwargs,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(model="test-model")
        ),
    )


def test_the_all_rank_runner_completes_an_all_rank_collective(monkeypatch):
    finished, errors = _run_group(monkeypatch, run_warmup_on_all_ranks)
    print(f"warmup_all_ranks|world={WORLD}|finished={finished}|errors={len(errors)}")
    assert finished == list(range(WORLD))
    assert errors == {}


def test_the_wave_runner_completes_it_on_no_rank(monkeypatch):
    finished, errors = _run_group(monkeypatch, neuron_worker.run_warmup_in_waves)
    print(f"warmup_waves|world={WORLD}|finished={finished}|errors={len(errors)}")
    assert finished == []
    assert len(errors) == WORLD


def test_prefill_graph_capture_never_calls_the_model():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    model_calls, capture_calls = [], []
    runner = _capture_stub(model_calls, capture_calls)
    NeuronModelRunner.extract_prefill_graphs(runner, 2048, 2048)
    print(f"capture_step|model_calls={len(model_calls)}|capture_calls={len(capture_calls)}")
    assert model_calls == []
    assert len(capture_calls) == 1


def test_the_prefill_warmup_call_does_call_the_model():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    model_calls, capture_calls = [], []
    runner = _capture_stub(model_calls, capture_calls)
    NeuronModelRunner.warmup_prefill(runner, 2048, 2048)
    print(f"warmup_step|model_calls={len(model_calls)}|capture_calls={len(capture_calls)}")
    assert len(model_calls) == 1
    assert capture_calls == []


def test_prefill_warmup_drives_the_all_rank_runner(monkeypatch):
    drivers = []
    monkeypatch.setattr(
        neuron_worker,
        "run_warmup_on_all_ranks",
        lambda phase, bucket, work: drivers.append(("all_ranks", phase)),
    )
    monkeypatch.setattr(
        neuron_worker,
        "run_warmup_in_waves",
        lambda phase, bucket, work: drivers.append(("waves", phase)),
    )
    worker = types.SimpleNamespace(_prefill_buckets=lambda: ([2048], [2048]))
    NeuronWorker._warmup_prefill(worker)
    assert drivers == [("all_ranks", "prefill")]


def test_decode_warmup_drives_the_all_rank_runner(monkeypatch):
    drivers = []
    monkeypatch.setattr(
        neuron_worker,
        "run_warmup_on_all_ranks",
        lambda phase, bucket, work: drivers.append(("all_ranks", phase)),
    )
    monkeypatch.setattr(
        neuron_worker,
        "run_warmup_in_waves",
        lambda phase, bucket, work: drivers.append(("waves", phase)),
    )
    worker = types.SimpleNamespace(_decode_compile_targets=lambda: [(1, 2048)])
    NeuronWorker._warmup_decode(worker)
    assert drivers == [("all_ranks", "decode")]


def test_each_rank_logs_its_resident_size_on_both_sides_of_the_call(monkeypatch):
    rows = _Rows()
    monkeypatch.setattr(neuron_worker, "logger", rows)
    finished, errors = _run_group(monkeypatch, run_warmup_on_all_ranks)
    assert finished == list(range(WORLD))
    assert errors == {}
    by_rank: dict[int, list[str]] = {}
    for row in rows.rows:
        rank = re.search(r"rank=(\d+)", row)
        if rank and row.startswith(("warmup_execution_entered|", "warmup_execution_left|")):
            by_rank.setdefault(int(rank.group(1)), []).append(row)
    written = [row for rank_rows in by_rank.values() for row in rank_rows]
    print(f"warmup_rss_rows|ranks={len(by_rank)}|rows={len(written)}")
    print(f"warmup_rss_sample|{by_rank[0][0]}")
    assert sorted(by_rank) == list(range(WORLD))
    assert all(
        [row.split("|")[0] for row in rank_rows]
        == ["warmup_execution_entered", "warmup_execution_left"]
        for rank_rows in by_rank.values()
    )
    assert all("rss_kib=" in row for row in written)
    assert [row for row in written if '"' in row or "'" in row] == []
