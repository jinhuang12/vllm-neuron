# SPDX-License-Identifier: Apache-2.0
"""Acceptance: the compiled graph loads in waves of ranks before the all-rank warmup call.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/worker/test_warmup_staged_neff_load.py \\
      -rA -s -p no:cacheprovider --timeout=5400

Loading the graph is a per-process step that talks to no other rank, and it holds the
larger part of the host memory a warmup call needs; the execution that follows the load
needs every rank at once. The load therefore runs in waves of
``VLLM_NEURON_NEFF_LOAD_WAVE_SIZE`` ranks, and the call runs on all ranks after the last
wave. A fake load stands in for the runtime's: it records which ranks are inside it and
holds a barrier one wave wide, so a wave narrower than the declared width times out.

Four items, ONE test each, no ``parametrize``:

* T01 -- the loads run in waves of the declared width, in wave order, never across waves.
* T02 -- the warmup call starts only after the last wave's loads have left.
* T03 -- every rank writes one row on each side of its own load, carrying its wave.
* T04 -- the control: with the wave size at the world size, every rank loads at one instant.

T01 and T04 print the largest number of ranks inside the load at once, so the transcript
carries that number and not only a verdict. Run pytest with ``-s``.
"""

import math
import re
import threading

from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import run_warmup_on_all_ranks

WORLD = 8
WAVE = 2
COLLECTIVE_TIMEOUT = 2.0
LOAD_TIMEOUT = 5.0
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


class _FakeLoad:
    """A stand-in for the runtime's graph load: records who is inside it, and how many at once."""

    def __init__(self, width: int, events: list, lock: threading.Lock) -> None:
        self._together = threading.Barrier(width)
        self._events = events
        self._lock = lock
        self._inside = 0
        self.peak_inside = 0
        self.timed_out = 0

    def __call__(self) -> None:
        with self._lock:
            self._inside += 1
            self.peak_inside = max(self.peak_inside, self._inside)
            self._events.append(("enter", _LOCAL.rank))
        try:
            self._together.wait(timeout=LOAD_TIMEOUT)
        except threading.BrokenBarrierError:
            with self._lock:
                self.timed_out += 1
        with self._lock:
            self._inside -= 1
            self._events.append(("leave", _LOCAL.rank))


def _waves_in_order(events: list, world_size: int, wave_size: int) -> bool:
    """True when the loads group into consecutive waves of this width that never overlap."""
    steps = [event for event in events if event[0] in ("enter", "leave")]
    if len(steps) != 2 * world_size:
        return False
    at = 0
    for wave in range(math.ceil(world_size / wave_size)):
        ranks = set(range(wave * wave_size, min((wave + 1) * wave_size, world_size)))
        enters = steps[at : at + len(ranks)]
        leaves = steps[at + len(ranks) : at + 2 * len(ranks)]
        at += 2 * len(ranks)
        if {kind for kind, _ in enters} != {"enter"}:
            return False
        if {kind for kind, _ in leaves} != {"leave"}:
            return False
        if {rank for _, rank in enters} != ranks:
            return False
        if {rank for _, rank in leaves} != ranks:
            return False
    return True


def _run_group(monkeypatch, wave_size: int):
    """Run the runner on WORLD threads with a fake load; return the load, events and errors."""
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_WAVE_SIZE", str(wave_size))
    monkeypatch.setattr(neuron_worker, "get_tp_group", lambda: _Group(WORLD))
    exchange = _Exchange(WORLD, EXCHANGE_TIMEOUT)
    monkeypatch.setattr(neuron_worker, "tp_sum_int", exchange)
    monkeypatch.setattr(neuron_worker, "tp_barrier", exchange)
    # The collective a device forward carries: every rank of the group must enter it.
    collective = threading.Barrier(WORLD)
    events: list[tuple[str, int]] = []
    lock = threading.Lock()
    load = _FakeLoad(wave_size, events, lock)
    errors: dict[int, BaseException] = {}

    def work():
        collective.wait(timeout=COLLECTIVE_TIMEOUT)
        with lock:
            events.append(("work", _LOCAL.rank))

    def body(rank):
        _LOCAL.rank = rank
        try:
            run_warmup_on_all_ranks("prefill", "2048/kv2048", work, load)
        except BaseException as exc:  # every rank's own end is part of the reading
            errors[rank] = exc

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=EXCHANGE_TIMEOUT + LOAD_TIMEOUT + COLLECTIVE_TIMEOUT)
    assert not [thread for thread in threads if thread.is_alive()]
    return load, events, errors


def test_the_loads_run_in_waves_of_the_declared_width(monkeypatch):
    load, events, errors = _run_group(monkeypatch, WAVE)
    print(
        f"neff_load_waves|world={WORLD}|wave_size={WAVE}"
        f"|waves={math.ceil(WORLD / WAVE)}|peak_inside={load.peak_inside}"
        f"|loads={len([event for event in events if event[0] == 'enter'])}"
    )
    assert errors == {}
    assert load.timed_out == 0
    assert load.peak_inside == WAVE
    assert _waves_in_order(events, WORLD, WAVE)


def test_the_warmup_call_starts_only_after_the_last_wave(monkeypatch):
    load, events, errors = _run_group(monkeypatch, WAVE)
    last_load = max(index for index, event in enumerate(events) if event[0] == "leave")
    first_call = min(index for index, event in enumerate(events) if event[0] == "work")
    print(f"neff_load_before_call|last_load={last_load}|first_call={first_call}")
    assert errors == {}
    assert len([event for event in events if event[0] == "work"]) == WORLD
    assert last_load < first_call


def test_every_rank_writes_a_row_on_each_side_of_its_own_load(monkeypatch):
    rows = _Rows()
    monkeypatch.setattr(neuron_worker, "logger", rows)
    load, events, errors = _run_group(monkeypatch, WAVE)
    assert errors == {}
    by_rank: dict[int, list[str]] = {}
    for row in rows.rows:
        rank = re.search(r"rank=(\d+)", row)
        if rank and row.startswith(("neff_load_entered|", "neff_load_left|")):
            by_rank.setdefault(int(rank.group(1)), []).append(row)
    written = [row for rank_rows in by_rank.values() for row in rank_rows]
    print(f"neff_load_rows|ranks={len(by_rank)}|rows={len(written)}")
    print(f"neff_load_sample|{by_rank[0][0]}")
    print(f"neff_load_left_sample|{by_rank[0][1]}")
    assert sorted(by_rank) == list(range(WORLD))
    assert all(
        [row.split("|")[0] for row in rank_rows] == ["neff_load_entered", "neff_load_left"]
        for rank_rows in by_rank.values()
    )
    assert all(
        f"|wave={rank // WAVE}|" in row for rank, rank_rows in by_rank.items() for row in rank_rows
    )
    assert all("rss_kib=" in row for row in written)
    assert [row for row in written if '"' in row or "'" in row] == []


def test_the_control_with_the_wave_size_at_the_world_size_loads_every_rank_at_once(monkeypatch):
    load, events, errors = _run_group(monkeypatch, WORLD)
    print(
        f"neff_load_control|world={WORLD}|wave_size={WORLD}"
        f"|peak_inside={load.peak_inside}|waves_in_order={_waves_in_order(events, WORLD, WAVE)}"
    )
    assert errors == {}
    assert load.timed_out == 0
    assert load.peak_inside == WORLD
    assert _waves_in_order(events, WORLD, WORLD)
    assert not _waves_in_order(events, WORLD, WAVE)
