# SPDX-License-Identifier: Apache-2.0
"""Acceptance: the compiled graph's builder runs in waves of ranks, signalling forward only.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/worker/test_warmup_staged_neff_load.py \\
      -rA -s -p no:cacheprovider --timeout=5400

The backend builds a graph's executable in one place and building it loads the whole graph
into host memory, so a group that builds on every rank at one instant exhausts the host. The
staged loader wraps that builder: a wave of ranks builds, signals with one flag file per rank,
and the next wave waits for those flags. The waiting points FORWARD only, because a rank that
has built is already inside the forward's collectives and cannot answer a group-wide wait.

The backend is never reached in CPU mode -- it returns the traced forward unchanged there --
so a fake builder stands in for it, and the wrapper is installed on a fake module attribute
exactly as it is installed on the vendor's.

Five items, ONE test each, no ``parametrize``:

* T01 -- the builds run in waves of the declared width, in wave order, never across waves.
* T02 -- a rank that never returns after its build does not stall the waves behind it.
* T03 -- every rank writes a row on each side of its own build, and one flag after it.
* T04 -- the control: with the wave size at the world size, every rank builds at one instant.
* T05 -- the wrapper calls the original builder once per call and returns its object.

T01 and T04 print the largest number of ranks inside the builder at once, so the transcript
carries that number and not only a verdict. Run pytest with ``-s``.
"""

import math
import re
import threading
import types

from vllm_neuron.vllm.patches import staged_neff_load

WORLD = 8
WAVE = 2
BUILD_TOGETHER_TIMEOUT = 5.0
WAIT_TIMEOUT = 5
JOIN_TIMEOUT = 30.0


class _Rows:
    """A logger that keeps every row it renders, so a test can read what a rank wrote."""

    def __init__(self) -> None:
        self.rows: list[str] = []
        self._lock = threading.Lock()

    def info(self, message, *args) -> None:
        with self._lock:
            self.rows.append(message % args if args else message)

    def warning(self, message, *args) -> None:
        self.info(message, *args)

    def debug(self, *args, **kwargs) -> None:
        pass


class _FakeBuild:
    """The vendor's builder, faked: records who is inside it, and how many at once."""

    def __init__(self, width: int | None) -> None:
        self._together = threading.Barrier(width) if width else None
        self._lock = threading.Lock()
        self._inside = 0
        self.peak_inside = 0
        self.calls = 0
        self.timed_out = 0
        self.events: list[tuple[str, int]] = []
        self.result = object()

    def __call__(self, hlo_filename=None, neff_filename=None, g_device_id=None,
                 g_device_count=None, artifacts=None):
        with self._lock:
            self._inside += 1
            self.calls += 1
            self.peak_inside = max(self.peak_inside, self._inside)
            self.events.append(("enter", g_device_id))
        if self._together is not None:
            try:
                self._together.wait(timeout=BUILD_TOGETHER_TIMEOUT)
            except threading.BrokenBarrierError:
                with self._lock:
                    self.timed_out += 1
        with self._lock:
            self._inside -= 1
            self.events.append(("leave", g_device_id))
        return self.result


def _waves_in_order(events: list, world_size: int, wave_size: int) -> bool:
    """True when the builds group into consecutive waves of this width that never overlap."""
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


def _install(monkeypatch, tmp_path, wave_size: int, build, rows=None):
    """Install the staged loader on a fake module holding this fake builder."""
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_WAVE_SIZE", str(wave_size))
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT", str(WAIT_TIMEOUT))
    if rows is not None:
        monkeypatch.setattr(staged_neff_load, "logger", rows)
    module = types.SimpleNamespace(build_executable=build)
    staged_neff_load.apply_staged_neff_load(module)
    staged_neff_load.name_this_load("prefill", "2048/kv2048")
    return module


def _run(module, park=None):
    """Call the wrapped builder once per rank on its own thread; return threads and results."""
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def body(rank):
        try:
            results[rank] = module.build_executable(
                hlo_filename="graph.hlo",
                neff_filename="graph.neff",
                g_device_id=rank,
                g_device_count=WORLD,
                artifacts=object(),
            )
        except BaseException as exc:  # every rank's own end is part of the reading
            errors[rank] = exc
        if park is not None and rank == park[0]:
            park[1].wait(timeout=JOIN_TIMEOUT)

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        if park is None or thread is not threads[park[0]]:
            thread.join(timeout=JOIN_TIMEOUT)
    return threads, results, errors


def test_the_builds_run_in_waves_of_the_declared_width(monkeypatch, tmp_path):
    build = _FakeBuild(WAVE)
    module = _install(monkeypatch, tmp_path, WAVE, build)
    threads, results, errors = _run(module)
    print(
        f"neff_load_waves|world={WORLD}|wave_size={WAVE}"
        f"|waves={math.ceil(WORLD / WAVE)}|peak_inside={build.peak_inside}|loads={build.calls}"
    )
    assert errors == {}
    assert not [thread for thread in threads if thread.is_alive()]
    assert build.timed_out == 0
    assert build.peak_inside == WAVE
    assert build.calls == WORLD
    assert _waves_in_order(build.events, WORLD, WAVE)
    assert sorted(results) == list(range(WORLD))


def test_a_rank_that_never_returns_after_its_build_does_not_stall_the_waves_behind_it(
    monkeypatch, tmp_path
):
    rows = _Rows()
    build = _FakeBuild(None)
    module = _install(monkeypatch, tmp_path, WAVE, build, rows)
    stuck = 2
    release = threading.Event()
    threads, results, errors = _run(module, park=(stuck, release))
    later = [rank for rank in range(WAVE * 2, WORLD) if rank in results]
    timeouts = [row for row in rows.rows if row.startswith("neff_load_wait_timeout|")]
    print(
        f"neff_load_forward_only|stuck_rank={stuck}|later_waves_finished={len(later)}"
        f"|stuck_alive={threads[stuck].is_alive()}|wait_timeout_rows={len(timeouts)}"
    )
    assert errors == {}
    assert threads[stuck].is_alive()
    assert later == list(range(WAVE * 2, WORLD))
    assert timeouts == []
    release.set()
    threads[stuck].join(timeout=JOIN_TIMEOUT)


def test_every_rank_writes_a_row_on_each_side_of_its_own_build(monkeypatch, tmp_path):
    rows = _Rows()
    build = _FakeBuild(WAVE)
    module = _install(monkeypatch, tmp_path, WAVE, build, rows)
    threads, results, errors = _run(module)
    assert errors == {}
    by_rank: dict[int, list[str]] = {}
    for row in rows.rows:
        rank = re.search(r"rank=(\d+)", row)
        if rank and row.startswith(("neff_load_entered|", "neff_load_left|")):
            by_rank.setdefault(int(rank.group(1)), []).append(row)
    written = [row for rank_rows in by_rank.values() for row in rank_rows]
    flags = sorted(path.name for path in tmp_path.glob("prefill/*/rank_*.done"))
    print(f"neff_load_rows|ranks={len(by_rank)}|rows={len(written)}|flags={len(flags)}")
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
    assert flags == sorted(f"rank_{rank}.done" for rank in range(WORLD))


def test_the_control_with_the_wave_size_at_the_world_size_builds_every_rank_at_once(
    monkeypatch, tmp_path
):
    rows = _Rows()
    build = _FakeBuild(WORLD)
    module = _install(monkeypatch, tmp_path, WORLD, build, rows)
    threads, results, errors = _run(module)
    staged = [
        row
        for row in rows.rows
        if row.startswith(("neff_load_entered|", "neff_load_left|", "neff_load_wait_timeout|"))
    ]
    print(
        f"neff_load_control|world={WORLD}|wave_size={WORLD}|peak_inside={build.peak_inside}"
        f"|waves_in_order={_waves_in_order(build.events, WORLD, WAVE)}|rows={len(staged)}"
    )
    assert errors == {}
    assert build.timed_out == 0
    assert build.peak_inside == WORLD
    assert _waves_in_order(build.events, WORLD, WORLD)
    assert not _waves_in_order(build.events, WORLD, WAVE)
    assert staged == []
    assert list(tmp_path.glob("prefill/*/rank_*.done")) == []


def test_the_wrapper_calls_the_original_builder_once_and_returns_its_object(
    monkeypatch, tmp_path
):
    build = _FakeBuild(WAVE)
    module = _install(monkeypatch, tmp_path, WAVE, build)
    wrapper = module.build_executable
    staged_neff_load.apply_staged_neff_load(module)
    threads, results, errors = _run(module)
    print(
        f"neff_load_wrapper|calls={build.calls}"
        f"|the_second_install_left_the_first={module.build_executable is wrapper}"
    )
    assert errors == {}
    assert wrapper is not build
    assert module.build_executable is wrapper
    assert build.calls == WORLD
    assert set(results.values()) == {build.result}
