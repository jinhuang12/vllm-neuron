# SPDX-License-Identifier: Apache-2.0
"""Acceptance: the registered compiler runs in waves of ranks, signalling forward only.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/worker/test_warmup_staged_neff_load.py \\
      -rA -s -p no:cacheprovider --timeout=5400

Compiling a graph loads it into host memory, so a group that compiles on every rank at one
instant exhausts the host. The stage brackets the compiler the plugin registers with
torch.compile: a wave of ranks runs it, each rank drops one flag file when it returns, and the
wave behind waits for those flags. The waiting points FORWARD only, because a rank that has
returned is already inside the forward's collectives and cannot answer a group-wide wait, and
the wait is bounded -- on timeout it logs a row and proceeds.

The real compiler returns the traced forward unchanged in CPU mode, so it never loads anything
here. A fake compiler stands in for it, and the stage brackets that fake exactly as the plugin
brackets the real one at registration.

Twelve items, ONE test each, no ``parametrize``:

* T01 -- the compilations run in waves of the declared width, in wave order.
* T02 -- a rank that never returns from the call after it does not stall the waves behind it.
* T03 -- every rank writes a row on each side of its own call, and one flag after it.
* T04 -- the control: with the wave size at the world size the gate is off and nothing stages.
* T05 -- the stage calls the compiler once per call and returns its object.
* T06 -- a wave that never signals costs the wave behind one bounded wait, not the run.
* T07 -- the gate is off, with its reason, when the context is incomplete.
* T08 -- a second load in one named window waves as well, on flags of its own.
* T09 -- flags an earlier attempt left behind do not satisfy this run's wait.
* T10 -- the backend name torch.compile resolves carries the stage, around the vendor's compiler.
* T11 -- the control: wave size zero stages nothing and says so.
* T12 -- a wait timeout of zero costs one row per waiting rank and no waiting.

T01, T04, T08 and T11 print the largest number of ranks inside the compiler at once, so the
transcript carries that number and not only a verdict. Run pytest with ``-s``.
"""

import math
import re
import threading

from vllm_neuron.vllm.patches import staged_neff_load

WORLD = 8
WAVE = 2
SMALL_WORLD = 4
CALL_TOGETHER_TIMEOUT = 5.0
WAIT_TIMEOUT = 5
SHORT_WAIT_TIMEOUT = 1
JOIN_TIMEOUT = 30.0
RUN = "run_under_test"
STALE_RUN = "run_that_already_ended"


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


class _FakeDist:
    """``torch.distributed`` as the stage reads it: one rank per thread, one world size."""

    def __init__(self, world_size: int, initialized: bool = True) -> None:
        self.world_size = world_size
        self._initialized = initialized
        self._local = threading.local()

    def take_rank(self, rank: int) -> None:
        self._local.rank = rank

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    def get_rank(self) -> int:
        return self._local.rank

    def get_world_size(self) -> int:
        return self.world_size


class _FakeCompile:
    """The registered compiler, faked: records who is inside it, and how many at once."""

    def __init__(self, group, width: int | None = None, park: dict | None = None) -> None:
        self._group = group
        self._together = threading.Barrier(width) if width else None
        self._park = park or {}
        self._lock = threading.Lock()
        self._inside = 0
        self.peak_inside = 0
        self.calls = 0
        self.timed_out = 0
        self.events: list[tuple[str, int]] = []
        self.result = object()

    def __call__(self, graph_module, example_inputs=None, options=None):
        rank = self._group.get_rank()
        with self._lock:
            self._inside += 1
            self.calls += 1
            self.peak_inside = max(self.peak_inside, self._inside)
            self.events.append(("enter", rank))
        if self._together is not None:
            try:
                self._together.wait(timeout=CALL_TOGETHER_TIMEOUT)
            except threading.BrokenBarrierError:
                with self._lock:
                    self.timed_out += 1
        if rank in self._park:
            self._park[rank].wait(timeout=JOIN_TIMEOUT)
        with self._lock:
            self._inside -= 1
            self.events.append(("leave", rank))
        return self.result


def _waves_in_order(events: list, world_size: int, wave_size: int) -> bool:
    """True when the calls group into consecutive waves of this width that never overlap."""
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


def _install(monkeypatch, tmp_path, wave_size, group, compiler, rows=None,
             timeout=WAIT_TIMEOUT, named=True, signal_dir=True):
    """Bracket this fake compiler with the stage, as the plugin brackets the real one."""
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_WAVE_SIZE", str(wave_size))
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR", str(tmp_path) if signal_dir else "")
    monkeypatch.setenv("VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT", str(timeout))
    monkeypatch.setenv(staged_neff_load.RUN_TOKEN_ENV, RUN)
    monkeypatch.setattr(staged_neff_load, "dist", group)
    monkeypatch.setattr(staged_neff_load, "_LOAD_ORDINAL", {})
    monkeypatch.setattr(staged_neff_load, "_ROOTS_LOGGED", set())
    if rows is not None:
        monkeypatch.setattr(staged_neff_load, "logger", rows)
    staged_neff_load.forget_this_load()
    if named:
        staged_neff_load.name_this_load("prefill", "2048/kv2048")
    return staged_neff_load.staged_compiler(compiler)


def _flags(tmp_path, run=RUN, load="load_*", rank="rank_*"):
    """Return the flag files one run left behind, newest layout: run/phase/bucket/load/rank."""
    return sorted(str(path) for path in tmp_path.glob(f"{run}/*/*/{load}/{rank}.done"))


def _run(staged, group, world_size, park_after=None, still_inside=(), loads=1):
    """Call the staged compiler once per rank on its own thread; return threads and results.

    A rank named in ``still_inside`` is not waited for: it is parked inside the compiler, and the
    reading it exists for is what the ranks behind it did while it was there. ``loads`` calls the
    staged compiler that many times per rank, as one warmup window does for the graphs it builds.
    """
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def body(rank):
        group.take_rank(rank)
        try:
            for _ in range(loads):
                results[rank] = staged(object(), [], options=None)
        except BaseException as exc:  # every rank's own end is part of the reading
            errors[rank] = exc
        if park_after is not None and rank == park_after[0]:
            park_after[1].wait(timeout=JOIN_TIMEOUT)

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for rank, thread in enumerate(threads):
        if rank in still_inside:
            continue
        if park_after is None or rank != park_after[0]:
            thread.join(timeout=JOIN_TIMEOUT)
    return threads, results, errors


def test_the_compilations_run_in_waves_of_the_declared_width(monkeypatch, tmp_path):
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WAVE)
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler)
    threads, results, errors = _run(staged, group, WORLD)
    print(
        f"neff_load_waves|world={WORLD}|wave_size={WAVE}|waves={math.ceil(WORLD / WAVE)}"
        f"|peak_inside={compiler.peak_inside}|loads={compiler.calls}"
    )
    assert errors == {}
    assert not [thread for thread in threads if thread.is_alive()]
    assert compiler.timed_out == 0
    assert compiler.peak_inside == WAVE
    assert compiler.calls == WORLD
    assert _waves_in_order(compiler.events, WORLD, WAVE)
    assert sorted(results) == list(range(WORLD))


def test_a_rank_that_never_returns_after_its_call_does_not_stall_the_waves_behind_it(
    monkeypatch, tmp_path
):
    rows = _Rows()
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group)
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler, rows)
    stuck = 2
    release = threading.Event()
    threads, results, errors = _run(staged, group, WORLD, park_after=(stuck, release))
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


def test_every_rank_writes_a_row_on_each_side_of_its_own_call(monkeypatch, tmp_path):
    rows = _Rows()
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WAVE)
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler, rows)
    threads, results, errors = _run(staged, group, WORLD)
    assert errors == {}
    by_rank: dict[int, list[str]] = {}
    for row in rows.rows:
        rank = re.search(r"rank=(\d+)", row)
        if rank and row.startswith(("neff_load_entered|", "neff_load_left|")):
            by_rank.setdefault(int(rank.group(1)), []).append(row)
    written = [row for rank_rows in by_rank.values() for row in rank_rows]
    flags = sorted(path.rsplit("/", 1)[1] for path in _flags(tmp_path))
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


def test_the_control_with_the_wave_size_at_the_world_size_stages_nothing(
    monkeypatch, tmp_path
):
    rows = _Rows()
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WORLD)
    staged = _install(monkeypatch, tmp_path, WORLD, group, compiler, rows)
    threads, results, errors = _run(staged, group, WORLD)
    staged_rows = [
        row
        for row in rows.rows
        if row.startswith(("neff_load_entered|", "neff_load_left|", "neff_load_wait_timeout|"))
    ]
    gate_off = [row for row in rows.rows if row == "neff_load_gate|off|reason=wave_size_covers_the_world"]
    print(
        f"neff_load_control|world={WORLD}|wave_size={WORLD}|peak_inside={compiler.peak_inside}"
        f"|waves_in_order={_waves_in_order(compiler.events, WORLD, WAVE)}"
        f"|staged_rows={len(staged_rows)}|gate_off_rows={len(gate_off)}"
    )
    assert errors == {}
    assert compiler.timed_out == 0
    assert compiler.peak_inside == WORLD
    assert _waves_in_order(compiler.events, WORLD, WORLD)
    assert not _waves_in_order(compiler.events, WORLD, WAVE)
    assert staged_rows == []
    assert len(gate_off) == WORLD
    assert _flags(tmp_path) == []


def test_the_stage_calls_the_compiler_once_and_returns_its_object(monkeypatch, tmp_path):
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WAVE)
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler)
    threads, results, errors = _run(staged, group, WORLD)
    print(
        f"neff_load_wrapper|calls={compiler.calls}"
        f"|results_are_the_compilers={set(results.values()) == {compiler.result}}"
        f"|the_stage_is_the_compiler={staged is compiler}"
    )
    assert errors == {}
    assert staged is not compiler
    assert compiler.calls == WORLD
    assert set(results.values()) == {compiler.result}


def test_a_wave_that_never_signals_costs_the_next_wave_one_bounded_wait(monkeypatch, tmp_path):
    rows = _Rows()
    group = _FakeDist(SMALL_WORLD)
    never_returns = threading.Event()
    compiler = _FakeCompile(group, park={1: never_returns})
    staged = _install(
        monkeypatch, tmp_path, WAVE, group, compiler, rows, timeout=SHORT_WAIT_TIMEOUT
    )
    threads, results, errors = _run(staged, group, SMALL_WORLD, still_inside=(1,))
    timeouts = [row for row in rows.rows if row.startswith("neff_load_wait_timeout|")]
    waiting = [rank for rank in range(WAVE, SMALL_WORLD) if rank in results]
    print(
        f"neff_load_timeout|waiting_ranks={len(waiting)}|timeout_rows={len(timeouts)}"
        f"|finished={len(waiting)}"
    )
    assert errors == {}
    assert waiting == list(range(WAVE, SMALL_WORLD))
    assert len(timeouts) == SMALL_WORLD - WAVE
    assert all("|missing=1" in row for row in timeouts)
    assert threads[1].is_alive()  # still inside the compiler, so its flag was never written
    assert _flags(tmp_path, rank="rank_1") == []
    never_returns.set()
    threads[1].join(timeout=JOIN_TIMEOUT)


def test_the_gate_is_off_with_its_reason_when_the_context_is_incomplete(monkeypatch, tmp_path):
    rows = _Rows()
    unnamed_group = _FakeDist(WORLD)
    unnamed = _FakeCompile(unnamed_group)
    staged = _install(monkeypatch, tmp_path, WAVE, unnamed_group, unnamed, rows, named=False)
    _run(staged, unnamed_group, WAVE)
    no_group = _FakeDist(WORLD, initialized=False)
    alone = _FakeCompile(no_group)
    staged = _install(monkeypatch, tmp_path, WAVE, no_group, alone, rows)
    _run(staged, no_group, WAVE)
    reasons = sorted(
        {row.split("reason=")[1] for row in rows.rows if row.startswith("neff_load_gate|off|")}
    )
    staged_rows = [row for row in rows.rows if row.startswith(("neff_load_entered|", "neff_load_left|"))]
    print(
        f"neff_load_gate_off|reasons={','.join(reasons)}|staged_rows={len(staged_rows)}"
        f"|calls={unnamed.calls + alone.calls}"
    )
    assert reasons == ["no_named_load", "no_process_group"]
    assert staged_rows == []
    assert unnamed.calls == WAVE
    assert alone.calls == WAVE
    assert _flags(tmp_path) == []
def test_a_second_load_in_the_same_window_waves_as_well(monkeypatch, tmp_path):
    rows = _Rows()
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WAVE)
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler, rows)
    threads, results, errors = _run(staged, group, WORLD, loads=2)
    first = _flags(tmp_path, load="load_0")
    second = _flags(tmp_path, load="load_1")
    roots = [row for row in rows.rows if row.startswith("neff_load_root|")]
    loads = sorted({row.split("|load=")[1] for row in rows.rows if "|load=" in row})
    print(
        f"neff_load_two_loads|calls={compiler.calls}|peak_inside={compiler.peak_inside}"
        f"|first_load_flags={len(first)}|second_load_flags={len(second)}"
        f"|loads_in_rows={','.join(loads)}|root_rows={len(roots)}"
    )
    assert errors == {}
    assert not [thread for thread in threads if thread.is_alive()]
    assert compiler.timed_out == 0
    assert compiler.calls == WORLD * 2
    assert compiler.peak_inside == WAVE
    assert _waves_in_order(compiler.events[: 2 * WORLD], WORLD, WAVE)
    assert _waves_in_order(compiler.events[2 * WORLD :], WORLD, WAVE)
    assert len(first) == WORLD
    assert len(second) == WORLD
    assert loads == ["0", "1"]
    assert len(roots) == 1


def test_flags_from_an_earlier_attempt_do_not_satisfy_this_runs_wait(monkeypatch, tmp_path):
    rows = _Rows()
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WAVE)
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler, rows)
    ended = tmp_path / STALE_RUN / "prefill" / "2048_kv2048" / "load_0"
    ended.mkdir(parents=True)
    for rank in range(WORLD):
        (ended / f"rank_{rank}.done").write_text("an attempt that already ended\n")
    monkeypatch.delenv(staged_neff_load.RUN_TOKEN_ENV)
    token = staged_neff_load.pin_this_run()
    named_again = staged_neff_load.pin_this_run()
    threads, results, errors = _run(staged, group, WORLD)
    mine = _flags(tmp_path, run=token)
    left_behind = sorted(path.name for path in ended.glob("rank_*.done"))
    timeouts = [row for row in rows.rows if row.startswith("neff_load_wait_timeout|")]
    print(
        f"neff_load_stale_flags|flags_left_behind={len(left_behind)}"
        f"|peak_inside={compiler.peak_inside}|my_flags={len(mine)}"
        f"|my_run_is_its_own={token != STALE_RUN}|named_once={named_again == token}"
        f"|timeout_rows={len(timeouts)}"
    )
    assert errors == {}
    assert token.startswith("run_")
    assert named_again == token
    assert compiler.peak_inside == WAVE
    assert compiler.timed_out == 0
    assert len(mine) == WORLD
    assert timeouts == []
    assert left_behind == sorted(f"rank_{rank}.done" for rank in range(WORLD))


def test_the_backend_name_torch_compile_resolves_carries_the_stage(monkeypatch):
    import torch._dynamo.backends.registry as registry

    import vllm_neuron  # noqa: F401  importing the plugin is what installs the bracket

    entry = registry.lookup_backend("neuron_libtorch")
    wrapped = getattr(entry, "__wrapped__", None)
    rows = _Rows()
    monkeypatch.setattr(staged_neff_load, "logger", rows)
    monkeypatch.setattr(staged_neff_load, "dist", _FakeDist(WORLD, initialized=False))
    try:
        entry(object(), [], options=None)
    except BaseException:  # the vendor's compiler refuses a bare object; the row is the reading
        pass
    gate = [row for row in rows.rows if row == "neff_load_gate|off|reason=no_process_group"]
    print(
        f"neff_load_registry|name={getattr(entry, '__name__', 'none')}"
        f"|staged={getattr(entry, 'staged_neff_load', False)}"
        f"|wraps={getattr(wrapped, '__module__', 'none')}.{getattr(wrapped, '__name__', 'none')}"
        f"|wrapped_is_a_stage={getattr(wrapped, 'staged_neff_load', False)}"
        f"|reached_the_stage={len(gate)}"
    )
    assert getattr(entry, "staged_neff_load", False)
    assert not getattr(wrapped, "staged_neff_load", False)
    assert getattr(wrapped, "__module__", "").startswith("libtorch_neuronx_lite")
    assert len(gate) == 1


def test_the_control_with_the_wave_size_at_zero_stages_nothing(monkeypatch, tmp_path):
    rows = _Rows()
    group = _FakeDist(WORLD)
    compiler = _FakeCompile(group, width=WORLD)
    staged = _install(monkeypatch, tmp_path, 0, group, compiler, rows)
    threads, results, errors = _run(staged, group, WORLD)
    gate_off = [row for row in rows.rows if row == "neff_load_gate|off|reason=wave_size_zero"]
    staged_rows = [
        row for row in rows.rows if row.startswith(("neff_load_entered|", "neff_load_left|"))
    ]
    print(
        f"neff_load_off_switch|wave_size=0"
        f"|knob={staged_neff_load.envs.VLLM_NEURON_NEFF_LOAD_WAVE_SIZE}"
        f"|peak_inside={compiler.peak_inside}|gate_off_rows={len(gate_off)}"
        f"|staged_rows={len(staged_rows)}"
    )
    assert errors == {}
    assert staged_neff_load.envs.VLLM_NEURON_NEFF_LOAD_WAVE_SIZE == 0
    assert compiler.timed_out == 0
    assert compiler.peak_inside == WORLD
    assert len(gate_off) == WORLD
    assert staged_rows == []
    assert _flags(tmp_path) == []


def test_a_wait_timeout_of_zero_costs_one_row_and_no_waiting(monkeypatch, tmp_path):
    rows = _Rows()
    group = _FakeDist(SMALL_WORLD)
    never_returns = threading.Event()
    compiler = _FakeCompile(group, park={1: never_returns})
    staged = _install(monkeypatch, tmp_path, WAVE, group, compiler, rows, timeout=0)
    threads, results, errors = _run(staged, group, SMALL_WORLD, still_inside=(1,))
    timeouts = [row for row in rows.rows if row.startswith("neff_load_wait_timeout|")]
    waits = [float(row.split("|waited_s=")[1].split("|")[0]) for row in timeouts]
    behind = [rank for rank in range(WAVE, SMALL_WORLD) if rank in results]
    print(
        f"neff_load_timeout_zero"
        f"|knob={staged_neff_load.envs.VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT}"
        f"|timeout_rows={len(timeouts)}|longest_wait_s={max(waits) if waits else 'none'}"
        f"|finished={len(behind)}"
    )
    assert errors == {}
    assert staged_neff_load.envs.VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT == 0
    assert len(timeouts) == SMALL_WORLD - WAVE
    assert max(waits) < 1.0
    assert behind == list(range(WAVE, SMALL_WORLD))
    assert threads[1].is_alive()
    never_returns.set()
    threads[1].join(timeout=JOIN_TIMEOUT)
