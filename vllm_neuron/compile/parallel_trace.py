# SPDX-License-Identifier: Apache-2.0
"""Fork-based pool for parallelizing graph trace.

Public API
----------

::

    parallel_trace.parallel_trace(jobs, parent_rank=0) -> None

A *job* is a ``(callable, kwargs)`` pair: a forked child invokes
``callable(**kwargs)``. The "callable" is typically a torch.compile
wrapper (``capture_backend_model``) so the call drives an FX→HLO trace
into the cache.

Why fork
--------

Dynamo's frame-eval / guard / code cache and torch_xla's IR tape are
process-global, so trace can't be parallelized within a single process.
``os.fork()`` from the already-fully-initialized parent worker is the
cheapest way to get N independent Dynamo states: the child inherits the
parent's distributed state (rank lists, group registry, vllm_config,
loaded model), so capture-side code looks up rank metadata and module
weights for free. The child only does FX→HLO lowering on synthetic
meta-device inputs — no real gloo collectives are executed, so the
inherited gloo socket is never touched.

Meta swap
---------

After fork, the parent's NRT runtime enters ``NRT_STATE_CHILD`` and
refuses any allocation or deallocation, so the inherited model — whose
parameters live on the neuron device — is unusable. Each child swaps
the *unique set* of underlying nn.Modules across its assigned jobs to
the meta device before running anything. KV caches that are different
views of one underlying buffer (e.g. ``typed_tensor[0]`` /
``typed_tensor[1]`` from ``initialize_kv_cache``) are remapped to a
single shared meta storage, so the capture backend's input-dedup pass
still collapses them to one FX placeholder rather than emitting two
independent ones.

Disable
-------

Set ``VLLM_NEURON_DISABLE_PARALLEL_TRACE=1`` to skip the fork pool and
run jobs sequentially in the parent process. Setting
``VLLM_NEURON_PARALLEL_TRACE_WORKERS=1`` is also honored — a
single forked child runs all jobs, useful for matching the fork code
path while disabling parallelism.

Host-RAM admission gate
-----------------------

Every rank calls ``parallel_trace`` independently and nothing in this
file or its callers knows about the other ranks' children, so
``VLLM_NEURON_PARALLEL_TRACE_WORKERS`` bounds *one* rank's fan-out while
host RAM is consumed by the **sum over ranks**. On a large-TP host the
sum can exceed physical RAM even though every individual rank is
configured conservatively, and the failure mode is an OOM kill of an
arbitrary child.

Set ``VLLM_NEURON_MAX_CONCURRENT_TRACERS`` to a positive value to cap the
number of trace children alive at once *across every rank on the host*,
and ``VLLM_NEURON_TRACER_ADMIT_TIMEOUT`` to bound how long a rank waits
for a slot. **Both default to 0, which is OFF**: at the default the gate
object is never constructed, no lock path is created or touched, no log
line is emitted and the fork ordering is unchanged, so the pool behaves
exactly as it does with no gate in the code at all. See
``_TracerAdmissionGate``.
"""

import fcntl
import logging
import os
import shutil
import signal
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch

from vllm_neuron import envs

logger = logging.getLogger(__name__)


# A trace job: a callable plus the kwargs that drive its forward pass.
Job = tuple[Callable[..., Any], dict[str, Any]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parallel_trace(jobs: list[Job], parent_rank: int = 0) -> None:
    """Run each ``(callable, kwargs)`` job in a forked child.

    Args:
        jobs: list of ``(callable, kwargs)``. Inside each child the call
            is ``callable(**kwargs)``. All tensors in ``kwargs`` must be
            on the meta device — they're traced, not executed.
        parent_rank: engine-local rank, used to namespace status files
            and tag log records.

    The pool size is ``VLLM_NEURON_PARALLEL_TRACE_WORKERS``
    (default ``8``), capped at ``len(jobs)``; setting it to 1 runs all
    jobs in a single forked child. Set
    ``VLLM_NEURON_DISABLE_PARALLEL_TRACE=1`` to bypass the pool entirely
    and run jobs in the parent process.

    That pool size is **per rank**. To bound the host-wide total, set
    ``VLLM_NEURON_MAX_CONCURRENT_TRACERS`` (plus
    ``VLLM_NEURON_TRACER_ADMIT_TIMEOUT``); both default to ``0`` = off, and
    at the default this function behaves exactly as it did before the gate
    existed.

    Raises:
        ValueError: if any job's kwargs include a non-meta tensor, or if
            ``VLLM_NEURON_MAX_CONCURRENT_TRACERS`` is set without
            ``VLLM_NEURON_TRACER_ADMIT_TIMEOUT``.
        TracerAdmissionTimeout: if the admission gate is enabled and no
            slot came free within the configured wait.
        RuntimeError: if any forked child fails. Other children are
            still waited on first so we don't leak processes.
    """
    if not jobs:
        return

    _validate_jobs_on_meta(jobs)

    num_workers = min(envs.VLLM_NEURON_PARALLEL_TRACE_WORKERS, len(jobs))
    logger.info(
        "Parallel trace: jobs=%d, lanes=%d, parent_rank=%d",
        len(jobs),
        num_workers,
        parent_rank,
    )
    t0 = time.perf_counter()
    _run_pool_fork(jobs, parent_rank, num_workers)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Parallel trace finished: jobs=%d, lanes=%d, %.2fs",
        len(jobs),
        num_workers,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Pool driver
# ---------------------------------------------------------------------------


def _run_pool_fork(jobs: list[Job], parent_rank: int, num_workers: int) -> None:
    """Fork one child per non-empty lane. Each child runs all jobs
    assigned to its lane (in order) and exits. Parent ``waitpid``s and
    reads each child's status file to detect failures.

    Forking once per lane (rather than once per job) amortizes the
    meta-swap cost across the lane's jobs.

    When ``VLLM_NEURON_MAX_CONCURRENT_TRACERS`` is set, each fork is gated on
    a host-wide admission slot taken immediately before it. With the key unset
    the gate is ``None`` and every gate call site below is skipped.
    """
    if not jobs:
        return

    lanes = _partition_round_robin(jobs, num_workers)
    # None unless VLLM_NEURON_MAX_CONCURRENT_TRACERS is positive. At the
    # default this constructs nothing, touches no path and logs nothing.
    # Built BEFORE the workdir so a refused configuration fails without
    # leaving a temp directory behind: nothing between mkdtemp and the try
    # below may raise.
    gate = _TracerAdmissionGate.from_env(parent_rank)
    workdir = tempfile.mkdtemp(prefix=f"trace_pool_rank{parent_rank}_")
    try:
        child_pids: dict[int, int] = {}
        result_paths: dict[int, str] = {}
        completed: dict[int, tuple[int, int]] = {}  # lane_idx -> (pid, exit_code)
        first_failure: str | None = None
        # Lanes this pool signalled itself, so that our own teardown SIGKILL
        # is never reported as a probable OOM kill.
        aborted_lanes: set[int] = set()

        def _reap(lane_idx: int, pid: int, status_word: int) -> None:
            nonlocal first_failure
            pool_initiated = lane_idx in aborted_lanes
            exit_code, death = _describe_child_death(status_word, pool_initiated)
            completed[lane_idx] = (pid, exit_code)
            if gate is not None:
                gate.note_release(lane_idx)
            child_status, child_err = _read_status_file(result_paths[lane_idx])
            if exit_code != 0 or child_status != "OK":
                _warn_on_fatal_signal(lane_idx, pid, status_word, pool_initiated)
                msg = (
                    f"lane={lane_idx} pid={pid} exit_code={exit_code} "
                    f"death={death} "
                    f"status={child_status} err={child_err}"
                )
                if first_failure is None:
                    first_failure = msg
                    logger.error(
                        "Parallel trace lane failed; aborting siblings: %s", msg
                    )

        try:
            for lane_idx, lane_jobs in enumerate(lanes):
                if not lane_jobs:
                    continue
                rp = os.path.join(workdir, f"lane{lane_idx}.status")
                result_paths[lane_idx] = rp
                # Admission goes here, immediately before the fork, because
                # the fork is what spends the memory being rationed.
                slot_fd = gate.acquire(lane_idx) if gate is not None else None
                pid = os.fork()
                if pid == 0:
                    # Child: run target and exit. Use os._exit to skip
                    # atexit handlers (which would otherwise try to clean
                    # up parent state we still want). Any inherited
                    # admission descriptor is deliberately left open and
                    # untouched: the kernel closes it at exit however the
                    # child dies, and that close IS the slot release.
                    try:
                        _fork_child_main(lane_idx, parent_rank, lane_jobs, rp)
                        os._exit(0)
                    except BaseException:
                        try:
                            with open(rp, "w") as f:
                                f.write("ERROR\n" + traceback.format_exc())
                        except Exception:
                            pass
                        os._exit(1)
                else:
                    child_pids[lane_idx] = pid
                    if gate is not None:
                        gate.hand_to_child(slot_fd, lane_idx, pid)
        except BaseException:
            # A refused admission or a failed fork must not strand the lanes
            # already started: unreaped children would become zombies and
            # would hold their slots until this process itself died.
            if child_pids:
                logger.error(
                    "Parallel trace aborting %d already-forked lane(s) after a "
                    "fan-out failure (rank=%d)",
                    len(child_pids),
                    parent_rank,
                )
                _abort_remaining(
                    dict(child_pids), completed, _reap, aborted_lanes
                )
            raise

        # Poll our own lane PIDs so we can early-abort surviving lanes
        # the moment one fails.
        pending = dict(child_pids)  # lane_idx -> pid

        while pending:
            for lane_idx in list(pending):
                pid = pending[lane_idx]
                try:
                    result_pid, status_word = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    # Already reaped (shouldn't happen in normal flow,
                    # but treat as exit_code=-1 so we still surface it).
                    pending.pop(lane_idx, None)
                    completed[lane_idx] = (pid, -1)
                    if gate is not None:
                        gate.note_release(lane_idx)
                    continue
                if result_pid == 0:
                    continue  # still running
                pending.pop(lane_idx, None)
                _reap(lane_idx, pid, status_word)
            if first_failure is not None and pending:
                # Early abort: SIGTERM remaining children, give them a
                # short grace period to flush their status files, then
                # SIGKILL stragglers so the workdir cleanup can run.
                _abort_remaining(pending, completed, _reap, aborted_lanes)
                break
            if pending:
                time.sleep(0.1)

        if first_failure is not None:
            raise RuntimeError(
                f"Parallel trace fork failed (rank={parent_rank}): {first_failure}"
            )
    finally:
        if gate is not None:
            gate.close()
        shutil.rmtree(workdir, ignore_errors=True)


_ABORT_GRACE_PERIOD_S = 5.0
"""Time we give SIGTERM'd children to flush their status files before
escalating to SIGKILL. Tracing children spend most of their time inside
torch_xla / Dynamo C extensions that don't always honour SIGTERM
promptly, so escalation is needed to make progress on a real failure;
the grace period is long enough that a child mid-write has time to
finish."""


def _abort_remaining(
    pending: dict[int, int],
    completed: dict[int, tuple[int, int]],
    reap: Callable[[int, int, int], None],
    aborted_lanes: set[int],
) -> None:
    """Kill the still-running lane children after another lane failed.

    Sends SIGTERM, polls briefly so cooperating children can flush
    their status files, then SIGKILLs stragglers. Reaps every PID
    via the supplied ``reap`` callback so the parent doesn't leave
    zombies behind. Mutates ``pending`` in place.

    Every lane it signals is recorded in ``aborted_lanes`` **before** the
    signal is sent, so the reaper can tell our own teardown apart from an
    external kill. Recording after signalling would leave a window in which
    a child we just SIGKILLed is reaped and mis-reported as an OOM victim.

    Admission slots need no explicit release here: each is owned by the
    child's inherited descriptor and the kernel drops it when the child
    dies, which is exactly what SIGTERM and SIGKILL cause.
    """
    for lane_idx, pid in list(pending.items()):
        aborted_lanes.add(lane_idx)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _ABORT_GRACE_PERIOD_S
    while pending and time.monotonic() < deadline:
        for lane_idx in list(pending):
            pid = pending[lane_idx]
            try:
                result_pid, status_word = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pending.pop(lane_idx, None)
                completed[lane_idx] = (pid, -1)
                continue
            if result_pid == 0:
                continue
            pending.pop(lane_idx, None)
            reap(lane_idx, pid, status_word)
        if pending:
            time.sleep(0.05)

    # Stragglers — SIGKILL and reap synchronously. Blocking waitpid is
    # safe here: SIGKILL guarantees prompt exit, and we own the PID.
    for lane_idx, pid in list(pending.items()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            _, status_word = os.waitpid(pid, 0)
        except ChildProcessError:
            pending.pop(lane_idx, None)
            completed[lane_idx] = (pid, -1)
            continue
        pending.pop(lane_idx, None)
        reap(lane_idx, pid, status_word)


def _partition_round_robin(items: list, n: int) -> list[list]:
    """Return n lanes; items round-robin so each lane runs ceil(len/n) jobs."""
    lanes: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        lanes[i % n].append(item)
    return lanes


# ---------------------------------------------------------------------------
# Cross-process admission gate
# ---------------------------------------------------------------------------

_ADMIT_POLL_INTERVAL_S = 0.25
"""How often a waiting rank re-scans the slot table. Each scan is at most
``cap`` non-blocking ``flock`` calls on already-created files, so this is
cheap; it is short relative to a trace so a freed slot is picked up
promptly."""

_ADMIT_HEARTBEAT_S = 30.0
"""How often a waiting rank logs that it is *waiting* rather than hung.
Without this line a staged trace and a deadlocked one look identical in the
log, which is the whole hazard the bounded wait exists to remove."""


class TracerAdmissionTimeout(RuntimeError):
    """A rank could not obtain a tracer slot before its deadline.

    Raised instead of waiting forever. An unbounded wait is not a safe
    default here: it converts a loud, well-attributed host-RAM OOM into a
    silent hang, and neither of the two timeouts that look like they would
    catch it actually does. ``VLLM_NEURON_BARRIER_TIMEOUT`` bounds the
    *post*-trace rendezvous, which is downstream of ``parallel_trace``
    returning, and a ``MemAvailable`` watchdog sees a **healthy** host during
    a staging stall precisely because everybody is waiting instead of
    tracing.
    """


class _TracerAdmissionGate:
    """Caps concurrently live trace children across all ranks on a host.

    Mechanism: a directory of ``cap`` zero-length lock files, one per slot,
    each guarded by an exclusive non-blocking ``flock``. A rank acquires a
    slot immediately before it forks a lane child, and the child inherits the
    descriptor. Admission is therefore taken *before* the memory is spent,
    which is the only ordering that bounds peak host demand.

    Why ``flock`` and not a semaphore — this is requirement R-41(a).
    A POSIX named semaphore, a ``multiprocessing`` semaphore, or any
    counter the holder must decrement itself is **not** crash-safe: a
    ``SIGKILL``\\ ed holder runs no handler, posts nothing, and its slot is
    lost forever. Losing slots is strictly worse than the OOM this gate
    prevents, because the pool then *deadlocks* instead of failing loudly.
    ``flock`` locks are owned by the **open file description**, so the kernel
    drops them when the last descriptor closes — including on ``SIGKILL``,
    ``os._exit``, an OOM kill, or a segfault. No handler, no cleanup pass and
    no stale-lock reclamation is involved, so there is no code path that can
    fail to release.

    Who holds a slot, and why the parent lets go. The parent acquires, forks,
    and then **closes its own descriptor** (``hand_to_child``), leaving the
    child as sole owner. Two things follow, and both matter:

    * A child's slot is released by the kernel the instant the child dies,
      whether or not the parent has reaped it yet.
    * A parent blocked in ``acquire`` for a later lane does not pin the slots
      of its earlier lanes. If the parent held them, a rank that got 3 of the
      4 slots it needs would sit in ``acquire`` — never reaching its reap
      loop, never releasing — and with every rank in that state the pool
      would deadlock at any cap below the total demand. Handing ownership to
      the child removes that failure mode by construction rather than by
      adding opportunistic reaping to the wait loop.

    Releasing uses ``os.close`` and **never** ``flock(LOCK_UN)``: an explicit
    unlock on *any* descriptor sharing the open file description releases the
    lock for the forked child too, which would readmit a tracer while the
    killed-or-still-running one still holds its RAM. Closing one duplicate
    only drops that one reference.

    Slot files are safe to reuse across runs. A stale file carries no stale
    lock, because the lock died with the process that held it.

    Known limitation, stated rather than assumed: ``flock`` is reliable on
    local filesystems and is **not** dependable over NFS, so the slot
    directory is rooted on the same local storage the compile cache uses.
    The resolved directory is logged on every admission so a configuration
    that accidentally partitions ranks into two gates is visible in the log
    rather than silently doubling the cap.
    """

    def __init__(self, cap: int, timeout_s: int, parent_rank: int) -> None:
        self._cap = cap
        self._timeout_s = timeout_s
        self._parent_rank = parent_rank
        self._dir = _admit_slots_dir(cap)
        os.makedirs(self._dir, exist_ok=True)
        self._held: dict[int, int] = {}  # fd -> slot index
        self._child_slots: dict[int, int] = {}  # lane_idx -> slot index
        self._open_failed: set[int] = set()  # slots already warned about

    @classmethod
    def from_env(cls, parent_rank: int) -> "_TracerAdmissionGate | None":
        """Build a gate, or return ``None`` when the feature is off.

        ``None`` is the default and it is a *total* no-op: this function
        returns before reading the timeout key, before touching the
        filesystem and before logging anything, so with the keys unset
        nothing in this class is reachable.
        """
        cap = envs.VLLM_NEURON_MAX_CONCURRENT_TRACERS
        if cap <= 0:
            return None
        timeout_s = envs.VLLM_NEURON_TRACER_ADMIT_TIMEOUT
        if timeout_s <= 0:
            raise ValueError(
                "VLLM_NEURON_MAX_CONCURRENT_TRACERS is set to "
                f"{cap} but VLLM_NEURON_TRACER_ADMIT_TIMEOUT is "
                f"{timeout_s}. The admission gate refuses to run with an "
                "unbounded wait: a lost slot would hang the trace silently "
                "instead of failing, and no other timeout in this stack "
                "covers it (VLLM_NEURON_BARRIER_TIMEOUT applies only after "
                "trace completes). Set both keys, or neither."
            )
        return cls(cap, timeout_s, parent_rank)

    def acquire(self, lane_idx: int) -> int:
        """Block until a slot is free; return the descriptor that owns it.

        Raises:
            TracerAdmissionTimeout: if no slot came free within
                ``VLLM_NEURON_TRACER_ADMIT_TIMEOUT`` seconds. The message
                carries the configured numbers and a census of the slot
                table so the stall is attributable without a second run.
        """
        t0 = time.monotonic()
        deadline = t0 + self._timeout_s
        next_heartbeat = t0 + _ADMIT_HEARTBEAT_S
        sweeps = 0
        # Stagger where each (rank, lane) starts scanning so N ranks don't
        # all contend on slot 0 and serialize their syscalls behind it.
        start = (self._parent_rank + lane_idx) % self._cap
        while True:
            for offset in range(self._cap):
                slot = (start + offset) % self._cap
                fd = self._try_slot(slot, lane_idx)
                if fd is not None:
                    logger.info(
                        "Tracer admission granted: rank=%d lane=%d "
                        "slot=%d/%d waited=%.1fs sweeps=%d dir=%s",
                        self._parent_rank,
                        lane_idx,
                        slot,
                        self._cap,
                        time.monotonic() - t0,
                        sweeps,
                        self._dir,
                    )
                    return fd
            sweeps += 1
            now = time.monotonic()
            if now >= deadline:
                census = "\n".join(self.census())
                raise TracerAdmissionTimeout(
                    f"Tracer admission timed out after "
                    f"{now - t0:.1f}s (VLLM_NEURON_TRACER_ADMIT_TIMEOUT="
                    f"{self._timeout_s}) waiting for 1 of "
                    f"{self._cap} slots (VLLM_NEURON_MAX_CONCURRENT_TRACERS="
                    f"{self._cap}); rank={self._parent_rank} "
                    f"lane={lane_idx} sweeps={sweeps} dir={self._dir}\n"
                    f"slot census (best-effort, sampled after the "
                    f"deadline):\n{census}"
                )
            if now >= next_heartbeat:
                # R-41(c): make "waiting" distinguishable from "hung".
                logger.info(
                    "Tracer admission WAITING (this is staging, not a hang): "
                    "rank=%d lane=%d waited=%.0fs of %ds cap=%d sweeps=%d "
                    "dir=%s",
                    self._parent_rank,
                    lane_idx,
                    now - t0,
                    self._timeout_s,
                    self._cap,
                    sweeps,
                    self._dir,
                )
                next_heartbeat = now + _ADMIT_HEARTBEAT_S
            time.sleep(_ADMIT_POLL_INTERVAL_S)

    def _try_slot(self, slot: int, lane_idx: int) -> int | None:
        """Try to take ``slot``. Return its fd, or None if already held."""
        path = os.path.join(self._dir, f"slot{slot}.lock")
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            # Warn once per slot, not once per sweep: the wait loop re-scans
            # every _ADMIT_POLL_INTERVAL_S, so an unopenable slot would
            # otherwise emit hundreds of identical lines per second and bury
            # the heartbeat that makes waiting legible. The timeout's census
            # reports the condition again, once, as `open-failed`.
            if slot not in self._open_failed:
                self._open_failed.add(slot)
                logger.warning(
                    "Tracer admission cannot open slot file %s: %s. This slot "
                    "is unusable, so the effective cap is below the configured "
                    "%d.",
                    path,
                    e,
                    self._cap,
                )
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        self._held[fd] = slot
        # Diagnostic breadcrumb only — the flock is the authority. Written
        # with pwrite so it does not disturb the file offset the forked child
        # shares with us.
        self._stamp(fd, lane_idx, os.getpid(), slot, "pre-fork")
        return fd

    def _stamp(self, fd: int, lane_idx: int, pid: int, slot: int, what: str) -> None:
        """Overwrite the slot file's diagnostic breadcrumb.

        Truncates first: the post-fork record is shorter than the pre-fork one,
        and ``pwrite`` alone would leave the tail of the longer record behind,
        so the census would report a spliced line. ``pwrite`` rather than
        ``write`` because the forked child shares this descriptor's file
        offset. A census that samples between the truncate and the write sees a
        short record, which is why the census is documented as best-effort.
        """
        rec = (
            f"slot={slot} rank={self._parent_rank} lane={lane_idx} "
            f"{what}_pid={pid} t={time.strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, rec.encode(), 0)
        except OSError:
            pass

    def hand_to_child(self, fd: int, lane_idx: int, child_pid: int) -> None:
        """Make the forked child the sole owner of ``fd``'s slot.

        Records the child pid for the census, then closes the parent's
        duplicate. After this the slot is released exactly when the child
        dies, by the kernel, on every death path including ``SIGKILL`` and
        an OOM kill — see the class docstring for why the parent must not
        keep holding it.
        """
        slot = self._held.pop(fd, -1)
        self._stamp(fd, lane_idx, child_pid, slot, "child")
        try:
            os.close(fd)
        except OSError:
            pass
        self._child_slots[lane_idx] = slot

    def note_release(self, lane_idx: int) -> None:
        """Log that a reaped child's slot is gone. Bookkeeping only.

        The release itself already happened in the kernel when the child
        died; there is deliberately nothing to undo here, which is what
        makes the release path impossible to skip.
        """
        slot = self._child_slots.pop(lane_idx, None)
        if slot is not None:
            logger.info(
                "Tracer admission slot released by child exit: rank=%d "
                "lane=%d slot=%d/%d",
                self._parent_rank,
                lane_idx,
                slot,
                self._cap,
            )

    def close(self) -> None:
        """Drop any descriptor this parent still owns (e.g. a failed fork).

        ``os.close`` and never ``flock(LOCK_UN)`` — see the class docstring.
        """
        for fd in list(self._held):
            self._held.pop(fd, None)
            try:
                os.close(fd)
            except OSError:
                pass

    def census(self) -> list[str]:
        """Best-effort snapshot of the slot table, for the abort message.

        Probing whether a slot is free means trying to lock it, so a free
        slot is briefly locked by this call. That is why the census runs only
        on the already-failing timeout path and never in the heartbeat: on a
        healthy poll it would race real acquirers for no benefit. Held slots
        are never disturbed — the probe simply fails to lock them.
        """
        rows: list[str] = []
        for slot in range(self._cap):
            path = os.path.join(self._dir, f"slot{slot}.lock")
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            except OSError as e:
                rows.append(f"  slot{slot}: open-failed ({e})")
                continue
            try:
                note = os.pread(fd, 256, 0).decode("utf-8", "replace").strip()
            except OSError:
                note = ""
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = "FREE"
            except OSError:
                state = "HELD"
            finally:
                os.close(fd)
            rows.append(
                f"  slot{slot}: {state} last_writer=[{note or '(never taken)'}]"
            )
        return rows


def _admit_slots_dir(cap: int) -> str:
    """Directory holding the ``cap`` slot lock files.

    Every rank on the host must resolve the same path or the gate silently
    partitions into independent gates, so this is derived rather than
    configured — no third env key. ``VLLM_CACHE_ROOT`` is the natural root:
    it is already run-scoped and already shared by every rank of one engine,
    which also keeps concurrent runs and other tenants in separate gates.
    The uid and the cap are in the name so a different user or a re-run at a
    different cap cannot inherit a slot table of the wrong size.
    """
    root = os.environ.get("VLLM_CACHE_ROOT") or tempfile.gettempdir()
    return os.path.join(root, f"vllm_neuron_tracer_admit_uid{os.getuid()}_c{cap}")


# ---------------------------------------------------------------------------
# Child death decoding
# ---------------------------------------------------------------------------


def _describe_child_death(status_word: int, pool_initiated: bool) -> tuple[int, str]:
    """Return ``(exit_code, description)`` for a ``waitpid`` status word.

    ``exit_code`` keeps its historical meaning **exactly** — ``WEXITSTATUS``
    on a normal exit, ``-1`` on anything else — so the ``exit_code=`` field
    in the failure message means what it has always meant. The description is
    purely additive, and it recovers what a ``WIFEXITED``-only decode throws
    away: the signal number. Without it an OOM kill, a segfault, an abort and
    this pool's own SIGTERM/SIGKILL of a sibling after a *different* lane
    failed all render as ``exit_code=-1 status=ERROR err=no status file
    written``, one message for four unrelated causes.

    ``pool_initiated`` marks the deaths this pool caused itself. Labelling
    those matters: reporting a self-inflicted SIGKILL as a probable OOM kill
    would manufacture memory-pressure evidence out of our own teardown, which
    is worse than the ambiguity being fixed.
    """
    if os.WIFEXITED(status_word):
        code = os.WEXITSTATUS(status_word)
        return code, f"exited({code})"
    if os.WIFSIGNALED(status_word):
        sig = os.WTERMSIG(status_word)
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"SIG{sig}"
        desc = f"killed by signal {sig} ({name})"
        if pool_initiated:
            desc += " [sent by this pool's sibling abort, not an external kill]"
        if os.WCOREDUMP(status_word):
            desc += " [core dumped]"
        return -1, desc
    if os.WIFSTOPPED(status_word):
        return -1, f"stopped by signal {os.WSTOPSIG(status_word)} (not reaped)"
    return -1, f"unrecognized wait status 0x{status_word:04x}"


def _warn_on_fatal_signal(
    lane_idx: int, pid: int, status_word: int, pool_initiated: bool
) -> None:
    """Explain a signalled child death, mirroring the compiler path.

    Same shape and the same three signals the shipped ``neuronx-cc`` wrapper
    already explains in ``compile/backend.py`` (``-9`` SIGKILL/OOM, ``-6``
    SIGABRT, ``-11`` SEGFAULT). Suppressed for pool-initiated kills, which
    carry no information about the host.
    """
    if pool_initiated or not os.WIFSIGNALED(status_word):
        return
    sig = os.WTERMSIG(status_word)
    if sig == signal.SIGKILL:
        logger.warning(
            "Trace lane=%d pid=%d was killed (SIG_KILL) and this pool did not "
            "send it. An external SIGKILL to a tracing child is "
            "characteristically the Linux Out Of Memory (OOM) killer "
            "reclaiming host RAM. Each child carries its own Dynamo / "
            "torch_xla state, and every rank on this host runs its own pool, "
            "so peak host demand is the SUM over ranks and not one rank's "
            "lane count. Consider capping concurrent children across ranks "
            "with VLLM_NEURON_MAX_CONCURRENT_TRACERS (which requires "
            "VLLM_NEURON_TRACER_ADMIT_TIMEOUT), lowering "
            "VLLM_NEURON_PARALLEL_TRACE_WORKERS, or tracing on an instance "
            "with more memory.",
            lane_idx,
            pid,
        )
    elif sig == signal.SIGABRT:
        logger.warning(
            "Trace lane=%d pid=%d aborted (SIG_ABORT). This is likely an "
            "unexpected internal condition (a bug) in a native extension "
            "reached during trace rather than host memory pressure.",
            lane_idx,
            pid,
        )
    elif sig == signal.SIGSEGV:
        logger.warning(
            "Trace lane=%d pid=%d crashed (SEGFAULT). Note that the child "
            "runs after os.fork() from a fully initialized worker; a "
            "segfault here is often an inherited native handle being used "
            "in the child rather than a fault in trace itself.",
            lane_idx,
            pid,
        )


# ---------------------------------------------------------------------------
# Per-lane child entrypoint + status I/O
# ---------------------------------------------------------------------------


def _fork_child_main(
    lane_idx: int,
    parent_rank: int,
    jobs_slice: list[Job],
    result_path: str,
) -> None:
    """Run inside the forked child.

    1. Tag log records with [trace lane=N rank=R] so interleaved output
       is readable.
    2. Set ``VLLM_NEURON_CPU_COMPILE=1`` so the capture backend's
       device validator accepts meta inputs.
    3. Meta-swap the unique underlying nn.Modules referenced by this
       lane's jobs (once each, regardless of how many jobs reuse them).
    4. Run each job: ``callable(**kwargs)``. The capture backend raises
       ``CaptureComplete`` after writing the HLO — swallowed here.
    5. Write a status file the parent reads after waitpid.
    """
    from vllm_neuron.compile.capture_backend import CaptureComplete

    status = "OK"
    err: str | None = None
    failing_job: int | None = None
    try:
        prefix = f"[trace lane={lane_idx} rank={parent_rank}] "

        class _Prefixer(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if not getattr(record, "_trace_prefixed", False):
                    record.msg = prefix + str(record.msg)
                    record._trace_prefixed = True  # type: ignore[attr-defined]
                return True

        for h in logging.getLogger().handlers:
            h.addFilter(_Prefixer())

        os.environ["VLLM_NEURON_CPU_COMPILE"] = "1"

        _swap_unique_models_to_meta([model for model, _ in jobs_slice])

        for j_idx, (model, kwargs) in enumerate(jobs_slice):
            failing_job = j_idx
            try:
                model(**kwargs)
            except CaptureComplete:
                # Successful trace — capture backend signals "done"
                # this way after writing the HLO.
                pass
        failing_job = None
    except BaseException as e:
        status = "ERROR"
        err = (
            f"job_index={failing_job}\n{e}\n{traceback.format_exc()}"
            if failing_job is not None
            else f"{e}\n{traceback.format_exc()}"
        )

    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(f"{status}\n")
        if err:
            f.write(err)
    os.rename(tmp_path, result_path)


def _read_status_file(path: str) -> tuple[str, str | None]:
    if not os.path.exists(path):
        return ("ERROR", "no status file written")
    with open(path) as f:
        content = f.read()
    if not content:
        return ("ERROR", "empty status file")
    lines = content.split("\n", 1)
    return (lines[0].strip(), lines[1] if len(lines) > 1 else None)


# ---------------------------------------------------------------------------
# Pre-fork input validation
# ---------------------------------------------------------------------------


def _validate_jobs_on_meta(jobs: list[Job]) -> None:
    """Fail fast if any kwargs tensor is not on the meta device.

    The capture backend would catch a non-meta input later
    (``compile/backend.py::_validate_inputs_on_device``), but only after
    fork — by which point a NRT_STATE_CHILD allocation will have already
    crashed the child. Raising here keeps the diagnostic local to the
    call site that built the inputs, and names the offending kwarg path
    so locating the leak doesn't require bisection.
    """
    for j_idx, (_, kwargs) in enumerate(jobs):
        for path, t in _walk_tensors(kwargs):
            if t.device.type != "meta":
                raise ValueError(
                    f"parallel_trace.parallel_trace: jobs[{j_idx}] kwarg "
                    f"{path!r} is on device {t.device} (expected meta) "
                    f"shape={tuple(t.shape)} dtype={t.dtype}. Build "
                    f"synthetic inputs with device='meta' before passing "
                    f"them to parallel_trace."
                )


def _walk_tensors(obj: Any, path: str = ""):
    """Yield ``(path, tensor)`` pairs for every ``torch.Tensor``
    reachable from ``obj`` via dict / list / tuple / dataclass-like
    attributes. ``path`` is a dotted/bracketed accessor ("a.b[3].c")
    so the validator's error message can name the offending field.
    """
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            sep = "" if not path else "."
            yield from _walk_tensors(v, f"{path}{sep}{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_tensors(v, f"{path}[{i}]")
        return
    # Dataclass-like (e.g. AttentionMetadata). Skip primitives, modules,
    # and anything without a sensible __dict__.
    if hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        type_name = type(obj).__name__
        for k, v in vars(obj).items():
            sep = "" if not path else "."
            yield from _walk_tensors(v, f"{path}{sep}<{type_name}>.{k}")


# ---------------------------------------------------------------------------
# Meta swap — move inherited models to meta inside the child without
# freeing the parent's neuron-device storage (NRT_STATE_CHILD forbids
# deallocations).
# ---------------------------------------------------------------------------


def _swap_unique_models_to_meta(models: list[Any]) -> None:
    """Apply ``_swap_to_meta_no_free`` to each unique underlying nn.Module
    referenced by ``models``. Sibling torch.compile wrappers sharing the
    same ``_orig_mod`` get swapped exactly once.
    """
    seen: set[int] = set()
    for m in models:
        underlying = _underlying_module(m)
        if underlying is None or id(underlying) in seen:
            continue
        if not isinstance(underlying, torch.nn.Module):
            continue
        seen.add(id(underlying))
        _swap_to_meta_no_free(underlying)


def _swap_to_meta_no_free(module: torch.nn.Module) -> None:
    """Replace every parameter, buffer, and tensor-valued attribute in
    ``module`` (recursive) with a meta-device counterpart, *without
    freeing* the original neuron storage.

    ``nn.Module.to("meta")`` would re-assign each parameter via ``_apply``,
    which decrements the refcount of the old neuron-device tensor. When
    that refcount hits zero, ``nrt_tensor_free`` runs — and fails in
    NRT_STATE_CHILD. We keep the originals alive in a module-level list
    so destructors don't fire until child exit.

    Beyond parameters/buffers, this also swaps plain tensor attributes
    (``__dict__`` entries that are torch.Tensor) — KV caches like
    ``self.k_cache`` / ``self.v_cache`` are bound this way via
    ``bind_kv_cache`` and live on the runtime device.

    Storage-identity preservation: when two attribute slots reference
    different views over the *same* underlying storage (the typical KV
    cache pattern: ``typed_tensor[0]`` / ``typed_tensor[1]`` both come
    from a shared raw buffer in ``initialize_kv_cache``), the meta
    replacements must also share storage. The capture backend's input
    dedup (``compile/backend.py::_detect_duplicate_inputs``) keys on
    ``(id(untyped_storage), storage_offset, shape, stride, dtype)``,
    so any two slots with that same key collapse to a single FX
    placeholder. If we were to allocate a fresh meta storage per slot,
    those keys would diverge — placeholder count balloons (e.g. KV
    caches go from 12 → 24 in GPT-OSS-20B), HBM usage doubles, and the
    HLO verifier rejects the graph.

    The ``storage_to_meta`` cache below maps each unique source storage
    to a single meta storage; per-slot meta tensors are then constructed
    as views over that storage matching the source's offset / shape /
    stride / dtype. ``id_to_meta`` further dedupes by Python identity.
    """
    storage_to_meta: dict[int, torch.UntypedStorage] = {}
    id_to_meta: dict[int, torch.Tensor] = {}

    def _replacement_for(src: torch.Tensor) -> torch.Tensor:
        cached = id_to_meta.get(id(src))
        if cached is not None:
            return cached
        storage_id = id(src.untyped_storage())
        meta_storage = storage_to_meta.get(storage_id)
        if meta_storage is None:
            meta_storage = torch.UntypedStorage(
                src.untyped_storage().nbytes(), device="meta"
            )
            storage_to_meta[storage_id] = meta_storage
        repl = torch.empty(0, dtype=src.dtype, device="meta").set_(
            meta_storage,
            src.storage_offset(),
            src.shape,
            src.stride(),
        )
        id_to_meta[id(src)] = repl
        # Hold a strong reference to the source so its storage survives
        # until child exit (no nrt_tensor_free calls).
        _META_PARAM_KEEPALIVE.append(src)
        return repl

    for submod in module.modules():
        for name, param in list(submod._parameters.items()):
            if param is None or param.device.type == "meta":
                continue
            meta_t = _replacement_for(param)
            submod._parameters[name] = torch.nn.Parameter(
                meta_t, requires_grad=param.requires_grad
            )
        for name, buf in list(submod._buffers.items()):
            if buf is None or buf.device.type == "meta":
                continue
            submod._buffers[name] = _replacement_for(buf)
        # Plain tensor attributes (e.g. k_cache / v_cache bound after
        # initialize_kv_cache). Skip the special _parameters / _buffers
        # / _modules dicts — already handled above.
        for name, val in list(submod.__dict__.items()):
            if name in ("_parameters", "_buffers", "_modules"):
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if val.device.type == "meta":
                continue
            submod.__dict__[name] = _replacement_for(val)


_META_PARAM_KEEPALIVE: list = []
"""Holds references to the original neuron parameters/buffers we replaced
during the in-child meta swap. The Python destructor for a neuron
tensor calls ``nrt_tensor_free``, which fails in NRT_STATE_CHILD. By
keeping a strong reference, we defer the free until child exit (where
the OS reaps process memory directly without going through NRT)."""


def _underlying_module(model: Any) -> Any:
    """Return the underlying nn.Module of a torch.compile-wrapped model.

    OptimizedModule keeps a reference to the original module under
    ``_orig_mod``. Mutations on that propagate to all sibling
    OptimizedModule wrappers that share the same underlying module.
    """
    if model is None:
        return None
    return getattr(model, "_orig_mod", model)
