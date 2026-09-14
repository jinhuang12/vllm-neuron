# SPDX-License-Identifier: Apache-2.0
"""Stages the compiled graph's load, so one wave of ranks loads at a time.

The compiler builds the executable for a graph in one place, and building it loads the whole
graph into host memory; the execution that follows holds far less. A tensor-parallel group that
loads on every rank at one instant therefore exhausts the host long before it runs anything.
This module brackets the compiler the plugin registers with torch.compile and lets the ranks
through it in waves. The bracket includes the compiler's own cache lookup, which is cheap on a
hit; a miss inside it would serialize a compilation as well as a load.

The signals between waves point FORWARD only: a wave waits for the wave before it and is never
waited on again. By the time a wave has signalled, its own ranks are inside the forward's
collectives, where a group-wide wait would deadlock. Each rank writes its flag after the
compiler returns, so a rank that blocks on the way out holds nobody up, and a wave that never
signals costs the next wave one bounded wait rather than the run.

Each load gets its own flag set. One warmup window builds several graphs, so a set shared by the
window would let every graph after the first pass the wait at once and load on the whole group
together, which is the failure the waves exist to prevent. The set is named by the run, the
phase, the bucket and the load's ordinal for this rank. The ordinal is counted locally and
needs no collective: every rank runs the same warmup code and builds the same graphs in the same
order. The first wave of a later load follows the last wave of the load before it, so the ranks
inside a compiler at any instant stay down to one wave across a whole window.

The stage is armed only for a load a caller has named. Anything else passes straight through.
"""

import functools
import math
import os
import re
import threading
import time

import torch.distributed as dist
from vllm.logger import init_logger

from vllm_neuron import envs

logger = init_logger(__name__)

POLL_SECONDS = 0.2
RUN_TOKEN_ENV = "VLLM_NEURON_NEFF_LOAD_RUN"
UNNAMED_RUN = "run_unnamed"

_LOAD_NAME: dict[str, str] = {}
_LOAD_ORDINAL: dict[tuple[int, str, str], int] = {}
_ORDINAL_LOCK = threading.Lock()
_ROOTS_LOGGED: set[str] = set()


def pin_this_run() -> str:
    """Name this run once, so its ranks never read a flag another attempt left behind."""
    if not envs.VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR:
        return ""
    token = os.environ.get(RUN_TOKEN_ENV)
    if not token:
        # Set here and inherited by every worker this process starts, as the plugin already does
        # for the prometheus directory. A rank that reads no token falls back to one shared name,
        # which keeps the ranks of one run together and is printed in the neff_load_root row.
        token = "run_%d_%d" % (os.getpid(), time.time_ns())
        os.environ[RUN_TOKEN_ENV] = token
    return token


def name_this_load(phase: str, bucket: str) -> None:
    """Tell the loader which warmup call the graph loads that follow belong to."""
    _LOAD_NAME["phase"] = phase
    _LOAD_NAME["bucket"] = bucket


def forget_this_load() -> None:
    """Disarm the stage, so a later compilation of any graph passes straight through."""
    _LOAD_NAME.clear()


def wave_plan(world_size: int) -> tuple[int, int]:
    """Return how many ranks load together, and how many waves that takes."""
    wave_size = envs.VLLM_NEURON_NEFF_LOAD_WAVE_SIZE
    if wave_size <= 0 or wave_size >= world_size:
        return world_size, 1
    return wave_size, math.ceil(world_size / wave_size)


def staged_compiler(compile_fn):
    """Return the given compiler, bracketed by this rank's own load wave."""

    @functools.wraps(compile_fn)
    def staged(*args, **kwargs):
        return stage_load(compile_fn, *args, **kwargs)

    staged.__wrapped__ = compile_fn
    staged.staged_neff_load = True
    return staged


def install_staged_compiler(registry, name: str, compile_fn):
    """Put the bracketed compiler under this backend name, wrapping whatever holds it now."""
    registered = _registered_compiler(registry, name)
    if getattr(registered, "staged_neff_load", False):
        return registered
    staged = staged_compiler(registered if registered is not None else compile_fn)
    _forget_registration(registry, name)
    registry.register_backend(compiler_fn=staged, name=name)
    logger.info(
        "neff_load_registered|name=%s|wrapped=%s.%s|was_registered=%s",
        name,
        getattr(staged.__wrapped__, "__module__", "?"),
        getattr(staged.__wrapped__, "__name__", "?"),
        registered is not None,
    )
    return staged


def stage_load(compile_fn, *args, **kwargs):
    """Run the compiler inside this rank's own wave, then signal the wave behind."""
    rank, world_size = _rank_and_world()
    phase = _LOAD_NAME.get("phase")
    bucket = _LOAD_NAME.get("bucket")
    run_root = _run_root()
    reason = _gate_off_reason(rank, world_size, phase, bucket, run_root)
    if reason is not None:
        logger.info("neff_load_gate|off|reason=%s", reason)
        return compile_fn(*args, **kwargs)
    load = _next_load(rank, phase, bucket)
    name_root = os.path.join(run_root, _one_component(phase), _one_component(bucket))
    directory = _load_directory(name_root, load)
    wave_size, waves = wave_plan(world_size)
    wave = rank // wave_size
    os.makedirs(directory, exist_ok=True)
    if run_root not in _ROOTS_LOGGED:
        _ROOTS_LOGGED.add(run_root)
        logger.info("neff_load_root|rank=%s|root=%s", rank, run_root)
    in_front = _wave_in_front(name_root, wave, waves, load)
    if in_front is not None:
        _wait_for_the_wave_in_front(in_front, rank, wave, wave_size, world_size, load)
    logger.info(
        "neff_load_entered|rank=%s|wave=%s|rss_kib=%s|phase=%s|bucket=%s|load=%s",
        rank,
        wave,
        _rss_anon_kib(),
        phase,
        bucket,
        load,
    )
    started = time.perf_counter()
    try:
        return compile_fn(*args, **kwargs)
    finally:
        # The row and the flag are written even when the compilation raises, because a wave that
        # never signals leaves the wave behind it waiting out its whole timeout.
        logger.info(
            "neff_load_left|rank=%s|wave=%s|rss_kib=%s|elapsed_s=%.1f|phase=%s|bucket=%s|load=%s",
            rank,
            wave,
            _rss_anon_kib(),
            time.perf_counter() - started,
            phase,
            bucket,
            load,
        )
        _signal_this_rank(directory, rank)


def _registered_compiler(registry, name: str):
    """Return the compiler this backend name holds now, or None when the name is free."""
    try:
        return registry.lookup_backend(name)
    except Exception:  # the registry raises its own type for an unknown name
        return None


def _forget_registration(registry, name: str) -> None:
    """Drop this name from the registry's tables, which refuse a second registration of it."""
    for table in ("_COMPILER_FNS", "_BACKENDS"):
        held = getattr(registry, table, None)
        if isinstance(held, dict):
            held.pop(name, None)


def _rank_and_world() -> tuple[int | None, int | None]:
    """Read this rank and the world size off the process group, as the compiler itself does."""
    if not dist.is_available() or not dist.is_initialized():
        return None, None
    return dist.get_rank(), dist.get_world_size()


def _gate_off_reason(rank, world_size, phase, bucket, run_root) -> str | None:
    """Say why this call is not staged, or None when it is."""
    if rank is None or world_size is None:
        return "no_process_group"
    if phase is None or bucket is None or not run_root:
        return "no_named_load"
    if envs.VLLM_NEURON_NEFF_LOAD_WAVE_SIZE <= 0:
        return "wave_size_zero"
    if wave_plan(world_size)[1] == 1:
        return "wave_size_covers_the_world"
    return None


def _run_root() -> str | None:
    """Return the directory this run signals in, or None when no directory is configured."""
    root = envs.VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR
    if not root:
        return None
    return os.path.join(root, _one_component(os.environ.get(RUN_TOKEN_ENV) or UNNAMED_RUN))


def _next_load(rank: int, phase: str, bucket: str) -> int:
    """Return this load's ordinal under its name, counting from zero for this rank."""
    with _ORDINAL_LOCK:
        key = (rank, phase, bucket)
        load = _LOAD_ORDINAL.get(key, 0)
        _LOAD_ORDINAL[key] = load + 1
        return load


def _one_component(name: str) -> str:
    """Return a name that is safe as a single path component."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "unknown"


def _load_directory(name_root: str, load: int) -> str:
    """Return the directory the ranks of one load signal each other in."""
    return os.path.join(name_root, "load_%d" % load)


def _wave_in_front(name_root: str, wave: int, waves: int, load: int):
    """Return the load directory and wave index this wave follows, or None when it is the first.

    A later wave of one load follows the wave before it. The first wave of a later load follows
    the LAST wave of the load before it, which keeps the ranks inside a compiler at any instant
    down to one wave across a window that builds several graphs.
    """
    if wave:
        return _load_directory(name_root, load), wave - 1
    if load:
        return _load_directory(name_root, load - 1), waves - 1
    return None


def _wait_for_the_wave_in_front(
    in_front, rank: int, wave: int, wave_size: int, world_size: int, load: int
) -> None:
    """Wait until every rank of the wave in front has signalled, or give up and proceed."""
    directory, ahead = in_front
    first = ahead * wave_size
    wanted = [
        os.path.join(directory, f"rank_{other}.done")
        for other in range(first, min(first + wave_size, world_size))
    ]
    timeout = envs.VLLM_NEURON_NEFF_LOAD_WAIT_TIMEOUT
    started = time.perf_counter()
    while True:
        missing = [path for path in wanted if not os.path.exists(path)]
        if not missing:
            return
        waited = time.perf_counter() - started
        if waited >= timeout:
            logger.warning(
                "neff_load_wait_timeout|rank=%s|wave=%s|waited_s=%.1f|missing=%s|load=%s"
                "|waited_for=%s",
                rank,
                wave,
                waited,
                len(missing),
                load,
                os.path.join(os.path.basename(directory), "wave_%d" % ahead),
            )
            return
        time.sleep(POLL_SECONDS)


def _signal_this_rank(directory: str, rank: int) -> None:
    """Write this rank's flag, which is the only signal the wave behind it waits for."""
    with open(os.path.join(directory, f"rank_{rank}.done"), "w", encoding="utf-8") as flag:
        flag.write(f"{os.getpid()}\n")


def _rss_anon_kib() -> int:
    """Read this process's anonymous resident size in KiB, or zero where /proc is not readable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0
