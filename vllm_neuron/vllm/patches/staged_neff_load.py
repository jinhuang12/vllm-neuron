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

The stage is armed only for a load a caller has named. Anything else passes straight through.
"""

import math
import os
import re
import time

import torch.distributed as dist
from vllm.logger import init_logger

from vllm_neuron import envs

logger = init_logger(__name__)

POLL_SECONDS = 0.2

_LOAD_NAME: dict[str, str] = {}


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
    """Return the registered compiler, bracketed by this rank's own load wave."""

    def staged(*args, **kwargs):
        return stage_load(compile_fn, *args, **kwargs)

    return staged


def stage_load(compile_fn, *args, **kwargs):
    """Run the compiler inside this rank's own wave, then signal the wave behind."""
    rank, world_size = _rank_and_world()
    phase = _LOAD_NAME.get("phase")
    bucket = _LOAD_NAME.get("bucket")
    directory = _signal_directory(phase, bucket)
    reason = _gate_off_reason(rank, world_size, directory)
    if reason is not None:
        logger.info("neff_load_gate|off|reason=%s", reason)
        return compile_fn(*args, **kwargs)
    wave_size, _ = wave_plan(world_size)
    wave = rank // wave_size
    os.makedirs(directory, exist_ok=True)
    if wave:
        _wait_for_previous_wave(directory, rank, wave, wave_size, world_size)
    logger.info(
        "neff_load_entered|rank=%s|wave=%s|rss_kib=%s|phase=%s|bucket=%s",
        rank,
        wave,
        _rss_anon_kib(),
        phase,
        bucket,
    )
    started = time.perf_counter()
    try:
        return compile_fn(*args, **kwargs)
    finally:
        # The row and the flag are written even when the compilation raises, because a wave that
        # never signals leaves the wave behind it waiting out its whole timeout.
        logger.info(
            "neff_load_left|rank=%s|wave=%s|rss_kib=%s|elapsed_s=%.1f|phase=%s|bucket=%s",
            rank,
            wave,
            _rss_anon_kib(),
            time.perf_counter() - started,
            phase,
            bucket,
        )
        _signal_this_rank(directory, rank)


def _rank_and_world() -> tuple[int | None, int | None]:
    """Read this rank and the world size off the process group, as the compiler itself does."""
    if not dist.is_available() or not dist.is_initialized():
        return None, None
    return dist.get_rank(), dist.get_world_size()


def _gate_off_reason(rank, world_size, directory) -> str | None:
    """Say why this call is not staged, or None when it is."""
    if rank is None or world_size is None:
        return "no_process_group"
    if directory is None:
        return "no_named_load"
    if wave_plan(world_size)[1] == 1:
        return "wave_size_covers_the_world"
    return None


def _signal_directory(phase, bucket) -> str | None:
    """Return this call's signal directory, or None when the load is unnamed or unconfigured."""
    root = envs.VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR
    if not root or phase is None or bucket is None:
        return None
    return os.path.join(root, _one_component(phase), _one_component(bucket))


def _one_component(name: str) -> str:
    """Return a name that is safe as a single path component."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "unknown"


def _wait_for_previous_wave(
    directory: str, rank: int, wave: int, wave_size: int, world_size: int
) -> None:
    """Wait until every rank of the wave before this one has signalled, or give up and proceed."""
    first = (wave - 1) * wave_size
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
                "neff_load_wait_timeout|rank=%s|wave=%s|waited_s=%.1f|missing=%s",
                rank,
                wave,
                waited,
                len(missing),
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
