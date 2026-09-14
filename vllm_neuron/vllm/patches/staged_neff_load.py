# SPDX-License-Identifier: Apache-2.0
"""Stages the compiled graph's load, so one wave of ranks loads at a time.

The compiler backend builds the executable for a graph in one place, and building it loads the
whole graph into host memory; the execution that follows holds far less. A tensor-parallel
group that loads on every rank at one instant therefore exhausts the host long before it runs
anything. This module wraps that builder and lets the ranks through it in waves.

The signals between waves point FORWARD only: a wave waits for the wave before it and is never
waited on again. By the time a wave has signalled, its own ranks are inside the forward's
collectives, where a group-wide wait would deadlock.
"""

import functools
import math
import os
import re
import time

from vllm.logger import init_logger

from vllm_neuron import envs

logger = init_logger(__name__)

POLL_SECONDS = 0.2

_LOAD_NAME = {"phase": "unknown", "bucket": "unknown"}


def name_this_load(phase: str, bucket: str) -> None:
    """Tell the loader which warmup call the graph loads that follow belong to."""
    _LOAD_NAME["phase"] = phase
    _LOAD_NAME["bucket"] = bucket


def wave_plan(world_size: int) -> tuple[int, int]:
    """Return how many ranks load together, and how many waves that takes."""
    wave_size = envs.VLLM_NEURON_NEFF_LOAD_WAVE_SIZE
    if wave_size <= 0 or wave_size >= world_size:
        return world_size, 1
    return wave_size, math.ceil(world_size / wave_size)


def stage_load(build, *args, **kwargs):
    """Run the graph builder inside this rank's own wave, then signal the next wave."""
    rank, world_size = _rank_and_world(args, kwargs)
    wave_size, waves = wave_plan(world_size)
    phase, bucket = _LOAD_NAME["phase"], _LOAD_NAME["bucket"]
    directory = _signal_directory(phase, bucket)
    if waves == 1 or directory is None:
        return build(*args, **kwargs)
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
        return build(*args, **kwargs)
    finally:
        # The row and the flag are written even when the build raises, because a wave that
        # never signals leaves every later wave waiting out its whole timeout.
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


def apply_staged_neff_load(backend=None) -> None:
    """Wrap the compiler backend's graph builder with the staged load."""
    # The builder's own module attribute is the only patchable point: the executable class it
    # returns is defined inside its body, and the backend resolves the builder's name from
    # module globals when it calls it.
    if backend is None:
        from libtorch_neuronx_lite.compile import backend as libtorch_backend

        backend = libtorch_backend
    original = backend.build_executable
    if getattr(original, "staged_neff_load", False):
        return

    @functools.wraps(original)
    def staged(*args, **kwargs):
        return stage_load(original, *args, **kwargs)

    staged.staged_neff_load = True
    backend.build_executable = staged
    logger.info("neff_load_staged|builder=%s", getattr(original, "__name__", "unknown"))


def _rank_and_world(args, kwargs) -> tuple[int, int]:
    """Read the rank and the world size out of the builder's own arguments."""
    if "g_device_id" in kwargs and "g_device_count" in kwargs:
        return int(kwargs["g_device_id"]), int(kwargs["g_device_count"])
    return int(args[2]), int(args[3])


def _signal_directory(phase: str, bucket: str) -> str | None:
    """Return this call's rendezvous directory, or None when none is configured."""
    root = envs.VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR
    if not root:
        return None
    return os.path.join(root, _one_component(phase), _one_component(bucket))


def _one_component(name: str) -> str:
    """Return a name that is safe as a single path component."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "unknown"


def _wait_for_previous_wave(
    directory: str, rank: int, wave: int, wave_size: int, world_size: int
) -> None:
    """Wait until every rank of the wave before this one has signalled."""
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
    """Write this rank's flag, which is the only signal the next wave waits for."""
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
