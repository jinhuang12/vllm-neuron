# SPDX-License-Identifier: Apache-2.0
"""The parallel compile compiles only the graphs that this start's extraction captured.

A compile cache root can hold incomplete entries that this start did not capture: the
HLOs that an earlier start wrote before it stopped, with no NEFF. The compile cache
library's parallel compile compiles every incomplete entry that holds an HLO of this
server, and then removes this server's rank directories from every entry.

The tests run ``NeuronModelRunner.recording_graph_captures`` around a fake graph
extraction, then ``NeuronModelRunner.parallel_compile`` and the library's own compile
loop, on a real cache directory. Before the start it holds:

* three stale incomplete entries: one with another rank's HLO, one with this rank's
  HLO, and one that this start captures again;
* two complete entries that this start finds compiled (warm hits), one of them with a
  rank directory that an earlier start left behind.

The test writes each capture as the capture backend does (a lookup, then the rank's
``graph.hlo`` and metadata on a miss), in this process or in the library's forked
trace children; it does not call the capture backend. The TP group and its gather are
fakes. The library's compile loop, fork pool and cache functions are real; only the
compiler call (``_do_compile``) is replaced: it writes the NEFF and the completion
marker, as the compiler does, and records the key.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import hashlib
import logging
import os
import time
import types
from pathlib import Path

import pytest
import torch.distributed as dist
from libtorch_neuronx_lite.compile import cache
from libtorch_neuronx_lite.compile import parallel_compile as library
from libtorch_neuronx_lite.compile.parallel_trace import parallel_trace
from libtorch_neuronx_lite.compile.schema import create_metadata
from vllm.distributed import parallel_state

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

#: get_server_prefix() for NEURON_VISIBLE_DEVICES=0-63
PREFIX = "dev0_63"


def graph_key(name: str) -> str:
    """A 32-hex compile cache key, as the cache names its entries."""
    return hashlib.md5(name.encode()).hexdigest()


NEW = tuple(sorted(graph_key(f"new-{i}") for i in range(3)))
#: Stale entries this start does not capture: another rank's HLO, this rank's HLO.
STALE = (graph_key("stale-rank40"), graph_key("stale-rank0"))
#: A stale entry with this rank's HLO that this start captures again.
REUSED = graph_key("stale-recaptured")
#: Complete entries this start looks up: a plain one, one with an old rank directory.
WARM = (graph_key("warm"), graph_key("warm-with-old-rank-dir"))
CAPTURED = tuple(sorted(NEW + (REUSED,)))
EXTRACTED = CAPTURED + WARM


def write_capture(entry: Path, rank: int) -> None:
    """What graph capture leaves on a miss: the rank's HLO and metadata, no NEFF."""
    rank_dir = entry / f"{PREFIX}.rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    (rank_dir / "graph.hlo").write_bytes(b"hlo " + entry.name.encode())
    cache.save_artifact_metadata(str(rank_dir), create_metadata(cache_key=entry.name))


def write_complete(entry: Path) -> None:
    """What a finished compile leaves at the entry root: HLO, metadata, NEFF, marker."""
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "graph.hlo").write_bytes(b"hlo " + entry.name.encode())
    cache.save_artifact_metadata(str(entry), create_metadata(cache_key=entry.name))
    (entry / f"graph_{entry.name}.neff").write_bytes(b"neff " + entry.name.encode())
    (entry / cache.NEURON_COMPILE_COMPLETE_FILE).write_text("completed:0\n")


def tree(path: Path, skip: tuple[str, ...] = ()) -> dict[str, bytes | None]:
    """Every directory (None) and file (its bytes) under ``path``, less ``skip``."""
    return {
        str(p.relative_to(path)): None if p.is_dir() else p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.name not in skip
    }


class World:
    """A compile cache directory, a faked TP group and a recording fake compiler."""

    def __init__(self, compile_dir: Path) -> None:
        self.compile_dir = compile_dir
        self.queued: list[str] = []
        self.compiled: list[str] = []
        self.hits: list[str] = []
        self.gathers: list[tuple] = []
        self.other_ranks: list[list[str]] = []
        self.rank = 0
        self.world_size = 1
        self.cpu_group = object()

    def compile_single(self, cache_dir, hlo_key, *args, **kwargs):
        self.queued.append(hlo_key)
        return self._compile_single(cache_dir, hlo_key, *args, **kwargs)

    def do_compile(self, compile_dir, options, hlo_key) -> None:
        self.compiled.append(hlo_key)
        neff = Path(compile_dir, f"graph_{hlo_key}.neff")
        neff.write_bytes(b"neff " + hlo_key.encode())
        marker = Path(compile_dir, cache.NEURON_COMPILE_COMPLETE_FILE)
        marker.write_text("completed:0\n")

    def tp_group(self):
        return types.SimpleNamespace(
            rank_in_group=self.rank,
            world_size=self.world_size,
            cpu_group=self.cpu_group,
        )

    def all_gather_object(self, out, obj, group=None) -> None:
        self.gathers.append((sorted(obj), group))
        rows = [sorted(obj), *self.other_ranks]
        assert len(out) == len(rows) == self.world_size
        out[:] = rows

    def runner(self) -> NeuronModelRunner:
        runner = NeuronModelRunner.__new__(NeuronModelRunner)
        runner.compile_options = {
            "compiler_workdir": str(self.compile_dir),
            "compiler_args": ["-O1"],
        }
        return runner

    def capture(self, key: str) -> None:
        """One graph's capture, as the capture backend makes it."""
        if cache.get_local(key, str(self.compile_dir)) is None:
            write_capture(self.compile_dir / key, self.rank)
        else:
            self.hits.append(key)

    def extract(self, keys, *, forked: bool = False) -> set[str]:
        """Graph extraction of ``keys`` on this rank, recorded."""
        jobs = [(self.capture, {"key": key}) for key in keys]
        with self.runner().recording_graph_captures() as captured:
            if forked:
                parallel_trace(jobs=jobs, parent_rank=self.rank)
            else:
                for job, kwargs in jobs:
                    job(**kwargs)
        return captured

    def is_complete(self, key: str) -> bool:
        return cache._is_cache_complete(str(self.compile_dir / key))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The stale and the complete entries, written an hour before this start."""
    for name in ("NEURON_LIBTORCH_REMOTE_CACHE", "VLLM_NEURON_REMOTE_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEURON_VISIBLE_DEVICES", "0-63")
    compile_dir = tmp_path / "neuron" / "compile_cache"
    write_capture(compile_dir / STALE[0], rank=40)
    write_capture(compile_dir / STALE[1], rank=0)
    write_capture(compile_dir / REUSED, rank=0)
    for key in WARM:
        write_complete(compile_dir / key)
    write_capture(compile_dir / WARM[1], rank=0)
    an_hour_ago = time.time() - 3600
    for path in compile_dir.rglob("*"):
        os.utime(path, (an_hour_ago, an_hour_ago))
    w = World(compile_dir)
    w._compile_single = library._compile_single
    monkeypatch.setattr(library, "_compile_single", w.compile_single)
    monkeypatch.setattr(library, "_do_compile", w.do_compile)
    monkeypatch.setattr(parallel_state, "get_tp_group", w.tp_group)
    monkeypatch.setattr(dist, "all_gather_object", w.all_gather_object)
    return w


def test_library_alone_compiles_every_incomplete_entry(world):
    """The hazard: without a key set, the stale entries compile and lose their HLO dir.

    If the library ever limits its compile to the keys of the start, this fails and
    the filter in ``parallel_compile`` can go.
    """
    for key in EXTRACTED:
        world.capture(key)

    library.parallel_compile(
        options=world.runner().compile_options, rank=0, world_size=1
    )

    assert sorted(world.queued) == sorted(CAPTURED + STALE)
    assert all(world.is_complete(key) for key in STALE)
    assert not list((world.compile_dir / WARM[1]).glob(f"{PREFIX}.rank*"))


@pytest.mark.parametrize("forked", [False, True], ids=["in-process", "forked"])
def test_only_the_graphs_this_start_captured_compile(world, forked, caplog):
    before = {key: tree(world.compile_dir / key) for key in STALE}

    captured = world.extract(EXTRACTED, forked=forked)
    assert captured == set(CAPTURED)
    with caplog.at_level(logging.WARNING):
        world.runner().parallel_compile(captured)

    assert sorted(world.queued) == list(CAPTURED)
    assert sorted(world.compiled) == list(CAPTURED)
    for key in CAPTURED:
        assert world.is_complete(key)
        assert not list((world.compile_dir / key).glob(f"{PREFIX}.rank*"))
    for key in STALE:
        assert tree(world.compile_dir / key) == before[key]
        assert not world.is_complete(key)
    messages = [record.getMessage() for record in caplog.records]
    skipped = [message for message in messages if "were not compiled" in message]
    assert skipped == [
        "Parallel compile: 2 incomplete compile cache entries were not compiled, as no "
        "rank captured them in this start's graph extraction (first: "
        f"{min(STALE)}); they are left as they are"
    ]


def test_a_warm_hit_is_looked_up_and_left_as_it_was(world):
    """A complete entry is a hit: it is not captured, queued or compiled.

    Its files do not change but for the lookup's own access marker, and the rank
    directory that an earlier start left in it stays, as no rank captured it now.
    """
    before = {key: tree(world.compile_dir / key) for key in WARM}

    captured = world.extract(EXTRACTED)
    world.runner().parallel_compile(captured)

    assert sorted(world.hits) == sorted(WARM)
    assert not set(WARM) & captured
    assert not set(WARM) & set(world.queued)
    marker = cache.NEURON_COMPILE_CACHE_ACCESSED_FILE
    for key in WARM:
        assert world.is_complete(key)
        assert tree(world.compile_dir / key, skip=(marker,)) == before[key]


def test_a_capture_into_a_cache_dir_not_made_yet_is_recorded(world, tmp_path):
    world.compile_dir = tmp_path / "fresh" / "neuron" / "compile_cache"

    assert world.extract(NEW) == set(NEW)


def test_graphs_of_every_tp_rank_compile_on_rank_0(world):
    world.world_size = 2
    world.other_ranks = [[NEW[2]]]
    write_capture(world.compile_dir / NEW[2], rank=1)  # rank 1's capture

    world.runner().parallel_compile(world.extract(NEW[:2]))

    assert world.gathers == [(list(NEW[:2]), world.cpu_group)]
    assert sorted(world.queued) == list(NEW)
    assert not any(world.is_complete(key) for key in STALE + (REUSED,))


def test_other_tp_ranks_join_the_gather_and_compile_nothing(world):
    world.rank = 1
    world.world_size = 2
    world.other_ranks = [list(NEW[1:])]
    captured = world.extract(NEW[:1])
    before = tree(world.compile_dir)

    world.runner().parallel_compile(captured)

    assert world.gathers == [(list(NEW[:1]), world.cpu_group)]
    assert world.queued == []
    assert tree(world.compile_dir) == before


def test_library_hooks_are_restored_after_a_failed_compile(world, monkeypatch):
    has_hlo, cleanup = library._has_hlo, library._cleanup_prefix_subdirs

    def fail(compile_dir, options, hlo_key):
        raise RuntimeError(f"compile of {hlo_key} failed")

    monkeypatch.setattr(library, "_do_compile", fail)
    with pytest.raises(RuntimeError, match="compile of .* failed"):
        world.runner().parallel_compile(world.extract(NEW))

    assert library._has_hlo is has_hlo
    assert library._cleanup_prefix_subdirs is cleanup
