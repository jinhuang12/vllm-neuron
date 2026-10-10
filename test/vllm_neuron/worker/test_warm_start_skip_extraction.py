# SPDX-License-Identifier: Apache-2.0
"""A start on a compile cache that holds every served graph skips graph extraction.

Every large graph is traced twice at start-up: once by graph extraction, which writes
its HLO for the parallel compile, and once by its warm-up, which loads the NEFF. On a
compile cache that already holds every graph, extraction, its two barriers and the
parallel compile find nothing to do. ``VLLM_NEURON_SKIP_EXTRACTION_WHEN_CACHED`` (on by
default) skips them when the warm-start manifest proves the cache warm:

* a full start that completes its warm-ups writes the manifest: the keys of every
  graph the start looked up, and the identity of the start (package files, versions,
  configs, compile options, compile targets, graph-relevant env);
* the next start skips extraction only when a manifest of the same identity names
  keys whose cache entries are all complete, and when every rank of the TP group
  agrees;
* a cold root, a partial root, a stale or malformed manifest, or one rank that
  disagrees runs the full pass exactly as before (``test_cold_start_call_sequence``).

The tests drive the real ``NeuronWorker.compile_or_warm_up_model`` with the fakes of
``test_cold_start_call_sequence`` on a real cache directory, and the real
``libtorch_neuronx_lite`` backend with its trace and compile faked for the post-start
cases.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import json
import os
import shutil
import types

import pytest
from libtorch_neuronx_lite.compile import backend, cache

from test.vllm_neuron.worker.test_cold_start_call_sequence import (
    BASELINE,
    DECODE_KEYS,
    PREFILL_KEYS,
    START_ENV,
    CompileCache,
    Steps,
    fake_get_local,
    graph_key,
    make_worker,
)
from vllm_neuron.vllm.worker import gc_policy, neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

KNOB = "VLLM_NEURON_SKIP_EXTRACTION_WHEN_CACHED"
ASSERT_CACHE_HIT = "NEURON_LIBTORCH_ASSERT_CACHE_HIT"
ALL_KEYS = PREFILL_KEYS + DECODE_KEYS
#: A warm start: the warm-ups and what follows them, no extraction, barrier or compile.
WARM = [
    ("warmup_prefill",),
    ("warmup_decode",),
    ("register_ec_cache",),
    ("gc_policy",),
]
FULL = BASELINE["both"]
#: The full pass with a vision encoder: extraction covers it, and its warm-up runs.
VISION_FULL = [
    (
        "extract_graphs",
        {
            "native_compile_only": False,
            "skip_decode": False,
            "skip_prefill": False,
            "skip_vision": False,
        },
    ),
    *FULL[1:6],
    ("warmup_vision",),
    *FULL[6:],
]
TP_SIZE = 64


class Rows:
    """The worker's logger, keeping each rendered row with its level."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def _keep(self, level: str, message, *args, **_kwargs) -> None:
        self.rows.append((level, message % args if args else str(message)))

    def debug(self, message, *args, **kwargs) -> None:
        self._keep("debug", message, *args, **kwargs)

    def info(self, message, *args, **kwargs) -> None:
        self._keep("info", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs) -> None:
        self._keep("warning", message, *args, **kwargs)

    def error(self, message, *args, **kwargs) -> None:
        self._keep("error", message, *args, **kwargs)

    def text(self, level: str | None = None) -> str:
        return "\n".join(row for lvl, row in self.rows if level in (None, lvl))

    def clear(self) -> None:
        self.rows.clear()


class Starts:
    """Starts of one worker on one compile cache root, one after the other."""

    def __init__(self, compile_cache: CompileCache) -> None:
        self.cache = compile_cache
        self.steps: Steps | None = None
        #: How many other ranks of the TP group vote that their cache is cold.
        self.cold_ranks = 0
        self.tp_rank = 0
        self.local_rank = 0
        self.log = Rows()
        #: How many steps had run when each vote was taken.
        self.vote_at: list[int] = []

    def tp_sum_int(self, value: int, timeout=None) -> int:
        self.steps.sums.append(value)
        self.vote_at.append(len(self.steps.events))
        return value + self.cold_ranks

    def tp_barrier(self) -> None:
        self.steps.tp_barrier()

    def tp_group(self):
        return types.SimpleNamespace(rank_in_group=self.tp_rank, world_size=TP_SIZE)

    def run(self, *, prefill=PREFILL_KEYS, decode=DECODE_KEYS, edit=None) -> Steps:
        """One start; ``edit(worker)`` changes the loaded worker before it starts."""
        self.steps = Steps(self.cache, prefill, decode)
        self.log.clear()
        self.vote_at = []
        worker = make_worker(self.steps)
        worker.local_rank = self.local_rank
        if edit is not None:
            edit(worker)
        NeuronWorker.compile_or_warm_up_model(worker)
        return self.steps

    @property
    def warm_start_dir(self):
        return self.cache.root / "neuron" / "warm_start"

    def manifests(self) -> list:
        if not self.warm_start_dir.is_dir():
            return []
        return sorted(p for p in self.warm_start_dir.iterdir() if p.suffix == ".json")

    def manifest(self):
        (path,) = self.manifests()
        return path

    def manifest_data(self) -> dict:
        return json.loads(self.manifest().read_text())


@pytest.fixture
def starts(tmp_path, monkeypatch):
    """Starts on one empty compile cache root; the worker is TP rank 0 of 64."""
    for name in START_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("VLLM_NEURON_GLUE_FUSED", "NEURON_RT_VISIBLE_CORES", "VLLM_NEURON_NEFF_LOAD_RUN"):
        monkeypatch.delenv(name, raising=False)
    compile_cache = CompileCache(tmp_path / "root")
    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(compile_cache.root))
    monkeypatch.setattr(cache, "get_local", fake_get_local)
    runs = Starts(compile_cache)
    monkeypatch.setattr(neuron_worker, "tp_barrier", runs.tp_barrier)
    monkeypatch.setattr(neuron_worker, "tp_sum_int", runs.tp_sum_int)
    monkeypatch.setattr(neuron_worker, "get_tp_group", runs.tp_group)
    monkeypatch.setattr(neuron_worker, "logger", runs.log)
    monkeypatch.setattr(
        gc_policy,
        "apply_post_warmup_gc_policy",
        lambda: runs.steps.events.append(("gc_policy",)),
    )
    return runs


# ------------------------------------------------------------------ (a) warm root


def test_a_warm_root_skips_extraction_and_compile_and_runs_every_warm_up(starts):
    """Second start on the same root: warm-ups only, after one unanimous vote."""
    starts.run()
    before = {key: sorted(os.listdir(starts.cache.entry(key))) for key in ALL_KEYS}

    second = starts.run()

    assert second.events == WARM
    assert second.sums == [0]
    assert "skipping graph extraction" in starts.log.text("info")
    # No "Extracting" line at all: the device line greps the server log for it.
    assert "Extracting" not in starts.log.text()
    # The warm start leaves the compile cache directory as it found it.
    assert starts.cache.entries() == sorted(ALL_KEYS)
    assert {key: sorted(os.listdir(starts.cache.entry(key))) for key in ALL_KEYS} == before


def test_a_warm_start_does_not_rewrite_the_manifest(starts):
    """Only a full pass writes the manifest."""
    starts.run()
    path = starts.manifest()
    text, mtime = path.read_text(), path.stat().st_mtime_ns

    starts.run()

    assert starts.manifest() == path
    assert path.read_text() == text
    assert path.stat().st_mtime_ns == mtime


def test_a_warm_start_reproduces_on_the_next_start(starts):
    """A third start from the same root skips again."""
    starts.run()
    starts.run()

    third = starts.run()

    assert third.events == WARM
    assert third.sums == [0]


# ------------------------------------------------------------------ (b) cold root


def test_a_cold_root_votes_cold_once_then_runs_the_full_pass(starts):
    """Empty root: one vote of 1, then the recorded cold-start sequence."""
    first = starts.run()

    assert first.events == FULL
    assert first.sums == [1]
    assert starts.vote_at == [0]  # before any extraction, barrier or warm-up step
    assert "no manifest" in starts.log.text("info")


def test_a_full_pass_writes_the_manifest_after_its_warm_ups(starts):
    """The manifest names every graph the start looked up, outside the cache dir."""
    starts.run()

    data = starts.manifest_data()
    assert data["format"] == neuron_worker.WARM_START_FORMAT
    assert data["keys"] == sorted(ALL_KEYS)
    assert isinstance(data["identity"], dict) and data["identity"]
    # The manifest is not a compile cache entry: the cache dir holds the graphs only.
    assert starts.cache.entries() == sorted(ALL_KEYS)
    assert starts.manifest().parent == starts.warm_start_dir


def test_a_warm_root_without_a_manifest_runs_the_full_pass_and_writes_one(starts):
    """A root that a tree without the skip filled: one full pass, then warm starts."""
    for key in ALL_KEYS:
        starts.cache.complete(key)

    first = starts.run()
    second = starts.run()

    assert first.events == FULL
    assert first.sums == [1]
    assert second.events == WARM


# ------------------------------------------------------------------ (c) partial root


def _remove_entry(entry):
    shutil.rmtree(entry)


def _remove_file(name):
    def remove(entry):
        (entry / name.format(key=entry.name)).unlink()

    return remove


@pytest.mark.parametrize(
    "damage",
    [
        _remove_entry,
        _remove_file("graph_{key}.neff"),
        _remove_file("graph.hlo"),
        _remove_file(".compilation_complete"),
        _remove_file(".artifact_metadata_v0.json"),
    ],
    ids=["entry", "neff", "hlo", "complete-marker", "metadata"],
)
def test_a_partial_root_runs_the_full_pass(starts, damage):
    """One incomplete entry of the manifest: the full pass, naming the key."""
    starts.run()
    missing = DECODE_KEYS[1]
    damage(starts.cache.entry(missing))

    second = starts.run()

    assert second.events == FULL
    assert second.sums == [1]
    assert missing in starts.log.text("info")
    assert "1 of 4" in starts.log.text("info")
    assert starts.cache.is_complete(missing)


# ------------------------------------------------------------------ (d) stale manifest


def _set_env(name, value):
    def edit(worker, monkeypatch):
        monkeypatch.setenv(name, value)

    return edit


def _scheduler(worker, monkeypatch):
    worker.vllm_config.scheduler_config.max_num_seqs = 64


def _compile_options(worker, monkeypatch):
    worker.model_runner.compile_options["compiler_args"] = ["-O2"]


def _neuron_config(worker, monkeypatch):
    worker.vllm_config.additional_config["neuron_config"] = {"ep_degree": 16}


def _buckets(worker, monkeypatch):
    worker.model_runner.neuron_config.kv_segment_size_buckets = [1024, 2048, 4096]


def _package(worker, monkeypatch):
    monkeypatch.setattr(neuron_worker, "_warm_start_package_digest", lambda: "0" * 64)


def _parallel(worker, monkeypatch):
    worker.vllm_config.parallel_config.compute_hash = lambda: "parallel-hash-tp32"


def _kv_cache(worker, monkeypatch):
    worker.model_runner.kv_cache_config = types.SimpleNamespace(
        num_blocks=101, kv_cache_tensors=[types.SimpleNamespace(size=101 * 4096)]
    )


def _hf_config(worker, monkeypatch):
    from transformers import PretrainedConfig

    worker.vllm_config.model_config.hf_config = PretrainedConfig(num_hidden_layers=3)


@pytest.mark.parametrize(
    "change, component",
    [
        (_package, "package"),
        (_set_env("VLLM_NEURON_GLUE_FUSED", "0"), "env:VLLM_NEURON_GLUE_FUSED"),
        (_scheduler, "scheduler"),
        (_compile_options, "compile_options"),
        (_neuron_config, "additional_config"),
        (_buckets, "compile_targets"),
        (_parallel, "vllm_config"),
        (_kv_cache, "kv_cache"),
        (_hf_config, "hf_config"),
    ],
    ids=[
        "package",
        "env",
        "scheduler",
        "compile-options",
        "additional-config",
        "buckets",
        "parallel",
        "kv-cache-blocks",
        "hf-config",
    ],
)
def test_a_changed_identity_runs_the_full_pass_and_names_what_changed(
    starts, monkeypatch, change, component
):
    """A digest, env or config change: no manifest of the new identity, full pass."""
    starts.run()

    second = starts.run(edit=lambda worker: change(worker, monkeypatch))

    assert second.events == FULL
    assert second.sums == [1]
    assert component in starts.log.text("info")
    # The full pass writes the manifest of its own identity beside the first one.
    assert len(starts.manifests()) == 2


def test_a_manifest_whose_identity_does_not_match_its_name_is_refused(starts):
    """A manifest at the start's path but of another identity: stale, full pass."""
    starts.run()
    data = starts.manifest_data()
    data["identity"]["package"] = "0" * 64
    starts.manifest().write_text(json.dumps(data))

    second = starts.run()

    assert second.events == FULL
    assert "stale" in starts.log.text("info")
    assert "package" in starts.log.text("info")


@pytest.mark.parametrize(
    "per_run",
    [
        _set_env("NEURON_RT_VISIBLE_CORES", "17"),
        _set_env("NEURON_RT_ROOT_COMM_ID", "10.0.0.1:62182"),
        _set_env("VLLM_NEURON_NEFF_LOAD_RUN", "run-2"),
        _set_env("VLLM_NEURON_NEFF_LOAD_SIGNAL_DIR", "/runs/2/neff_load"),
        _set_env("VLLM_DP_MASTER_PORT", "40123"),
        _set_env("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "1200"),
        _set_env("VLLM_NEURON_DISABLE_PARALLEL_TRACE", "1"),
        lambda worker, monkeypatch: worker.vllm_config.additional_config.update(
            neuron_profiler={"output_dir": "/runs/2/nrt_profile"}
        ),
    ],
    ids=[
        "visible-cores",
        "root-comm-id",
        "neff-load-run",
        "neff-load-signal-dir",
        "dp-port",
        "compile-timeout",
        "parallel-trace",
        "profiler-output-dir",
    ],
)
def test_a_per_run_value_keeps_the_root_warm(starts, monkeypatch, per_run):
    """Values that change per run or per rank, not the graphs, keep the skip."""
    starts.run()

    second = starts.run(edit=lambda worker: per_run(worker, monkeypatch))

    assert second.events == WARM


# ------------------------------------------------------------------ (e) the assert flag


def _post_start_compile(monkeypatch, key):
    """Compile one new graph after the start through the real backend entry point.

    The trace, the compile and the executable are faked; the cache lookup, the
    assert-flag check and the cache-miss path are the backend's own.
    """
    compiled = []
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "0")
    monkeypatch.setattr(backend, "preprocess_and_validate_inputs", lambda gm, ex, opt: (gm, ex))
    monkeypatch.setattr(backend, "_apply_platform_compiler_args", lambda options: options)
    monkeypatch.setattr(cache, "create_cache_hash", lambda gm, ex, opt: key)

    def fetch_remote_or_compile(hash_key, local_cache_dir, callback, timeout, **kwargs):
        compiled.append(hash_key)
        return types.SimpleNamespace(hlo_filename="graph.hlo", neff_filename="graph.neff")

    monkeypatch.setattr(cache, "fetch_remote_or_compile", fetch_remote_or_compile)
    monkeypatch.setattr(backend, "build_executable", lambda **kwargs: "executable")
    result = backend.compile(types.SimpleNamespace(), [], {"compiler_args": []})
    return result, compiled


def test_after_a_warm_start_a_new_graph_compiles_instead_of_raising(starts, monkeypatch):
    """The skip never sets the assert flag: a graph first seen after READY compiles."""
    starts.run()
    assert starts.run().events == WARM

    assert ASSERT_CACHE_HIT not in os.environ
    assert cache.get_local is fake_get_local
    new_key = graph_key("first-seen-after-ready")
    result, compiled = _post_start_compile(monkeypatch, new_key)

    assert result == "executable"
    assert compiled == [new_key]


def test_an_operator_set_assert_flag_keeps_the_skip_and_its_own_contract(starts, monkeypatch):
    """With the flag set by the operator the warm start still skips; the flag still raises."""
    starts.run()
    monkeypatch.setenv(ASSERT_CACHE_HIT, "1")

    second = starts.run()

    assert second.events == WARM
    assert os.environ[ASSERT_CACHE_HIT] == "1"
    with pytest.raises(RuntimeError, match="NEURON_LIBTORCH_ASSERT_CACHE_HIT"):
        _post_start_compile(monkeypatch, graph_key("first-seen-after-ready"))


# ------------------------------------------------------------------ (f) manifest writes


def test_the_manifest_is_written_only_after_the_warm_ups_succeed(starts):
    """A failed warm-up raises as before and leaves no manifest; the next start writes it."""

    def failing_decode():
        starts.steps.events.append(("warmup_decode",))
        raise RuntimeError("decode warm-up failed")

    def edit(worker):
        worker._warmup_decode = failing_decode

    with pytest.raises(RuntimeError, match="decode warm-up failed"):
        starts.run(edit=edit)

    assert starts.manifests() == []
    assert cache.get_local is fake_get_local
    second = starts.run()
    assert second.events == FULL
    assert starts.manifest_data()["keys"] == sorted(ALL_KEYS)


def test_only_local_rank_0_writes_the_manifest(starts):
    """Local rank 5 runs the full pass and writes nothing; local rank 0 writes it."""
    starts.tp_rank = starts.local_rank = 5

    first = starts.run()

    assert first.events == FULL
    assert first.sums == [1]
    assert starts.manifests() == []
    starts.tp_rank = starts.local_rank = 0
    starts.run()
    assert len(starts.manifests()) == 1


def _rewrite(edit_data):
    def damage(path):
        data = json.loads(path.read_text())
        edit_data(data)
        path.write_text(json.dumps(data))

    return damage


def _text(text):
    return lambda path: path.write_text(text)


@pytest.mark.parametrize(
    "damage, defect",
    [
        (_text("{"), "not JSON"),
        (_text("[]"), "not a JSON object"),
        (_rewrite(lambda d: d.update(format=99)), "format"),
        (_rewrite(lambda d: d.pop("keys")), "keys"),
        (_rewrite(lambda d: d.update(keys=[])), "keys"),
        (_rewrite(lambda d: d.update(keys="0" * 32)), "keys"),
        (_rewrite(lambda d: d["keys"].append("../../etc")), "key"),
        (_rewrite(lambda d: d.pop("identity")), "identity"),
    ],
    ids=[
        "not-json",
        "not-object",
        "format",
        "no-keys",
        "empty-keys",
        "keys-not-list",
        "bad-key",
        "no-identity",
    ],
)
def test_a_malformed_manifest_is_refused_by_name_and_replaced(starts, damage, defect):
    """Refused and logged with its path and defect; the full pass writes a good one."""
    starts.run()
    path = starts.manifest()
    damage(path)

    second = starts.run()

    assert second.events == FULL
    assert second.sums == [1]
    refusal = [row for level, row in starts.log.rows if "refused" in row]
    assert refusal, starts.log.text()
    assert str(path) in refusal[0] and defect in refusal[0]
    assert starts.manifest_data()["keys"] == sorted(ALL_KEYS)


def test_a_miss_during_a_skipped_warm_up_removes_the_manifest(starts):
    """A graph the manifest did not name: it compiles in place, the next start is full."""
    starts.run()
    extra = graph_key("decode-b1-ctx1024")

    second = starts.run(decode=DECODE_KEYS + (extra,))

    assert second.events == WARM[:2] + [("compile_in_place", extra)] + WARM[2:]
    assert extra in starts.log.text("warning")
    assert starts.manifests() == []
    third = starts.run(decode=DECODE_KEYS + (extra,))
    assert third.events == FULL
    assert starts.manifest_data()["keys"] == sorted(ALL_KEYS + (extra,))


# ------------------------------------------------------------------ the TP group vote


def test_one_cold_rank_makes_every_rank_run_the_full_pass(starts):
    """The skip needs every rank: one rank that finds its cache cold decides for all."""
    starts.run()
    starts.cold_ranks = 1

    second = starts.run()

    assert second.events == FULL
    assert second.sums == [0]
    assert f"1 of {TP_SIZE}" in starts.log.text("info")


def test_a_rank_whose_check_raises_votes_cold(starts, monkeypatch):
    """An error inside the local check is a cold vote, never an exception."""
    starts.run()

    def broken():
        raise OSError("read error")

    monkeypatch.setattr(neuron_worker, "_warm_start_package_digest", broken)
    second = starts.run()

    assert second.events == FULL
    assert second.sums == [1]
    assert "read error" in starts.log.text()


# ------------------------------------------------------------------ preconditions


def _knob_off(worker, monkeypatch):
    monkeypatch.setenv(KNOB, "0")


def _no_cache(worker, monkeypatch):
    monkeypatch.setenv("NEURON_LIBTORCH_DISABLE_COMPILE_CACHE", "1")


def _remote_cache(worker, monkeypatch):
    monkeypatch.setenv("NEURON_LIBTORCH_REMOTE_CACHE", "/mnt/shared-cache")


def _pipeline(worker, monkeypatch):
    worker.vllm_config.parallel_config.pipeline_parallel_size = 2


def _vision(worker, monkeypatch):
    worker.model_runner.vision_capture_backend = object()
    worker.model_runner.vision_neuron_config = types.SimpleNamespace()


def _no_compile_options(worker, monkeypatch):
    del worker.model_runner.compile_options


@pytest.mark.parametrize(
    "precondition, reason",
    [
        (_knob_off, KNOB),
        (_no_cache, "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"),
        (_remote_cache, "NEURON_LIBTORCH_REMOTE_CACHE"),
        (_pipeline, "pipeline"),
        (_vision, "vision"),
        (_no_compile_options, "compile options"),
    ],
    ids=[
        "knob-off",
        "cache-disabled",
        "remote-cache",
        "pipeline-parallel",
        "vision",
        "no-compile-options",
    ],
)
def test_an_unsupported_start_keeps_the_full_pass_without_a_vote(
    starts, monkeypatch, precondition, reason
):
    """Refused by name: no vote, no manifest, the extraction path of before."""
    starts.run()
    assert len(starts.manifests()) == 1
    shutil.rmtree(starts.warm_start_dir)

    second = starts.run(edit=lambda worker: precondition(worker, monkeypatch))

    assert second.sums == []
    assert second.events == (VISION_FULL if precondition is _vision else FULL)
    assert starts.manifests() == []
    assert reason in starts.log.text("info")


def test_the_knob_forces_the_full_pass_on_a_warm_root(starts, monkeypatch):
    """``VLLM_NEURON_SKIP_EXTRACTION_WHEN_CACHED=0``: the full pass, manifest or not."""
    starts.run()
    assert len(starts.manifests()) == 1
    monkeypatch.setenv(KNOB, "0")

    second = starts.run()

    assert second.events == FULL
    assert second.sums == []


def test_the_native_backend_keeps_its_path_without_a_vote(starts, monkeypatch):
    """The native route precompiles in extraction; the skip does not apply to it."""
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")

    first = starts.run()

    assert first.sums == []
    assert first.events == BASELINE["native"]
    assert starts.manifests() == []
