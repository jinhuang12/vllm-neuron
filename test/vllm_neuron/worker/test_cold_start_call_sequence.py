# SPDX-License-Identifier: Apache-2.0
"""A start on a cold compile cache extracts, compiles and warms up in one fixed order.

The tests drive the real ``NeuronWorker.compile_or_warm_up_model`` on an empty compile
cache. Graph extraction, the parallel compile and the warm-up steps are replaced by
fakes that act on a real cache directory the way the real steps do:

* extraction looks each graph up in the cache, then writes its ``graph.hlo``;
* the parallel compile completes every entry that has a ``graph.hlo`` only;
* a warm-up step looks each of its graphs up and, on a miss, compiles it in place.

Each case compares the recorded calls, with their arguments, with ``BASELINE``: the
sequence these tests recorded on the worker before the warm-start skip existed. So
a change to the order, the arguments or the count of the cold-start steps fails
here. The cache directory must hold the served entries and nothing else after the
start, so that a count of its entries sees only graphs.

The TP group's one-integer sum, which only the warm-start decision calls on this
path, is recorded on its own and kept out of the comparison.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import hashlib
import json
import os
import types
from pathlib import Path

import pytest
from libtorch_neuronx_lite.compile import cache
from transformers import PretrainedConfig

from vllm_neuron.vllm.worker import gc_policy, neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker


def graph_key(name: str) -> str:
    """A 32-hex compile cache key, as the cache names its entries."""
    return hashlib.md5(name.encode()).hexdigest()


PREFILL_KEYS = (graph_key("prefill-1024-kv1024"), graph_key("prefill-1024-kv2048"))
DECODE_KEYS = (graph_key("decode-b1-ctx2048"), graph_key("decode-b1-ctx4096"))
#: Env levers that change which warm-up phases run, the backend, or the cache; cleared
#: so the host env of the test run cannot pick the path.
START_ENV = (
    "VLLM_NEURON_CPU_COMPILE",
    "VLLM_NEURON_BACKEND",
    "VLLM_NEURON_SKIP_PREFILL_WARMUP",
    "VLLM_NEURON_SKIP_DECODE_WARMUP",
    "VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP",
    "VLLM_NEURON_SKIP_ENCODER_WARMUP",
    "VLLM_NEURON_SKIP_EXTRACTION_WHEN_CACHED",
    "VLLM_CACHE_ROOT",
    "NEURON_LIBTORCH_ASSERT_CACHE_HIT",
    "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE",
    "NEURON_LIBTORCH_REMOTE_CACHE",
)


class CompileCache:
    """A compile cache root on disk, with the entry layout the cache checks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dir = root / "neuron" / "compile_cache"
        self.dir.mkdir(parents=True, exist_ok=True)

    def entry(self, key: str) -> Path:
        return self.dir / key

    def capture(self, key: str) -> None:
        """What extraction leaves: the graph's HLO, no NEFF yet."""
        self.entry(key).mkdir(parents=True, exist_ok=True)
        (self.entry(key) / "graph.hlo").write_bytes(b"hlo " + key.encode())

    def complete(self, key: str) -> None:
        """What a finished compile leaves: HLO, NEFF, metadata, then the marker."""
        self.capture(key)
        entry = self.entry(key)
        (entry / f"graph_{key}.neff").write_bytes(b"neff " + key.encode())
        (entry / ".artifact_metadata_v0.json").write_text(json.dumps({"cache_key": key}))
        (entry / ".compilation_complete").write_text("done")

    def is_complete(self, key: str) -> bool:
        return cache._is_cache_complete(str(self.entry(key)))

    def entries(self) -> list[str]:
        return sorted(os.listdir(self.dir))


class Steps:
    """The start's work, faked on a real cache directory; ``events`` keeps the calls."""

    def __init__(self, compile_cache: CompileCache, prefill=PREFILL_KEYS, decode=DECODE_KEYS):
        self.cache = compile_cache
        self.prefill = tuple(prefill)
        self.decode = tuple(decode)
        self.events: list[tuple] = []
        self.sums: list[int] = []
        self.lookups: list[tuple[str, bool]] = []

    def look_up(self, key: str):
        # Through the module attribute, as the compile backends call it.
        found = cache.get_local(key, str(self.cache.dir))
        return found

    def extract_graphs(self, **kwargs) -> None:
        self.events.append(("extract_graphs", dict(sorted(kwargs.items()))))
        keys = (() if kwargs["skip_prefill"] else self.prefill) + (
            () if kwargs["skip_decode"] else self.decode
        )
        for key in keys:
            if self.look_up(key) is None:
                self.cache.capture(key)

    def parallel_compile(self) -> None:
        self.events.append(("parallel_compile",))
        for key in os.listdir(self.cache.dir):
            if (self.cache.entry(key) / "graph.hlo").exists() and not self.cache.is_complete(
                key
            ):
                self.cache.complete(key)

    def _warm_up(self, phase: str, keys) -> None:
        self.events.append((f"warmup_{phase}",))
        for key in keys:
            if self.look_up(key) is None:
                self.events.append(("compile_in_place", key))
                self.cache.complete(key)

    def warmup_prefill(self) -> None:
        self._warm_up("prefill", self.prefill)

    def warmup_decode(self) -> None:
        self._warm_up("decode", self.decode)

    def warmup_vision(self) -> None:
        self.events.append(("warmup_vision",))

    def tp_barrier(self) -> None:
        self.events.append(("tp_barrier",))

    def tp_sum_int(self, value: int, timeout=None) -> int:
        self.sums.append(value)
        return value


def fake_get_local(hash_key: str, local_cache_dir: str):
    """The cache's local lookup: an entry for a complete key, None for a miss."""
    if cache._is_cache_complete(os.path.join(local_cache_dir, hash_key)):
        return types.SimpleNamespace(hash_key=hash_key)
    return None


def _config(name: str, **fields):
    return types.SimpleNamespace(compute_hash=lambda: f"{name}-hash", **fields)


def make_vllm_config(*, kv_transfer_config=None, max_num_seqs: int = 1):
    """The configs the start reads, with the hashes vLLM's configs provide."""
    return types.SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        model_config=_config(
            "model",
            enforce_eager=False,
            model="test-model",
            max_model_len=4096,
            hf_config=PretrainedConfig(num_hidden_layers=2),
        ),
        cache_config=_config("cache"),
        parallel_config=_config("parallel", pipeline_parallel_size=1),
        scheduler_config=types.SimpleNamespace(
            compute_hash=lambda: "scheduler-hash",
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=1024,
            async_scheduling=True,
        ),
        additional_config={
            "neuron_config": {"num_batched_tokens_buckets": [1024]},
            "neuron_profiler": {"output_dir": "/runs/1/nrt_profile"},
        },
    )


def make_worker(steps: Steps, *, capture_backend=True, pooling=False, tensor_capture=False):
    """A loaded worker whose extraction, compile and warm-up steps are ``steps``."""
    worker = NeuronWorker.__new__(NeuronWorker)
    worker.rank = 0
    worker.local_rank = 0
    worker.model_runner = types.SimpleNamespace(
        model=object(),
        mm_encoder_only=False,
        mm_language_model_only=False,
        is_pooling_model=pooling,
        vision_neuron_config=None,
        capture_backend_model=object() if capture_backend else None,
        vision_capture_backend=None,
        drafter=None,
        max_model_len=4096,
        kv_cache_config=types.SimpleNamespace(
            num_blocks=100, kv_cache_tensors=[types.SimpleNamespace(size=100 * 4096)]
        ),
        compile_options={"alias_meta_to_neuron": True, "compiler_args": ["-O1"]},
        neuron_config=types.SimpleNamespace(
            tensor_capture=tensor_capture,
            num_batched_tokens_buckets=[1024],
            kv_segment_size_buckets=[1024, 2048],
            num_seqs_buckets=[1],
            decode_context_length_buckets=[2048],
        ),
        parallel_compile=steps.parallel_compile,
        enable_capture=lambda: steps.events.append(("enable_capture",)),
        maybe_register_ec_cache=lambda: steps.events.append(("register_ec_cache",)),
    )
    worker.vllm_config = make_vllm_config()
    worker._startup_start_time = 0.0
    worker._extract_graphs = steps.extract_graphs
    worker._warmup_prefill = steps.warmup_prefill
    worker._warmup_decode = steps.warmup_decode
    worker._warmup_vision_encoder = steps.warmup_vision
    return worker


@pytest.fixture
def start(tmp_path, monkeypatch):
    """One empty compile cache root, the TP group faked as rank 0 of 64."""
    for name in START_ENV:
        monkeypatch.delenv(name, raising=False)
    compile_cache = CompileCache(tmp_path / "root")
    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(compile_cache.root))
    monkeypatch.setattr(cache, "get_local", fake_get_local)
    steps = Steps(compile_cache)
    monkeypatch.setattr(neuron_worker, "tp_barrier", steps.tp_barrier)
    monkeypatch.setattr(neuron_worker, "tp_sum_int", steps.tp_sum_int)
    monkeypatch.setattr(
        neuron_worker,
        "get_tp_group",
        lambda: types.SimpleNamespace(rank_in_group=0, world_size=64),
    )
    monkeypatch.setattr(
        gc_policy,
        "apply_post_warmup_gc_policy",
        lambda: steps.events.append(("gc_policy",)),
    )
    return steps


def _extract(skip_prefill=False, skip_decode=False, native=False):
    return (
        "extract_graphs",
        {
            "native_compile_only": native,
            "skip_decode": skip_decode,
            "skip_prefill": skip_prefill,
            "skip_vision": True,
        },
    )


_TAIL = [("register_ec_cache",), ("gc_policy",)]
#: The calls each cold start made on the worker before the warm-start skip existed
#: (recorded there by these tests).
BASELINE = {
    "both": [
        _extract(),
        ("tp_barrier",),
        ("parallel_compile",),
        ("tp_barrier",),
        ("warmup_prefill",),
        ("warmup_decode",),
        *_TAIL,
    ],
    "decode-only": [
        _extract(skip_prefill=True),
        ("tp_barrier",),
        ("parallel_compile",),
        ("tp_barrier",),
        ("warmup_decode",),
        *_TAIL,
    ],
    "prefill-only": [
        _extract(skip_decode=True),
        ("tp_barrier",),
        ("parallel_compile",),
        ("tp_barrier",),
        ("warmup_prefill",),
        *_TAIL,
    ],
    "kv-consumer": [
        _extract(skip_prefill=True),
        ("tp_barrier",),
        ("parallel_compile",),
        ("tp_barrier",),
        ("warmup_decode",),
        *_TAIL,
    ],
    "pooling": [
        _extract(skip_decode=True),
        ("tp_barrier",),
        ("parallel_compile",),
        ("tp_barrier",),
        ("warmup_prefill",),
        *_TAIL,
    ],
    "tensor-capture": [
        _extract(),
        ("tp_barrier",),
        ("parallel_compile",),
        ("tp_barrier",),
        ("warmup_prefill",),
        ("warmup_decode",),
        ("enable_capture",),
        *_TAIL,
    ],
    "native": [
        _extract(native=True),
        ("tp_barrier",),
        ("tp_barrier",),
        ("warmup_prefill",),
        *[("compile_in_place", key) for key in PREFILL_KEYS],
        ("warmup_decode",),
        *[("compile_in_place", key) for key in DECODE_KEYS],
        *_TAIL,
    ],
    "no-capture-backend": [
        ("warmup_prefill",),
        *[("compile_in_place", key) for key in PREFILL_KEYS],
        ("warmup_decode",),
        *[("compile_in_place", key) for key in DECODE_KEYS],
        *_TAIL,
    ],
}
#: The entries each case leaves in the compile cache directory.
SERVED = {
    "both": PREFILL_KEYS + DECODE_KEYS,
    "decode-only": DECODE_KEYS,
    "prefill-only": PREFILL_KEYS,
    "kv-consumer": DECODE_KEYS,
    "pooling": PREFILL_KEYS,
    "tensor-capture": PREFILL_KEYS + DECODE_KEYS,
    "native": PREFILL_KEYS + DECODE_KEYS,
    "no-capture-backend": PREFILL_KEYS + DECODE_KEYS,
}


def _case_worker(case: str, steps: Steps, monkeypatch):
    if case == "decode-only":
        monkeypatch.setenv("VLLM_NEURON_SKIP_PREFILL_WARMUP", "1")
    elif case == "prefill-only":
        monkeypatch.setenv("VLLM_NEURON_SKIP_DECODE_WARMUP", "1")
    elif case == "native":
        monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")
    worker = make_worker(
        steps,
        capture_backend=case != "no-capture-backend",
        pooling=case == "pooling",
        tensor_capture=case == "tensor-capture",
    )
    if case == "kv-consumer":
        worker.vllm_config.kv_transfer_config = types.SimpleNamespace(
            is_kv_consumer=True,
            is_kv_producer=False,
            kv_role="kv_consumer",
            compute_hash=lambda: "kv-transfer-hash",
        )
    return worker


@pytest.mark.parametrize("case", sorted(BASELINE))
def test_a_cold_start_runs_the_recorded_call_sequence(start, monkeypatch, case):
    """Extraction, barriers, compile and warm-ups: the recorded order and arguments."""
    worker = _case_worker(case, start, monkeypatch)

    NeuronWorker.compile_or_warm_up_model(worker)

    assert start.events == BASELINE[case]
    assert start.cache.entries() == sorted(SERVED[case])
    assert all(start.cache.is_complete(key) for key in SERVED[case])
    # The decision is at most one sum of one integer, before any other step.
    assert start.sums in ([], [1])


@pytest.mark.parametrize("case", sorted(BASELINE))
def test_the_cold_start_leaves_the_cache_lookup_as_it_found_it(start, monkeypatch, case):
    """After the start, the cache's lookup is the function it was before."""
    worker = _case_worker(case, start, monkeypatch)

    NeuronWorker.compile_or_warm_up_model(worker)

    assert cache.get_local is fake_get_local
