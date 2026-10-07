# SPDX-License-Identifier: Apache-2.0
"""A worker that will serve freezes its GC heap once, after warmup.

Each TP=64 worker of the bs=64 GLM-5.3-Flash line tracks 5.2-5.4 M objects in
generation 2 after warmup, so one gen-2 pass takes 3.5-6.0 s. The rank in GC does
not submit its step and the other 63 ranks wait for it: that is the 3.8-6 s decode
stall (DECODE_BREAKDOWN_v2.md 5.2). ``vllm.utils.gc_utils.freeze_gc_heap``
(``gc.collect(0/1/2)`` then ``gc.freeze()``) moves that static heap out of the
collector's reach; vLLM's GPU worker calls it at the end of its own
``compile_or_warm_up_model``.

The tests drive the real ``NeuronWorker.compile_or_warm_up_model`` with the
extraction, compile and warmup steps replaced by recorders, and replace the freeze
with a recorder too: a real ``gc.freeze()`` here would freeze the pytest process.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import types

import pytest
import vllm.utils.gc_utils as gc_utils
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

FREEZE = "freeze_gc_heap"
#: The steps of the main path that create the post-warmup heap. The freeze must
#: come after every one of them that runs.
WARMUP_WORK = (
    "extract_graphs",
    "parallel_compile",
    "warmup_prefill",
    "warmup_decode",
)
#: Env levers that change which warmup phases run; cleared so the host env of the
#: test run cannot pick the path.
WARMUP_ENV = (
    "VLLM_NEURON_CPU_COMPILE",
    "VLLM_NEURON_BACKEND",
    "VLLM_NEURON_SKIP_PREFILL_WARMUP",
    "VLLM_NEURON_SKIP_DECODE_WARMUP",
    "VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP",
    "VLLM_NEURON_SKIP_ENCODER_WARMUP",
)


@pytest.fixture
def events(monkeypatch):
    """Ordered record of the steps the worker ran, freeze calls included."""
    record: list[str] = []
    for name in WARMUP_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(neuron_worker, "tp_barrier", lambda: record.append("tp_barrier"))

    def freeze_gc_heap() -> None:
        record.append(FREEZE)

    # The worker may import the function at call time or at module scope; both
    # names point at the recorder, so the real gc.freeze() never runs here.
    monkeypatch.setattr(gc_utils, "freeze_gc_heap", freeze_gc_heap)
    monkeypatch.setattr(neuron_worker, "freeze_gc_heap", freeze_gc_heap, raising=False)
    return record


def _worker(record: list[str], *, model=None, enforce_eager: bool = False):
    """A loaded worker whose extraction, compile and warmup steps only record."""
    worker = NeuronWorker.__new__(NeuronWorker)
    worker.model_runner = types.SimpleNamespace(
        model=object() if model is None else model,
        mm_encoder_only=False,
        mm_language_model_only=False,
        is_pooling_model=False,
        vision_neuron_config=None,
        capture_backend_model=object(),
        vision_capture_backend=None,
        neuron_config=types.SimpleNamespace(
            tensor_capture=False,
            num_batched_tokens_buckets=[1024],
            num_seqs_buckets=[1, 64],
        ),
        parallel_compile=lambda: record.append("parallel_compile"),
        maybe_register_ec_cache=lambda: record.append("register_ec_cache"),
    )
    worker.vllm_config = types.SimpleNamespace(
        kv_transfer_config=None,
        model_config=types.SimpleNamespace(
            enforce_eager=enforce_eager, model="test-model"
        ),
    )
    worker._startup_start_time = 0.0
    worker._extract_graphs = lambda **kwargs: record.append("extract_graphs")
    worker._warmup_prefill = lambda: record.append("warmup_prefill")
    worker._warmup_decode = lambda: record.append("warmup_decode")
    worker._warmup_vision_encoder = lambda: record.append("warmup_vision")
    return worker


@pytest.mark.parametrize(
    "skip_env, warmup_ran",
    [
        (None, ["warmup_prefill", "warmup_decode"]),
        ("VLLM_NEURON_SKIP_PREFILL_WARMUP", ["warmup_decode"]),
        ("VLLM_NEURON_SKIP_DECODE_WARMUP", ["warmup_prefill"]),
        ("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP", []),
    ],
    ids=["both", "decode-only", "prefill-only", "no-lm-warmup"],
)
def test_main_path_freezes_once_after_the_warmup_work(
    events, monkeypatch, skip_env, warmup_ran
):
    """The heap is frozen exactly once, after every warmup step that ran."""
    if skip_env is not None:
        monkeypatch.setenv(skip_env, "1")

    result = NeuronWorker.compile_or_warm_up_model(_worker(events))

    assert isinstance(result, CompilationTimes)
    assert events.count(FREEZE) == 1
    assert [e for e in events if e.startswith("warmup_")] == warmup_ran
    freeze_at = events.index(FREEZE)
    for step in WARMUP_WORK:
        if step in events:
            assert events.index(step) < freeze_at, (step, events)


@pytest.mark.parametrize("missing", ["model_runner", "model"])
def test_no_freeze_when_no_model_is_loaded(events, missing):
    """No model, nothing to serve: the early return leaves the heap alone."""
    worker = _worker(events)
    if missing == "model_runner":
        worker.model_runner = None
    else:
        worker.model_runner.model = None

    result = NeuronWorker.compile_or_warm_up_model(worker)

    assert result == CompilationTimes(language_model=0.0, encoder=0.0)
    assert events == []


def test_synthetic_model_freezes_once_and_skips_warmup(events):
    """The synthetic DI-test model serves without warmup, so it freezes too."""
    from vllm_neuron.model.synthetic import SyntheticNeuronModel

    model = SyntheticNeuronModel.__new__(SyntheticNeuronModel)

    NeuronWorker.compile_or_warm_up_model(_worker(events, model=model))

    assert events == [FREEZE]


def test_cpu_eager_mode_freezes_once_and_skips_warmup(events):
    """CPU eager mode serves without warmup, so it freezes too."""
    NeuronWorker.compile_or_warm_up_model(_worker(events, enforce_eager=True))

    assert events == [FREEZE]
