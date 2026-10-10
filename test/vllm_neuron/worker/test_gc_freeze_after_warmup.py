# SPDX-License-Identifier: Apache-2.0
"""A worker that will serve applies its GC policy once, after warmup.

Each TP=64 worker of the bs=64 GLM-5.3-Flash line tracks 5.2-5.4 M objects in
generation 2 after warmup, so one gen-2 pass takes 3.5-6.0 s. The rank in GC does
not submit its step and the other 63 ranks wait for it: that is the 3.8-6 s decode
stall. ``rare_gen2`` (``gc_policy.py``) raises the
gen-2 threshold so those passes stop, and runs no collection and no freeze: vLLM's
``freeze_gc_heap()`` (``gc.collect(0/1/2)`` then ``gc.freeze()``), which vLLM's GPU
worker calls at the end of its own ``compile_or_warm_up_model``, slowed every later
bs=1 decode step on TP=64. ``VLLM_NEURON_GC_POLICY`` selects ``rare_gen2``
(default), ``off`` (CPython default GC), ``freeze_rare_gen2`` (freeze, then raise) or
``freeze`` (freeze alone). The ordering cases run ``rare_gen2``; the default is
pinned on every serving path below and in ``test_gc_policy_default.py``.

The tests drive the real ``NeuronWorker.compile_or_warm_up_model`` with the
extraction, compile and warmup steps replaced by recorders, and replace the freeze,
``gc.set_threshold``, ``gc.disable`` and the GC-debug hook with recorders too: the
real calls here would change the pytest process.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import gc
import types

import pytest
import vllm.utils.gc_utils as gc_utils
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_neuron.vllm.worker import gc_policy, neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

FREEZE = "freeze_gc_heap"
#: The gen-2 threshold change of the raising policies (gen-0/1 thresholds kept).
RAISE_GEN2 = "set_threshold_gen2"
GC_DEBUG = "gc_debug"
#: ``rare_gen2``'s GC calls, in order: no freeze.
RARE_GEN2_GC = [RAISE_GEN2, GC_DEBUG]
#: ``freeze_rare_gen2``'s GC calls, in order.
FREEZE_RARE_GEN2_GC = [FREEZE, RAISE_GEN2, GC_DEBUG]
#: Every recorded GC call.
GC_CALLS = (FREEZE, RAISE_GEN2, GC_DEBUG)
POLICY_ENV = "VLLM_NEURON_GC_POLICY"
#: The steps of the main path that create the post-warmup heap. The policy must
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
    """Ordered record of the steps the worker ran, freeze calls included.

    The policy is ``rare_gen2`` unless a case sets or clears it.
    """
    record: list[str] = []
    for name in WARMUP_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(POLICY_ENV, gc_policy.RARE_GEN2)
    monkeypatch.setattr(neuron_worker, "tp_barrier", lambda: record.append("tp_barrier"))

    def freeze_gc_heap() -> None:
        record.append(FREEZE)

    t0, t1, _ = gc.get_threshold()

    def set_threshold(*thresholds) -> None:
        # Only the gen-2 threshold may change; anything else is recorded as is.
        if thresholds == (t0, t1, gc_policy.GEN2_THRESHOLD):
            record.append(RAISE_GEN2)
        else:
            record.append(f"set_threshold{thresholds}")

    # The worker may import the function at call time or at module scope; both
    # names point at the recorder, so the real gc.freeze() never runs here.
    monkeypatch.setattr(gc_utils, "freeze_gc_heap", freeze_gc_heap)
    monkeypatch.setattr(neuron_worker, "freeze_gc_heap", freeze_gc_heap, raising=False)
    monkeypatch.setattr(
        gc_utils, "maybe_attach_gc_debug_callback", lambda: record.append(GC_DEBUG)
    )
    monkeypatch.setattr(gc, "set_threshold", set_threshold)
    monkeypatch.setattr(gc, "disable", lambda: record.append("gc_disable"))
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
def test_main_path_applies_the_gc_policy_once_after_the_warmup_work(
    events, monkeypatch, skip_env, warmup_ran
):
    """Raise gen-2 (no freeze), exactly once, after every warmup step that ran."""
    if skip_env is not None:
        monkeypatch.setenv(skip_env, "1")

    result = NeuronWorker.compile_or_warm_up_model(_worker(events))

    assert isinstance(result, CompilationTimes)
    assert [e for e in events if e in GC_CALLS or e.startswith(("set_", "gc_"))] == (
        RARE_GEN2_GC
    )
    assert [e for e in events if e.startswith("warmup_")] == warmup_ran
    policy_at = events.index(RARE_GEN2_GC[0])
    for step in WARMUP_WORK:
        if step in events:
            assert events.index(step) < policy_at, (step, events)
    assert events[policy_at:] == RARE_GEN2_GC, events


@pytest.mark.parametrize("missing", ["model_runner", "model"])
def test_no_gc_change_when_no_model_is_loaded(events, missing):
    """No model, nothing to serve: the early return leaves the GC alone."""
    worker = _worker(events)
    if missing == "model_runner":
        worker.model_runner = None
    else:
        worker.model_runner.model = None

    result = NeuronWorker.compile_or_warm_up_model(worker)

    assert result == CompilationTimes(language_model=0.0, encoder=0.0)
    assert events == []


def _synthetic_model():
    from vllm_neuron.model.synthetic import SyntheticNeuronModel

    return SyntheticNeuronModel.__new__(SyntheticNeuronModel)


def test_synthetic_model_applies_the_gc_policy_once_and_skips_warmup(events):
    """The synthetic DI-test model serves without warmup, so it gets the policy too."""
    NeuronWorker.compile_or_warm_up_model(_worker(events, model=_synthetic_model()))

    assert events == RARE_GEN2_GC


def test_cpu_eager_mode_applies_the_gc_policy_once_and_skips_warmup(events):
    """CPU eager mode serves without warmup, so it gets the policy too."""
    NeuronWorker.compile_or_warm_up_model(_worker(events, enforce_eager=True))

    assert events == RARE_GEN2_GC


@pytest.mark.parametrize("path", ["main", "synthetic", "cpu-eager"])
def test_the_default_raises_gen2_without_a_freeze_on_every_serving_path(
    events, monkeypatch, path
):
    """Knob unset: ``rare_gen2`` on every serving path, no freeze."""
    monkeypatch.delenv(POLICY_ENV)
    worker = _worker(
        events,
        model=_synthetic_model() if path == "synthetic" else None,
        enforce_eager=path == "cpu-eager",
    )

    NeuronWorker.compile_or_warm_up_model(worker)

    assert [e for e in events if e in GC_CALLS or e.startswith(("set_", "gc_"))] == (
        RARE_GEN2_GC
    )


@pytest.mark.parametrize("path", ["main", "synthetic", "cpu-eager"])
def test_kill_switch_keeps_cpython_default_gc_on_every_serving_path(
    events, monkeypatch, path
):
    """``VLLM_NEURON_GC_POLICY=off``: no freeze, no threshold change, gc stays on."""
    monkeypatch.setenv(POLICY_ENV, "off")
    worker = _worker(
        events,
        model=_synthetic_model() if path == "synthetic" else None,
        enforce_eager=path == "cpu-eager",
    )

    NeuronWorker.compile_or_warm_up_model(worker)

    assert FREEZE not in events
    assert [e for e in events if e.startswith(("set_", "gc_"))] == [GC_DEBUG]


def test_freeze_rare_gen2_policy_freezes_then_raises_once(events, monkeypatch):
    """``VLLM_NEURON_GC_POLICY=freeze_rare_gen2``: an earlier default, for A/B runs."""
    monkeypatch.setenv(POLICY_ENV, "freeze_rare_gen2")

    NeuronWorker.compile_or_warm_up_model(_worker(events))

    assert [e for e in events if e in GC_CALLS or e.startswith(("set_", "gc_"))] == (
        FREEZE_RARE_GEN2_GC
    )


def test_freeze_policy_freezes_once_and_keeps_the_thresholds(events, monkeypatch):
    """``VLLM_NEURON_GC_POLICY=freeze``: the e9aa679 behaviour, for A/B runs."""
    monkeypatch.setenv(POLICY_ENV, "freeze")

    NeuronWorker.compile_or_warm_up_model(_worker(events))

    assert [e for e in events if e == FREEZE or e.startswith(("set_", "gc_"))] == [
        FREEZE,
        GC_DEBUG,
    ]
