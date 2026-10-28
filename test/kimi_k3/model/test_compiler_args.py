# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 compiler arguments per graph: the prefill graph gets more splitter repeat runs.

The prefill graph's list extends the one hlo2tensorizer entry of the base list, and every
other graph keeps the base list. The worker's trace jobs carry their graph type, the two
Qwen policies keep the base list for prefill, and the prefill trace job, warmup and
in-process capture resolve one list. No model is built or loaded. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import types

import torch
from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from libtorch_neuronx_lite.compile.compiler_args import apply_scoped_compiler_args

from vllm_neuron.model.kimi_k3.model import KimiK3ForConditionalGeneration

FLAG = "--internal-hlo2tensorizer-options="
OPTION = "--partitioner-max-disj-runs=16"
ENTRY = f"{FLAG}--modular-flow-mac-threshold=10"
# The runner's base list at the served form, which sets no optimization level.
BASE = [
    "--auto-cast=none",
    "--verbose=35",
    "-O2",
    ENTRY,
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]
BUCKETS = [(8192, 8192), (512, 0)]


def _k3():
    """A Kimi K3 model object with no state, for its policy."""
    return object.__new__(KimiK3ForConditionalGeneration)


def _worker(runner):
    """A worker over ``runner``, on the first model runner."""
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    worker = object.__new__(NeuronWorker)
    worker.__dict__.update(
        model_runner=runner, vllm_config=types.SimpleNamespace(use_v2_model_runner=False)
    )
    return worker


def _synthetic(*args, **kwargs):
    """Synthetic inputs: the callables here read none of them."""
    return {"attn_metadata": {}}


def test_prefill_extends_the_one_hlo2tensorizer_entry():
    """The prefill list is the base list with its one entry extended by the option."""
    base = list(BASE)
    args = _k3().compiler_args_for("prefill", None, base)
    assert base == BASE
    assert [a for a in args if a.startswith(FLAG)] == [f"{ENTRY} {OPTION}"]
    i = BASE.index(ENTRY)
    assert len(args) == len(BASE)
    assert args[:i] + args[i + 1 :] == BASE[:i] + BASE[i + 1 :]


def test_other_graph_types_keep_the_base():
    """Every other graph type, and no type, gets None, so the base list applies."""
    for graph_type in ("decode", "spec_verify", "draft", "encoder", None):
        assert _k3().compiler_args_for(graph_type, None, list(BASE)) is None, graph_type


def test_worker_trace_jobs_carry_their_graph_type():
    """Prefill trace jobs resolve "prefill" and decode trace jobs "decode", on both routes."""
    runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        drafter=None,
        speculative_config=None,
        model=object(),
        capture_backend_model=object(),
        _build_prefill_synthetic_inputs=_synthetic,
        _build_decode_synthetic_inputs=_synthetic,
        resolve_graph_compiler_args=lambda graph_type: [graph_type],
    )
    worker = _worker(runner)
    targets = [(32, 32768), (1, 2048)]
    for native in (True, False):
        prefill = worker._build_prefill_trace_jobs(BUCKETS, native_compile_only=native)
        decode = worker._build_decode_trace_jobs(targets, native_compile_only=native)
        assert [job[2] for job in prefill] == [["prefill"]] * len(BUCKETS), native
        assert [job[2] for job in decode] == [["decode"]] * 2, native


def test_qwen_policies_keep_the_prefill_base():
    """The Qwen3-ASR and Qwen3-VL policies leave the prefill graph on the base list."""
    from vllm_neuron.model.qwen3_asr.model_bf16 import Qwen3ASRForConditionalGeneration
    from vllm_neuron.model.qwen3_vl.model_common import Qwen3VLGenerationMixin

    for cls in (Qwen3ASRForConditionalGeneration, Qwen3VLGenerationMixin):
        args = cls.compiler_args_for(object.__new__(cls), "prefill", None, list(BASE))
        assert (BASE if args is None else args) == BASE, cls.__name__


def test_trace_job_warmup_and_capture_resolve_one_prefill_list():
    """The prefill trace job, warmup and in-process capture give the backend one list."""
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    seen = []

    def compiled(**kwargs):
        """Record the list a backend reads from the scope before its cache hash."""
        seen.append(apply_scoped_compiler_args({"compiler_args": list(BASE)})["compiler_args"])

    def capture(**kwargs):
        """Record the list, then end the capture as the capture backend does."""
        compiled(**kwargs)
        raise CaptureComplete()

    runner = object.__new__(NeuronModelRunner)
    runner.__dict__.update(
        device=torch.device("cpu"),
        drafter=None,
        speculative_config=None,
        compile_options={"compiler_args": list(BASE)},
        _per_graph_compiler_args=True,
        _compiler_args_policy=_k3().compiler_args_for,
        model=compiled,
        capture_backend_model=capture,
        kv_cache_config=object(),
        _tensor_replacer=None,
        use_async_scheduling=False,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(model="kimi-k3")),
        _build_prefill_synthetic_inputs=_synthetic,
    )
    jobs = _worker(runner)._build_prefill_trace_jobs(BUCKETS, native_compile_only=True)
    for bucket, segment in BUCKETS:
        runner.warmup_prefill(bucket, segment)
        runner.extract_prefill_graphs(bucket, segment)
    want = _k3().compiler_args_for("prefill", None, list(BASE))
    assert want != BASE
    assert [job[2] for job in jobs] == [want] * len(BUCKETS)
    assert seen == [want] * (2 * len(BUCKETS))
