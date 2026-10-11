# SPDX-License-Identifier: Apache-2.0
"""Graph extraction and the warm-up share one Dynamo trace per graph.

The runner's served wrapper traces each graph in extraction, gives a copy of it to the
capture backend, and keeps the graph for its first call outside extraction, which hands
it to the served backend. The toy-model tests pin that contract with stand-in backends
registered under the production names. The tiny-root test drives the runner's own
extraction and warm-up methods over the tiny GLM-5.3 root on CPU, and compares the trace
count, the cache keys, the HLO and the outputs with the start that keeps a wrapper each.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
        test/vllm_neuron/worker/test_trace_once.py
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch._dynamo.backends import registry
from torch._dynamo.utils import counters
from torch._guards import TracingContext

from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete

SERVED = "neuron_libtorch"
CAPTURE = "neuron_libtorch_graph_capture"
LOGGER = "vllm_neuron.vllm.worker.trace_once"


class _Toy(torch.nn.Module):
    """Two layers: enough nodes for a graph edit to show in the graph's text."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


class _Backends:
    """Stand-ins for the served and the capture backend.

    The served stand-in runs the graph it gets without autograd, as the device's
    executor does. Each call records the graph text it was given and whether Dynamo was
    tracing at that time.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.capture_edit = None
        self.capture_raises = None
        self.bail_raises = CaptureComplete

    def served(self, gm, example_inputs, **kwargs):
        self.calls.append(self._record("served", gm, kwargs))

        def run(*args):
            with torch.no_grad():
                return gm.forward(*args)

        return run

    def capture(self, gm, example_inputs, **kwargs):
        self.calls.append(self._record("capture", gm, kwargs))
        if self.capture_raises is not None:
            raise self.capture_raises
        if self.capture_edit is not None:
            self.capture_edit(gm)
        raises = self.bail_raises

        def bail(*args, **kw):
            raise raises()

        return bail

    @staticmethod
    def _record(name, gm, kwargs) -> dict:
        return {
            "name": name,
            "gm": gm,
            "graph": str(gm.graph),
            "kwargs": kwargs,
            "in_trace": TracingContext.try_get() is not None,
        }

    def names(self) -> list[str]:
        return [call["name"] for call in self.calls]


def _register(monkeypatch, served, capture) -> None:
    """Hold the production backend names with the given compilers for one test."""
    for name, fn in ((SERVED, served), (CAPTURE, capture)):
        monkeypatch.setitem(registry._BACKENDS, name, None)
        monkeypatch.setitem(registry._COMPILER_FNS, name, fn)


@pytest.fixture
def backends(monkeypatch):
    stand_ins = _Backends()
    _register(monkeypatch, stand_ins.served, stand_ins.capture)
    monkeypatch.setenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE", "1")
    monkeypatch.delenv("VLLM_NEURON_BACKEND", raising=False)
    torch._dynamo.reset()
    counters.clear()
    yield stand_ins
    torch._dynamo.reset()


@pytest.fixture
def lines():
    """The records the trace-once module logs, read from its own logger."""
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Keep(level=logging.DEBUG)
    log = logging.getLogger(LOGGER)
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    yield records
    log.removeHandler(handler)
    log.setLevel(old_level)


def _text(records) -> list[str]:
    return [record.getMessage() for record in records]


def _runner_shell(module, *, tensor_replacement=None):
    """A runner carrying only what the wrapping reads."""
    return SimpleNamespace(
        model=module,
        neuron_config=SimpleNamespace(tensor_replacement=tensor_replacement),
        compile_options={"compiler_args": ["--auto-cast=none"]},
    )


def _wrap(runner, *, capture: bool = True) -> None:
    """Wrap the runner's model as ``NeuronModelRunner.load_model`` does."""
    from vllm_neuron.vllm.worker.trace_once import compile_target_model

    runner.model, runner.capture_backend_model = compile_target_model(
        runner.model,
        capture=capture,
        fullgraph=True,
        options=runner.compile_options,
        tensor_replacement=bool(runner.neuron_config.tensor_replacement),
    )


def _extract(runner, **kwargs) -> None:
    with pytest.raises(CaptureComplete):
        runner.capture_backend_model(**kwargs)


def test_extraction_traces_the_graph_that_the_warm_up_serves(backends):
    module = _Toy()
    x = torch.randn(4, 8)
    eager = module(x)
    runner = _runner_shell(module)
    _wrap(runner)

    _extract(runner, x=x)
    assert backends.names() == ["capture"]

    out = runner.model(x=x)
    assert backends.names() == ["capture", "served"]
    assert counters["stats"]["unique_graphs"] == 1
    torch.testing.assert_close(out, eager)


def test_the_served_backend_gets_the_traced_graph_after_extraction(backends):
    runner = _runner_shell(_Toy())
    _wrap(runner)
    _extract(runner, x=torch.randn(4, 8))
    runner.model(x=torch.randn(4, 8))

    capture, served = backends.calls
    assert capture["gm"] is not served["gm"]
    assert capture["graph"] == served["graph"]
    # The NEFF load belongs to the warm-up: the served backend runs at the first call
    # outside extraction, not inside the extraction trace.
    assert capture["in_trace"] is True
    assert served["in_trace"] is False
    assert capture["kwargs"] == served["kwargs"] == {"options": runner.compile_options}


def test_an_edit_by_the_capture_backend_does_not_reach_the_served_graph(backends):
    def drop_the_relu(gm):
        for node in list(gm.graph.nodes):
            if node.op == "call_function" and node.target is torch.relu:
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
        gm.recompile()

    backends.capture_edit = drop_the_relu
    runner = _runner_shell(_Toy())
    _wrap(runner)
    _extract(runner, x=torch.randn(4, 8))
    runner.model(x=torch.randn(4, 8))

    capture, served = backends.calls
    assert "relu" in capture["graph"]
    assert str(served["gm"].graph) == capture["graph"]


def test_a_graph_first_traced_outside_extraction_compiles_in_its_trace(backends, lines):
    runner = _runner_shell(_Toy())
    _wrap(runner)
    runner.model(x=torch.randn(4, 8))

    assert backends.names() == ["served"]
    assert backends.calls[0]["in_trace"] is True
    assert not [m for m in _text(lines) if "outside graph extraction" in m]


def test_the_served_graph_runs_under_fail_on_recompile(backends):
    module = _Toy()
    runner = _runner_shell(module)
    _wrap(runner)
    _extract(runner, x=torch.randn(4, 8))
    runner.model(x=torch.randn(4, 8))

    x = torch.randn(4, 8)
    with torch.compiler.set_stance("fail_on_recompile"):
        out = runner.model(x=x)
    torch.testing.assert_close(out, module(x))
    assert backends.names() == ["capture", "served"]


def test_a_failed_capture_leaves_extraction(backends):
    backends.capture_raises = RuntimeError("capture refused this graph")
    runner = _runner_shell(_Toy())
    _wrap(runner)
    with pytest.raises(Exception, match="capture refused this graph"):
        runner.capture_backend_model(x=torch.randn(4, 8))

    backends.capture_raises = None
    runner.model(x=torch.randn(4, 8))
    assert backends.names() == ["capture", "served"]
    assert backends.calls[-1]["in_trace"] is True


def test_extraction_refuses_a_graph_that_ran_instead_of_being_captured(backends):
    runner = _runner_shell(_Toy())
    _wrap(runner)
    x = torch.randn(4, 8)
    runner.model(x=x)

    with pytest.raises(RuntimeError, match="graph extraction ran a compiled graph"):
        runner.capture_backend_model(x=x)
    assert backends.names() == ["served"]


def test_a_second_trace_while_extraction_graphs_wait_is_logged(backends, lines):
    runner = _runner_shell(_Toy())
    _wrap(runner)
    _extract(runner, x=torch.randn(4, 8))
    # A new shape: the frame's guards fail, so Dynamo traces again.
    runner.model(x=torch.randn(2, 8))

    assert backends.names() == ["capture", "served"]
    assert backends.calls[-1]["in_trace"] is True
    traced_again = [m for m in _text(lines) if "outside graph extraction" in m]
    assert traced_again == [
        "Dynamo traced a graph outside graph extraction while 1 graph(s) that "
        "extraction traced are not yet used"
    ]


def test_rank_zero_names_the_shared_trace_and_each_reuse(backends, lines):
    runner = _runner_shell(_Toy())
    _wrap(runner)
    _extract(runner, x=torch.randn(4, 8))
    _extract(runner, x=torch.randn(2, 8))
    runner.model(x=torch.randn(4, 8))
    runner.model(x=torch.randn(2, 8))
    runner.model(x=torch.randn(2, 8))

    text = _text(lines)
    assert "Graph extraction and the warm-up share one Dynamo trace per graph" in text
    assert [m for m in text if "uses the trace of graph extraction" in m] == [
        "Warm-up uses the trace of graph extraction (1 of 2 graphs)",
        "Warm-up uses the trace of graph extraction (2 of 2 graphs)",
    ]


def test_other_ranks_log_nothing(backends, lines, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 3)
    runner = _runner_shell(_Toy())
    _wrap(runner)
    _extract(runner, x=torch.randn(4, 8))
    runner.model(x=torch.randn(2, 8))
    runner.model(x=torch.randn(4, 8))

    assert backends.names() == ["capture", "served", "served"]
    assert [r for r in lines if r.levelno >= logging.INFO] == []


@pytest.mark.parametrize(
    "state, expect",
    [
        ("parallel_trace", "parallel trace pool"),
        ("tensor_replacement", "tensor replacement is configured"),
    ],
)
def test_a_refused_state_keeps_a_wrapper_each(
    backends, lines, monkeypatch, state, expect
):
    module = _Toy()
    replacement = None
    if state == "parallel_trace":
        monkeypatch.delenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE")
    else:
        replacement = SimpleNamespace(resolve=None)
    runner = _runner_shell(module, tensor_replacement=replacement)
    _wrap(runner)

    # The parallel trace pool meta-swaps the module under the capture wrapper.
    assert runner.capture_backend_model._orig_mod is module
    assert runner.model._orig_mod is module
    assert runner.capture_backend_model is not runner.model
    _extract(runner, x=torch.randn(4, 8))
    runner.model(x=torch.randn(4, 8))
    assert backends.names() == ["capture", "served"]
    assert [call["in_trace"] for call in backends.calls] == [True, True]
    assert counters["stats"]["unique_graphs"] == 2
    refused = [m for m in _text(lines) if "trace each graph separately" in m]
    assert len(refused) == 1 and expect in refused[0]


def test_the_native_backend_is_refused_by_name(monkeypatch):
    from vllm_neuron.vllm.worker.trace_once import trace_once_refusal

    monkeypatch.setenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE", "1")
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")
    assert "native backend" in trace_once_refusal(tensor_replacement=False)
    monkeypatch.delenv("VLLM_NEURON_BACKEND")
    assert trace_once_refusal(tensor_replacement=False) is None


def test_no_capture_keeps_the_served_wrapper_only(backends):
    module = _Toy()
    runner = _runner_shell(module)
    _wrap(runner, capture=False)

    assert runner.capture_backend_model is None
    assert runner.model._orig_mod is module
    runner.model(x=torch.randn(4, 8))
    assert backends.names() == ["served"]
    assert backends.calls[0]["in_trace"] is True


class _StandInModel(torch.nn.Module):
    """A model class with the one constructor ``load_model`` calls."""

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        return cls()

    def forward(self, x):
        return x + 1


@pytest.mark.parametrize("parallel_trace", [False, True])
def test_load_model_builds_the_wrappers_of_the_start(
    backends, monkeypatch, parallel_trace
):
    """The runner's own ``load_model``, through its CPU-compile path (no weights)."""
    from vllm_neuron.vllm.worker import neuron_model_runner as runner_module
    from vllm_neuron.vllm.worker.trace_once import ExtractionEntry

    if parallel_trace:
        monkeypatch.delenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE")
    monkeypatch.setenv("VLLM_NEURON_CPU_COMPILE", "1")
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "0")
    for name in (
        "VLLM_NEURON_DEBUG_MODE",
        "NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND",
        "VLLM_NEURON_TP_ALLREDUCE_FUSE",
    ):
        monkeypatch.delenv(name, raising=False)
    # load_model lifts dynamo's cache limit for the process; restore it afterwards.
    dynamo_config = torch._dynamo.config
    monkeypatch.setattr(
        dynamo_config, "cache_size_limit", dynamo_config.cache_size_limit
    )
    monkeypatch.setattr(
        runner_module.ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (_StandInModel, "StandIn"),
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                architecture="StandIn",
                enforce_eager=False,
                hf_config=None,
                runner_type="generate",
                model="stand-in",
                quantization=None,
            ),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            load_config=SimpleNamespace(download_dir=None),
            optimization_level=SimpleNamespace(value=1),
        ),
        neuron_config=SimpleNamespace(tensor_replacement=None),
        vision_neuron_config=None,
        is_eagle3_spec=False,
        is_mtp_spec=False,
        drafter=None,
        device=torch.device("cpu"),
    )
    runner_module.NeuronModelRunner.load_model(runner)

    assert isinstance(runner.model._orig_mod, _StandInModel)
    if parallel_trace:
        assert runner.capture_backend_model._orig_mod is runner.model._orig_mod
    else:
        assert isinstance(runner.capture_backend_model, ExtractionEntry)


# --- The tiny GLM-5.3 root, through the runner's own extraction and warm-up ----------


def _no_device_check():
    from libtorch_neuronx_lite.compile import backend as lite_backend

    return mock.patch.object(
        lite_backend, "_validate_inputs_on_device", lambda *a, **k: None
    )


def _flat(value) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _flat(item)]
    return []


class _LiteBackends:
    """The library's capture backend, and a served stand-in that hashes as compile().

    The capture backend runs unaltered except for its two CPU refusals (the CPU-mode
    flag and the device check), and writes ``graph.hlo`` under the run's compile
    directory. The served stand-in computes the cache key the way the library's
    ``compile()`` does before its cache lookup, lowers the graph that it got to HLO (the
    work of a cache miss), and runs the graph without autograd, as the device's
    executor does.
    """

    def __init__(self, workdir: str) -> None:
        self.workdir = workdir
        self.captured: list[tuple[str, str]] = []
        self.served_keys: list[tuple[str, str, bool]] = []
        self.outputs: list[list[torch.Tensor]] = []

    def capture(self, gm, example_inputs, **kwargs):
        from libtorch_neuronx_lite.compile.capture_backend import capture

        before = set(os.listdir(self.workdir))
        cpu_mode_off = mock.patch.dict(os.environ, {"NEURON_LIBTORCH_CPU_MODE": "0"})
        with _no_device_check(), cpu_mode_off:
            bail = capture(gm, example_inputs, **kwargs)
        (key,) = set(os.listdir(self.workdir)) - before
        with open(os.path.join(self.workdir, key, "graph.hlo"), "rb") as f:
            self.captured.append((key, hashlib.sha256(f.read()).hexdigest()))
        return bail

    def served(self, gm, example_inputs, **kwargs):
        from libtorch_neuronx_lite.compile import cache
        from libtorch_neuronx_lite.compile.backend import (
            _apply_platform_compiler_args,
            preprocess_and_validate_inputs,
        )
        from libtorch_neuronx_lite.compile.capture_backend import run_fx_to_hlo_pipeline

        options = kwargs.get("options", {})
        work = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
        with _no_device_check():
            work, inputs = preprocess_and_validate_inputs(
                work, list(example_inputs), options
            )
        options = _apply_platform_compiler_args(options)
        key = cache.create_cache_hash(work, inputs, options)
        lowered = os.path.join(self.workdir, "served", key)
        os.makedirs(lowered, exist_ok=True)
        hlo = run_fx_to_hlo_pipeline(work, inputs, options, lowered)[0]
        self.served_keys.append(
            (key, hashlib.sha256(hlo.SerializeToString()).hexdigest(),
             TracingContext.try_get() is not None)
        )
        outputs = self.outputs

        def run(*args):
            with torch.no_grad():
                out = gm.forward(*args)
            outputs.append([t.detach().clone() for t in _flat(out)])
            return out

        return run


def _tiny_start(monkeypatch, tmp_path, *, trace_once: bool) -> dict:
    """One start of the tiny root: extraction, warm-up, then one served step."""
    from test.vllm_neuron.model.glm5_next.tiny import (
        test_tiny_glm5next_capture_sites as sites,
    )

    if trace_once:
        monkeypatch.setenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE", "1")
    else:
        monkeypatch.delenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE", raising=False)
    workdir = tmp_path / ("trace_once" if trace_once else "two_wrappers")
    workdir.mkdir()
    lite = _LiteBackends(str(workdir))
    _register(monkeypatch, lite.served, lite.capture)
    torch._dynamo.reset()
    counters.clear()

    root, _ = sites._bound_root()
    runner = sites._runner(root)
    runner._tensor_replacer = None
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="tiny"))
    runner.neuron_config.tensor_replacement = None
    runner.compile_options = {
        "compiler_args": ["--auto-cast=none"],
        "compiler_workdir": str(workdir),
    }
    _wrap(runner)

    runner.extract_prefill_graphs(sites.PREFILL_BUCKET, 0)
    runner.extract_decode_graphs(sites.DECODE_BATCH)
    traced_by_extraction = counters["stats"]["unique_graphs"]
    runner.warmup_prefill(sites.PREFILL_BUCKET, 0)
    runner.warmup_decode(sites.DECODE_BATCH)
    traced = counters["stats"]["unique_graphs"]
    with torch.compiler.set_stance("fail_on_recompile"):
        runner.warmup_decode(sites.DECODE_BATCH)
    torch._dynamo.reset()
    return {
        "traced_by_extraction": traced_by_extraction,
        "traced": traced,
        "captured": lite.captured,
        "served": lite.served_keys,
        "outputs": lite.outputs,
    }


@pytest.mark.forked
def test_the_tiny_root_traces_each_graph_once_with_the_same_keys_and_hlo(
    monkeypatch, tmp_path
):
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e

    e2e._require_cpu_mode()
    monkeypatch.delenv("VLLM_NEURON_BACKEND", raising=False)
    two = _tiny_start(monkeypatch, tmp_path, trace_once=False)
    once = _tiny_start(monkeypatch, tmp_path, trace_once=True)
    for name, start in (("two wrappers", two), ("trace once", once)):
        print(f"{name}: traced {start['traced']} graphs "
              f"({start['traced_by_extraction']} in extraction)")
        for key, digest in start["captured"]:
            print(f"  capture key {key} hlo sha256 {digest}")
        for key, digest, in_trace in start["served"]:
            print(
                f"  served  key {key} hlo sha256 {digest} "
                f"compiled in its trace: {in_trace}"
            )

    assert (two["traced_by_extraction"], two["traced"]) == (2, 4)
    assert (once["traced_by_extraction"], once["traced"]) == (2, 2)
    assert once["captured"] == two["captured"]
    assert [s[:2] for s in once["served"]] == [s[:2] for s in two["served"]]
    assert [key for key, _ in two["captured"]] == [s[0] for s in two["served"]]
    assert [s[2] for s in two["served"]] == [True, True]
    assert [s[2] for s in once["served"]] == [False, False]
    # The two warm-up steps and the served step after them, in one order on both starts.
    assert len(once["outputs"]) == len(two["outputs"]) == 3
    for mine, theirs in zip(once["outputs"], two["outputs"]):
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert torch.equal(a, b)
