"""Dispatch rows stay off the compiled graph: the runner's own full-graph captures trace whole.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_dispatch_rows_out_of_trace.py

The runner captures each leg by calling ``torch.compile(model, backend=..., fullgraph=True)`` on
the translated synthetic inputs. This file installs that entry with a backend that keeps the
graphs and drives the runner's own prefill and decode captures on the NKI arm. A folded dispatch
row (``@torch._dynamo.assume_constant_result``) is written on the host at trace time; an unfolded
one is a ``logging.Logger`` call inside the trace, which the full-graph compile refuses. The
control re-installs an unfolded row and shows that refusal, and a source reader lists every
logger, print and warnings call the seams carry, by class.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch._dynamo as dynamo

from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete

import vllm_neuron
from vllm_neuron.functional.dsa import sentinel_order
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_parallel_arguments as par

pytestmark = [pytest.mark.fast, pytest.mark.forked]

PACKAGE = pathlib.Path(vllm_neuron.__file__).resolve().parent
#: The two legs the runner captures, driven through its own entry points.
LEGS = ("prefill", "decode")
MODEL_FILES = ("model/glm5_next/model_fp8.py", "model/glm5_next/mtp.py")
#: Paths the reader leaves out, each with the reason it is not on the traced forward.
EXCLUDED = {
    "functional/vendored_kernels": "config factories reached only through folded helpers",
    "functional/process_groups.py": "process-group construction at start-up",
}
LOGGER_NAMES = ("logger", "_logger", "LOGGER", "log")
LEVELS = ("debug", "info", "warning", "error", "exception", "critical", "log")
FOLD = "assume_constant_result"
_REFUSAL_NAMES = ("Unsupported", "GraphBreakError", "FullGraphError")
_PLANTED = '''
import logging
import warnings
logger = logging.getLogger(__name__)

def _record_nki_dispatch(rows: int) -> None:
    logger.info("kernel=nki rows=%d", rows)

def _printed(x):
    print("kernel=nki")
    return x

def _warned(x):
    warnings.warn("kernel=nki")
    return x

def _oracle_torch(t):
    logger.info("kernel=torch")

def dispatch(x):
    if not can_run_dispatch(x):
        logger.debug("torch path")
    return x

@torch._dynamo.assume_constant_result
def _folded(n: int) -> None:
    logger.info("n=%d", n)

logger.debug("import time")
'''


def _refusal_classes() -> tuple[type, ...]:
    """The classes dynamo raises when a traced region breaks, resolved by name."""
    from torch._dynamo import exc as dynamo_exc

    found = tuple(
        getattr(dynamo_exc, name)
        for name in _REFUSAL_NAMES
        if isinstance(getattr(dynamo_exc, name, None), type)
    )
    if not found:
        pytest.fail(f"none of {_REFUSAL_NAMES} names a class in torch._dynamo.exc")
    return found


def _runner():
    """The tiny root bound on the capture harness's runner shell, on the NKI arm."""
    landed._require_cpu_mode()
    assert can_run_kernel(), "the NKI arm is off; run the captures under NKI_SIMULATOR=1"
    _root, _caches, runner = par._bound_runner()
    return runner


class _CompiledCapture:
    """The runner's compile entry, ``fullgraph=True`` over the model, keeping every graph."""

    def __init__(self, model) -> None:
        self.graphs: list = []
        self.seen: list[dict] = []

        def keep(graph_module, _example_inputs):
            self.graphs.append(graph_module)
            return graph_module.forward

        dynamo.reset()
        self.compiled = torch.compile(model, backend=keep, fullgraph=True, dynamic=False)

    def __call__(self, **kwargs):
        self.seen.append(kwargs)
        self.compiled(**kwargs)
        raise CaptureComplete


class _Recorder:
    """A capture backend that keeps the translated mapping and enters no model."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def __call__(self, **kwargs):
        self.seen.append(kwargs)
        raise CaptureComplete


def _drive(runner, leg: str) -> None:
    """The runner's own capture entry for one leg."""
    if leg == "prefill":
        runner.extract_prefill_graphs(sites.PREFILL_BUCKET, 0)
    else:
        runner.extract_decode_graphs(sites.DECODE_BATCH)


def _translated(runner, leg: str) -> dict:
    """The mapping the runner hands its capture backend for one leg."""
    recorder = _Recorder()
    runner.capture_backend_model = recorder
    _drive(runner, leg)
    assert len(recorder.seen) == 1, (leg, len(recorder.seen))
    return recorder.seen[0]


def _full_graph(runner, leg: str) -> tuple[int, str | None, int]:
    """One leg through the compiled entry: graphs kept, the refusal's first line, dispatches."""
    capture = _CompiledCapture(runner.model)
    runner.capture_backend_model = capture
    before = sentinel_order.sentinel_order_dispatch_counters()[0]
    refusal = None
    try:
        _drive(runner, leg)
    except _refusal_classes() as error:
        refusal = str(error).splitlines()[0]
    dispatched = sentinel_order.sentinel_order_dispatch_counters()[0] - before
    return len(capture.graphs), refusal, dispatched


def _explain(model, translated: dict) -> tuple[int, int, list[str]]:
    """Graph count, break count and the first line of every break reason for one traced step."""
    dynamo.reset()
    report = dynamo.explain(model.forward)(**translated)
    reasons = [str(reason.reason).splitlines()[0] for reason in report.break_reasons]
    return int(report.graph_count), int(report.graph_break_count), reasons


def _logging_breaks(reasons: list[str]) -> list[str]:
    """The break reasons that name a logging call."""
    return [reason for reason in reasons if "logging" in reason.lower()]


def _emissions(source: str, where: str) -> list[dict]:
    """Every logger, print and warnings call in ``source`` with its function, fold and fallback."""
    found: list[dict] = []

    def kind_of(call: ast.Call) -> str | None:
        func = call.func
        if isinstance(func, ast.Name) and func.id == "print":
            return "print"
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "warnings":
                return "warnings"
            if func.value.id in LOGGER_NAMES and func.attr in LEVELS:
                return "logger"
        return None

    def visit(node, funcs, fallback):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            folded = any(FOLD in ast.unparse(decorator) for decorator in node.decorator_list)
            funcs = funcs + [(node.name, folded)]
            fallback = fallback or node.name.endswith("_torch")
        if isinstance(node, ast.If):
            in_body = fallback or "not can_run_" in ast.unparse(node.test)
            for child in node.body:
                visit(child, funcs, in_body)
            for child in node.orelse:
                visit(child, funcs, fallback)
            return
        if isinstance(node, ast.Call):
            kind = kind_of(node)
            if kind is not None:
                name, folded = funcs[-1] if funcs else ("<module>", False)
                found.append(
                    {
                        "at": f"{where}:{node.lineno}",
                        "kind": kind,
                        "function": name,
                        "folded": folded,
                        "fallback": fallback,
                    }
                )
        for child in ast.iter_child_nodes(node):
            visit(child, funcs, fallback)

    visit(ast.parse(source), [], False)
    return found


def _classify(call: dict) -> str:
    """folded, load_time, fallback, or shipped_unfolded -- the class that must stay empty."""
    if call["folded"]:
        return "folded"
    if call["function"] == "<module>":
        return "load_time"
    if call["fallback"]:
        return "fallback"
    return "shipped_unfolded"


def _seam_files() -> tuple[list[pathlib.Path], list[str]]:
    """The files the reader covers and the excluded paths it names."""
    files: list[pathlib.Path] = []
    excluded: list[str] = []
    for path in sorted((PACKAGE / "functional").rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if any(rel.startswith(prefix) for prefix in EXCLUDED):
            excluded.append(rel)
            continue
        files.append(path)
    files += [PACKAGE / name for name in MODEL_FILES]
    return files, excluded


def test_the_captures_trace_with_no_logging_break():
    """No graph break of either captured leg names a logging call; every reason is printed."""
    runner = _runner()
    named: dict[str, list[str]] = {}
    for leg in LEGS:
        graphs, breaks, reasons = _explain(runner.model, _translated(runner, leg))
        logging_breaks = _logging_breaks(reasons)
        print(
            f"TRACE|EXPLAIN|leg={leg} graphs={graphs} breaks={breaks} "
            f"logging_breaks={len(logging_breaks)} reasons={reasons}"
        )
        assert graphs >= 1, (leg, graphs)
        if logging_breaks:
            named[leg] = logging_breaks
    count = sum(len(found) for found in named.values())
    assert not named, f"{count} graph break(s) name a logging call: {named}"


def test_the_captures_compile_as_one_full_graph_each():
    """Both legs go through the runner's ``fullgraph=True`` entry and hand the backend one graph."""
    runner = _runner()
    readings: dict[str, tuple[int, str | None]] = {}
    for leg in LEGS:
        graphs, refusal, dispatched = _full_graph(runner, leg)
        print(
            f"TRACE|FULLGRAPH|leg={leg} graphs={graphs} refusal={refusal} "
            f"sentinel_dispatches={dispatched}"
        )
        readings[leg] = (graphs, refusal)
    for leg, (graphs, refusal) in readings.items():
        assert refusal is None, f"the {leg} full-graph capture refused: {refusal}"
        assert graphs == 1, (leg, graphs)


def test_an_unfolded_dispatch_row_is_what_the_capture_refuses(monkeypatch):
    """With the fold removed, the prefill capture breaks on the logging call and refuses."""
    runner = _runner()

    def unfolded(rows: int, k: int) -> None:
        sentinel_order.logger.info(
            "[dsa-sentinel-order] kernel=nki rows=%d select_k=%d", rows, k
        )

    monkeypatch.setattr(sentinel_order, "_record_nki_dispatch", unfolded)
    _graphs, _breaks, reasons = _explain(runner.model, _translated(runner, "prefill"))
    logging_breaks = _logging_breaks(reasons)
    _count, refusal, _dispatched = _full_graph(runner, "prefill")
    print(f"TRACE|CONTROL|leg=prefill logging_breaks={len(logging_breaks)} refusal={refusal}")
    assert logging_breaks, (
        "the unfolded row broke no graph, so this file cannot see the defect it guards"
    )
    assert refusal is not None and "logging" in refusal.lower(), refusal


def test_no_shipped_path_emission_stands_unfolded():
    """Every logger, print or warnings call on a shipped path is folded; the reader is proven."""
    files, excluded = _seam_files()
    calls: list[dict] = []
    for path in files:
        calls += _emissions(path.read_text(), path.relative_to(PACKAGE).as_posix())
    by_class: dict[str, list[str]] = {}
    for call in calls:
        by_class.setdefault(_classify(call), []).append(call["at"])
    kinds = {kind: sum(1 for call in calls if call["kind"] == kind) for kind in
             ("logger", "print", "warnings")}
    planted = _emissions(_PLANTED, "planted")
    planted_classes = sorted({_classify(call) for call in planted})
    flagged = [call["function"] for call in planted if _classify(call) == "shipped_unfolded"]
    shipped = by_class.get("shipped_unfolded", [])
    print(
        f"TRACE|READER|files={len(files)} excluded={excluded} calls={len(calls)} "
        f"loggers={kinds['logger']} prints={kinds['print']} warnings={kinds['warnings']} "
        f"folded={len(by_class.get('folded', []))} fallback={len(by_class.get('fallback', []))} "
        f"load_time={len(by_class.get('load_time', []))} shipped_unfolded={shipped} "
        f"planted_classes={planted_classes} planted_flagged={flagged}"
    )
    assert planted_classes == ["fallback", "folded", "load_time", "shipped_unfolded"], planted
    assert flagged == ["_record_nki_dispatch", "_printed", "_warned"], flagged
    assert not shipped, f"{len(shipped)} emission(s) stand unfolded on a shipped path: {shipped}"
