"""Dispatch rows stay off the compiled graph: the tiny prefill forward traces whole on the NKI arm.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_dispatch_rows_out_of_trace.py

A folded dispatch row (``@torch._dynamo.assume_constant_result``) is written on the host at trace
time. An unfolded one is a ``logging.Logger`` call inside the trace, which the runner's
``fullgraph=True`` compile refuses. The control below re-installs an unfolded row and shows that
refusal, and a source reader lists every logger call the seams carry, by class.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch._dynamo as dynamo

import vllm_neuron
from vllm_neuron.functional.dsa import sentinel_order

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_parallel_arguments as par

pytestmark = [pytest.mark.fast, pytest.mark.forked]

PACKAGE = pathlib.Path(vllm_neuron.__file__).resolve().parent
MODEL_FILES = ("model/glm5_next/model_fp8.py",)
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
logger = logging.getLogger(__name__)

def _record_nki_dispatch(rows: int) -> None:
    logger.info("kernel=nki rows=%d", rows)

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


def _prefill():
    """A bound tiny root and its translated prompt step, on the NKI arm."""
    landed._require_cpu_mode()
    root, _caches, runner = par._bound_runner()
    return root, par._translated_prompt(runner)


def _explain(root, translated) -> tuple[int, int, list[str]]:
    """Graph count, break count and the first line of every break reason for one traced step."""
    dynamo.reset()
    report = dynamo.explain(root.forward)(**translated)
    reasons = [str(reason.reason).splitlines()[0] for reason in report.break_reasons]
    return int(report.graph_count), int(report.graph_break_count), reasons


def _logging_breaks(reasons: list[str]) -> list[str]:
    """The break reasons that name a logging call."""
    return [reason for reason in reasons if "logging" in reason.lower()]


def _full_graph(root, translated) -> tuple[int, str | None]:
    """Compile one step with ``fullgraph=True``: graphs handed to the backend, and any refusal."""
    graphs = []

    def keep(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    dynamo.reset()
    try:
        torch.compile(root.forward, fullgraph=True, backend=keep, dynamic=False)(**translated)
    except _refusal_classes() as error:
        return len(graphs), str(error).splitlines()[0]
    return len(graphs), None


def _logger_calls(source: str, where: str) -> list[dict]:
    """Every logger call in ``source`` with its enclosing function, fold and fallback readings."""
    found: list[dict] = []

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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id in LOGGER_NAMES:
                if node.func.attr in LEVELS:
                    name, folded = funcs[-1] if funcs else ("<module>", False)
                    found.append(
                        {
                            "at": f"{where}:{node.lineno}",
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


def test_the_prefill_forward_traces_with_no_logging_break():
    """No graph break of the traced prefill step names a logging call."""
    root, translated = _prefill()
    graphs, breaks, reasons = _explain(root, translated)
    logging_breaks = _logging_breaks(reasons)
    print(
        f"TRACE|EXPLAIN|graphs={graphs} breaks={breaks} "
        f"logging_breaks={len(logging_breaks)} reasons={reasons}"
    )
    assert graphs >= 1, graphs
    assert not logging_breaks, (
        f"{len(logging_breaks)} graph break(s) name a logging call: {logging_breaks}"
    )


def test_the_prefill_forward_compiles_as_one_full_graph():
    """The runner's own form, ``fullgraph=True``, completes and hands the backend one graph."""
    root, translated = _prefill()
    graphs, refusal = _full_graph(root, translated)
    print(f"TRACE|FULLGRAPH|graphs={graphs} refusal={refusal}")
    assert refusal is None, f"the full-graph trace refused: {refusal}"
    assert graphs == 1, graphs


def test_an_unfolded_dispatch_row_is_what_the_trace_refuses(monkeypatch):
    """With the fold removed, the step breaks on the logging call and the full graph refuses."""
    root, translated = _prefill()

    def unfolded(rows: int, k: int) -> None:
        sentinel_order.logger.info(
            "[dsa-sentinel-order] kernel=nki rows=%d select_k=%d", rows, k
        )

    monkeypatch.setattr(sentinel_order, "_record_nki_dispatch", unfolded)
    _graphs, _breaks, reasons = _explain(root, translated)
    logging_breaks = _logging_breaks(reasons)
    _count, refusal = _full_graph(root, translated)
    print(f"TRACE|CONTROL|logging_breaks={len(logging_breaks)} refusal={refusal}")
    assert logging_breaks, (
        "the unfolded row broke no graph, so this file cannot see the defect it guards"
    )
    assert refusal is not None and "logging" in refusal.lower(), refusal


def test_no_shipped_path_logger_call_stands_unfolded():
    """Every logger call on a seam's shipped path is folded; the reader proves itself on a plant."""
    files, excluded = _seam_files()
    calls: list[dict] = []
    for path in files:
        calls += _logger_calls(path.read_text(), path.relative_to(PACKAGE).as_posix())
    by_class: dict[str, list[str]] = {}
    for call in calls:
        by_class.setdefault(_classify(call), []).append(call["at"])
    planted = _logger_calls(_PLANTED, "planted")
    planted_classes = sorted({_classify(call) for call in planted})
    flagged = [call["function"] for call in planted if _classify(call) == "shipped_unfolded"]
    shipped = by_class.get("shipped_unfolded", [])
    print(
        f"TRACE|READER|files={len(files)} excluded={excluded} calls={len(calls)} "
        f"folded={len(by_class.get('folded', []))} fallback={len(by_class.get('fallback', []))} "
        f"load_time={len(by_class.get('load_time', []))} shipped_unfolded={shipped} "
        f"planted_classes={planted_classes} planted_flagged={flagged}"
    )
    assert planted_classes == ["fallback", "folded", "load_time", "shipped_unfolded"], planted
    assert flagged == ["_record_nki_dispatch"], flagged
    assert not shipped, f"{len(shipped)} logger call(s) stand unfolded on a shipped path: {shipped}"
