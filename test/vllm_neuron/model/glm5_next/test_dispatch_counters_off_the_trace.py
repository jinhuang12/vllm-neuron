"""Dispatch counters stay off the traced graph: a compiled forward runs twice under fail_on_recompile.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/test_dispatch_counters_off_the_trace.py

A module-level counter that a traced function reads and stores becomes a Dynamo guard on its
value, so the first call after warmup can never match the compiled graph. The counters are bumped
inside ``@torch._dynamo.assume_constant_result`` helpers, which run at trace time and put nothing
in the graph. The stance is set here explicitly: the runner's forward context skips
``fail_on_recompile`` in CPU mode (``vllm_neuron/utils/neuron_utils.py``). Four readings: a source census of every store to module-level state in the seams;
the router seam compiled and called twice, with a control that restores an in-trace store; the
miniature the load visits, compiled and called twice per class; the tiny root through the runner's
own prefill and decode captures, compiled and called twice, with Dynamo's guard list read for any
counter.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import pathlib
import re

import pytest
import torch
import torch._dynamo as dynamo

import vllm_neuron
from vllm_neuron.functional.moe import router

from test.vllm_neuron.model.glm5_next import test_router as router_item
from test.vllm_neuron.model.glm5_next.test_prepared_weight_release import (  # noqa: F401
    _first,
    _forward_inputs,
    _impl,
    _loaded,
    _written_checkpoint,
    single_rank_process_group,
)
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_dispatch_rows_out_of_trace as rows
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_parallel_arguments as par

pytestmark = [pytest.mark.fast, pytest.mark.forked]

PACKAGE = pathlib.Path(vllm_neuron.__file__).resolve().parent
FOLD = "assume_constant_result"
MODEL_DIR = "model/glm5_next"
COUNTER_FIELDS = re.compile(r"\.(nki_dispatch|torch_fallback|scale_layout_builds|last_kernel)\b|_COUNTERS\b|_BOUND\b")
VALUE_GUARDS = ("EQUALS_MATCH", "CONSTANT_MATCH", "DICT_KEYS_MATCH", "TYPE_MATCH")

_PLANTED = '''
import torch
_COUNTERS = _Counters()
def reset_dispatch_counters():
    _COUNTERS.nki_dispatch = 0
@torch._dynamo.assume_constant_result
def _record():
    _COUNTERS.last_kernel = "k"
def dispatch(x):
    if not can_run_kernel(x):
        _COUNTERS.torch_fallback += 1
        return _torch(x)
    _COUNTERS.nki_dispatch += 1
    _record()
    return x
def _torch(x):
    global _CALLS
    _CALLS = 1
    return x
'''


# ---- the source census ------------------------------------------------------------------------


def _module_names(tree: ast.Module) -> set[str]:
    """Every name bound at module scope by an assignment."""
    names: set[str] = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _stores(source: str, where: str) -> list[dict]:
    """Every store to module-level state inside a function, with its function, fold and branch."""
    tree = ast.parse(source)
    names = _module_names(tree)
    found: list[dict] = []

    def record(node, target, funcs, fallback, kind):
        name, folded = funcs[-1]
        found.append(
            {"at": f"{where}:{node.lineno}", "target": target, "function": name, "kind": kind,
             "folded": folded, "fallback": fallback, "reset": name.startswith("reset_")}
        )

    def visit(node, funcs, fallback):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            folded = any(FOLD in ast.unparse(decorator) for decorator in node.decorator_list)
            funcs = funcs + [(node.name, folded)]
            fallback = fallback or node.name.endswith("_torch") or "oracle" in node.name
        if isinstance(node, ast.If):
            in_body = fallback or "not can_run_" in ast.unparse(node.test)
            for child in node.body:
                visit(child, funcs, in_body)
            for child in node.orelse:
                visit(child, funcs, fallback)
            return
        if funcs and isinstance(node, ast.Global):
            for name in node.names:
                record(node, name, funcs, fallback, "global")
        if funcs and isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                base = target
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(target, (ast.Attribute, ast.Subscript)) and isinstance(base, ast.Name) and base.id in names:
                    record(node, ast.unparse(target), funcs, fallback, "augstore" if isinstance(node, ast.AugAssign) else "store")
        for child in ast.iter_child_nodes(node):
            visit(child, funcs, fallback)

    visit(tree, [], False)
    return found


def _classify(store: dict) -> str:
    """folded, reset, or traced -- the class that must stay empty."""
    if store["folded"]:
        return "folded"
    if store["reset"]:
        return "reset"
    return "traced"


def _seam_files() -> tuple[list[pathlib.Path], list[str]]:
    """The seam files and the model files the census covers, and the excluded paths."""
    files: list[pathlib.Path] = []
    excluded: list[str] = []
    for path in sorted((PACKAGE / "functional").rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if any(rel.startswith(prefix) for prefix in rows.EXCLUDED):
            excluded.append(rel)
            continue
        files.append(path)
    files += sorted((PACKAGE / MODEL_DIR).glob("*.py"))
    return files, excluded


def test_no_store_to_module_state_stands_in_a_traced_function():
    """Every store to module-level state inside a seam function is folded or a reset; the reader is proven."""
    files, excluded = _seam_files()
    stores: list[dict] = []
    for path in files:
        stores += _stores(path.read_text(), path.relative_to(PACKAGE).as_posix())
    by_class: dict[str, list[dict]] = {}
    for store in stores:
        by_class.setdefault(_classify(store), []).append(store)
    for store in stores:
        print(f"CENSUS|store|{store['at']}|{store['target']}|{store['function']}|{store['kind']}|{_classify(store)}")
    traced = [f"{s['at']} {s['target']} in {s['function']}" for s in by_class.get("traced", [])]
    planted = _stores(_PLANTED, "planted")
    planted_classes = [_classify(store) for store in planted]
    print(
        f"CENSUS|reader|files={len(files)} excluded={sorted(excluded)} stores={len(stores)} "
        f"folded={len(by_class.get('folded', []))} reset={len(by_class.get('reset', []))} "
        f"traced={len(traced)} planted_classes={planted_classes}"
    )
    assert planted_classes == ["reset", "folded", "traced", "traced", "traced"], planted
    assert [s["kind"] for s in planted] == ["store", "store", "augstore", "augstore", "global"], planted
    assert not traced, f"{len(traced)} store(s) to module state stand in traced functions: {traced}"


# ---- compile once, run twice ------------------------------------------------------------------


class _GuardLog(logging.Handler):
    """Collects Dynamo's guard artifact lines."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines += self.format(record).splitlines()


@contextlib.contextmanager
def _guard_log():
    """Capture Dynamo's guard listing for the compiles inside the block; yields the line list."""
    handler = _GuardLog()
    handler.setFormatter(logging.Formatter("%(message)s"))
    loggers = [logging.getLogger(name) for name in ("torch._dynamo.guards.__guards", "torch._dynamo.guards")]
    torch._logging.set_logs(guards=True)
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield handler.lines
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
        torch._logging.set_logs()


def _global_value_guards(lines: list[str]) -> list[str]:
    """The guard lines that pin a VALUE read off a module global."""
    return [line.strip() for line in lines if "G[" in line and any(kind in line for kind in VALUE_GUARDS)]


def _counter_guards(lines: list[str]) -> list[str]:
    """Every guard line that names a dispatch counter object or field, whatever the guard kind."""
    return [line.strip() for line in lines if COUNTER_FIELDS.search(line)]


def _twice(fn, args: tuple, kwargs: dict, *, fullgraph: bool):
    """Compile ``fn`` with the eager backend, run it once, then once more under fail_on_recompile."""
    dynamo.reset()
    dynamo.utils.counters.clear()
    compiled = torch.compile(fn, backend="eager", fullgraph=fullgraph, dynamic=False)
    first = compiled(*args, **kwargs)
    with torch.compiler.set_stance("fail_on_recompile"):
        second = compiled(*args, **kwargs)
    breaks = sum(dynamo.utils.counters["graph_break"].values())
    return first, second, breaks


def _same(one, two) -> bool:
    """Byte equality over tensors, and over the tensors inside tuples, lists and dicts."""
    if isinstance(one, torch.Tensor):
        return isinstance(two, torch.Tensor) and one.shape == two.shape and torch.equal(one, two)
    if isinstance(one, (tuple, list)):
        return len(one) == len(two) and all(_same(a, b) for a, b in zip(one, two))
    if isinstance(one, dict):
        return one.keys() == two.keys() and all(_same(one[k], two[k]) for k in one)
    return one == two


def _router_call() -> tuple[tuple, dict]:
    """The fused router seam's arguments at the router item's declared extents."""
    hidden_states, gamma, router_weights, bias = router_item.build_hidden_states()
    return (), dict(
        hidden_states=hidden_states,
        gamma=gamma,
        router_weights=router_weights,
        correction_bias=bias,
        top_k=router_item.DECLARED_TOP_K,
    )


def test_the_router_seam_compiles_once_and_runs_twice():
    """The fused router seam, compiled whole, answers a second identical call without a recompile."""
    landed._require_cpu_mode()
    args, kwargs = _router_call()
    router.reset_noaux_tc_counters()
    with _guard_log() as guard_lines:
        first, second, breaks = _twice(router.noaux_tc_rmsnorm_router_topk, args, kwargs, fullgraph=True)
    counted = router.noaux_tc_dispatch_counters()
    flagged = _counter_guards(guard_lines)
    for line in _global_value_guards(guard_lines)[:200]:
        print(f"CENSUS|guard|router|{line}")
    print(
        f"COUNTERS|router|dispatch={counted}|breaks={breaks}|equal={_same(first, second)}|"
        f"guard_lines={len(guard_lines)}|global_value_guards={len(_global_value_guards(guard_lines))}|"
        f"counter_guards={flagged}"
    )
    assert guard_lines, "no guard listing was captured, so the census would be empty by construction"
    assert _same(first, second)
    assert counted[1] == 0 and counted[0] >= 1, counted
    assert not flagged, flagged


def test_an_in_trace_store_is_what_the_second_call_refuses(monkeypatch):
    """With the increment restored inside the traced function, the second call raises on recompile."""
    landed._require_cpu_mode()

    def unfolded() -> None:
        router._NOAUX_TC_COUNTERS.nki_dispatch += 1

    monkeypatch.setattr(router, "_count_nki_dispatch", unfolded)
    args, kwargs = _router_call()
    router.reset_noaux_tc_counters()
    with _guard_log() as guard_lines, pytest.raises(RuntimeError) as raised:
        _twice(router.noaux_tc_rmsnorm_router_topk, args, kwargs, fullgraph=True)
    message = str(raised.value)
    first_line = message.splitlines()[0]
    listed = _counter_guards(guard_lines)
    failed = next((line.strip() for line in message.splitlines() if "nki_dispatch" in line), "absent")
    print(f"CONTROL|router|{first_line[:160]}|listed={listed[:3]}|failed={failed[:200]}")
    assert listed, "the restored in-trace increment put no counter guard in Dynamo's list"
    assert "Detected recompile" in message, message
    assert "nki_dispatch" in message, message


@pytest.mark.usefixtures("single_rank_process_group")
def test_the_miniature_forwards_compile_once_and_run_twice(tmp_path):
    """Every class the load visits, compiled and called twice on the loaded miniature."""
    landed._require_cpu_mode()
    impl = _impl()
    model = _loaded(_written_checkpoint(tmp_path))
    inputs = _forward_inputs()
    quant_config = impl.Glm5NextQuantConfig.from_model_config(model.config)
    _, block = _first(model, impl.Glm5NextMoEBlock)
    _, dense = _first(model, impl.Glm5NextDenseMLP)
    _, mla = _first(model, impl.Glm5NextMLAAttention)
    _, indexer = _first(model, impl.Glm5NextDSAIndexer)
    with torch.no_grad():
        _, _, value = mla.project_qkv(inputs["normed"])
        q_latent = mla.project_query_latent(inputs["normed"])
    calls = {
        "moe_block": (
            block.forward,
            (inputs["hidden"], inputs["normed"]),
            dict(router_gamma=inputs["gamma"], text_config=model.text_config, quant_config=quant_config),
        ),
        "dense_mlp": (dense.forward, (inputs["normed"],), dict(quant_config=quant_config)),
        "mla_qkv": (mla.project_qkv, (inputs["normed"],), {}),
        "mla_output": (mla.project_output, (value,), {}),
        "indexer_stage": (indexer.project_stage, (inputs["normed"], q_latent), {}),
    }
    flagged_all: dict[str, list[str]] = {}
    unequal: list[str] = []
    for name, (fn, args, kwargs) in calls.items():
        with torch.no_grad(), _guard_log() as guard_lines:
            first, second, breaks = _twice(fn, args, kwargs, fullgraph=False)
        flagged = _counter_guards(guard_lines)
        equal = _same(first, second)
        print(
            f"COUNTERS|miniature|class={name}|breaks={breaks}|equal={equal}|guard_lines={len(guard_lines)}|"
            f"global_value_guards={len(_global_value_guards(guard_lines))}|counter_guards={flagged}"
        )
        assert guard_lines, f"{name}: no guard listing was captured, so the census would be empty by construction"
        if flagged:
            flagged_all[name] = flagged
        if not equal:
            unequal.append(name)
    assert not flagged_all, flagged_all
    assert not unequal, unequal


def test_the_tiny_captures_compile_once_and_run_twice():
    """Both runner legs on the tiny root, compiled whole, answer a second identical call without a recompile."""
    landed._require_cpu_mode()
    _root, _caches, runner = par._bound_runner()
    flagged_all: dict[str, list[str]] = {}
    for leg in rows.LEGS:
        translated = rows._translated(runner, leg)
        with torch.no_grad(), _guard_log() as guard_lines:
            first, second, breaks = _twice(runner.model, (), translated, fullgraph=True)
        flagged = _counter_guards(guard_lines)
        value_guards = _global_value_guards(guard_lines)
        for line in value_guards[:200]:
            print(f"CENSUS|guard|{leg}|{line}")
        print(
            f"COUNTERS|tiny|leg={leg}|breaks={breaks}|equal={_same(first, second)}|"
            f"guard_lines={len(guard_lines)}|global_value_guards={len(value_guards)}|counter_guards={flagged}"
        )
        assert guard_lines, f"{leg}: no guard listing was captured, so the census would be empty by construction"
        if flagged:
            flagged_all[leg] = flagged
    assert not flagged_all, flagged_all
