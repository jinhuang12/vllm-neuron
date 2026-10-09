# SPDX-License-Identifier: Apache-2.0
"""The DSA decode indexer adds no host work to a decode step.

The runner compiles a decode step into one graph (``torch.compile(model,
fullgraph=True)``). The python of the functions inside it runs once, while Dynamo
traces. What runs again on every step is the runner's carrier build, the guards Dynamo
checks before it reuses the graph, and the backend call that binds the graph's inputs.
The indexer's decode stages (the query rotation, the ring step, the scores and the
selection) can therefore add host work to a step in three ways only. Each is pinned:

* **A tensor built from python data, or a tensor value read, on the traced path.** The
  build is a real tensor beside fake ones under graph extraction; the read is a host
  round trip. The scan finds neither in :data:`SCOPE` (outside the torch references and
  the kernel bodies, and outside a ``values_are_readable`` guard for a read).
* **A guard that runs python.** An environment read that the trace makes (``envs.X``,
  ``os.environ``, ``os.getenv``) becomes a guard, and that guard reads the environment
  again before every step: ``vllm_neuron.envs.__getattr__``, then ``os.getenv``. The scan
  finds no such read in :data:`SCOPE` except inside a function that
  ``torch._dynamo.assume_constant_result`` folds, which runs while tracing and leaves no
  guard; the compiled leg below reads, per step, no setting the indexer reads.
* **A graph input or a constant.** The compiled leg's inputs are the operands it is
  called with and the indexer's own parameter, and it holds no tensor constant and no
  tensor build.

The compiled leg is the batched decode indexer as ``Glm5NextMLAAttention`` runs it (the
query rotation, then ``Glm5NextDSAIndexer.forward_requests`` with the projections handed
in), compiled with ``fullgraph=True`` at the served grid (``NEURON_LOGICAL_NC_CONFIG=2``)
in both regimes, selecting and inside the bound, at ``B = 1`` and ``B = 4``. The second
step (new lengths, same shapes) must reuse the first step's graph.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/dsa/test_decode_host_path.py
"""

from __future__ import annotations

import ast
import collections
import pathlib
import sys

import pytest
import torch

from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from test.vllm_neuron.model.glm5_next.test_traced_gate_preconditions import (
    OFF_PATH_MARKS,
    guarded_lines,
)
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.utils import neuron_utils
from vllm_neuron.utils.neuron_utils import can_run_kernel

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The decode indexer's stages that this tree's change edits or adds, relative to the
#: imported plugin: the module whole, or ``(module, function names)``.
SCOPE = (
    "functional/dsa/decode_batch.py",
    "functional/dsa/decode_select.py",
    "functional/dsa/kpool_hadamard.py",
    "functional/dsa/launch_grid.py",
    ("model/glm5_next/model_fp8.py", ("forward_requests", "_forward_requests")),
)

#: Calls that build a tensor from python data: real tensors even under a fake trace.
DATA_BUILDERS = frozenset({"tensor", "as_tensor", "from_numpy"})
#: Reads of a tensor's values on the host.
VALUE_READS = frozenset({"item", "tolist", "numpy", "cpu"})
#: What reads the process environment.
ENVIRONMENT_MODULES = frozenset({"envs", "os"})
#: The decorator that folds a call into the graph as a constant, with no guard.
FOLDED = "assume_constant_result"

#: Requests' lengths step by one between the two steps; the selecting regime's lengths
#: are past the bound (``bypass_max_context(2048, 4) = 2051``), the other's inside it.
SELECTING_CONTEXT, BOUND_CONTEXT = 4096, 2048
SERVED_LNC = "2"


def _plugin_root() -> pathlib.Path:
    return pathlib.Path(neuron_utils.__file__).resolve().parent.parent


def _scoped_paths() -> list[pathlib.Path]:
    return [_plugin_root() / (entry if isinstance(entry, str) else entry[0])
            for entry in SCOPE]


def _scoped_functions():
    """``(path, function)`` per function in :data:`SCOPE`, under the imported plugin. A
    scoped module this tree lacks has nothing to scan; the scan test then fails on it."""
    for entry, path in zip(SCOPE, _scoped_paths()):
        names = None if isinstance(entry, str) else entry[1]
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if names is None or node.name in names:
                    yield path, node


def _is_folded(function: ast.FunctionDef) -> bool:
    return any(FOLDED in ast.unparse(decorator) for decorator in function.decorator_list)


def _own_nodes(function: ast.FunctionDef):
    """The function's nodes, without those of the functions defined inside it."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def host_work(path: pathlib.Path, function: ast.FunctionDef) -> list[str]:
    """Every per-step host-work spelling in one function: ``"file:line function: what"``."""
    found = []
    off_path = any(mark in function.name for mark in OFF_PATH_MARKS)
    guarded = guarded_lines(function)
    where = f"{path.name}:{{}} {function.name}"
    for node in _own_nodes(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, owner = node.func.attr, node.func.value
            torch_call = isinstance(owner, ast.Name) and owner.id == "torch"
            if not off_path and function.name != "__init__":
                if (attr in DATA_BUILDERS and torch_call) or attr == "new_tensor":
                    found.append(f"{where.format(node.lineno)}: builds a tensor from "
                                 f"python data ({ast.unparse(node.func)})")
                if attr in VALUE_READS and node.lineno not in guarded:
                    found.append(f"{where.format(node.lineno)}: reads a tensor's values "
                                 f"(.{attr}())")
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in ENVIRONMENT_MODULES
                and (node.value.id == "envs" or node.attr in ("environ", "getenv"))
                and not _is_folded(function)):
            found.append(f"{where.format(node.lineno)}: reads the environment "
                         f"({ast.unparse(node)}), a guard that reads it again every step")
    return found


def scan() -> list[str]:
    found = []
    for path, function in _scoped_functions():
        found += host_work(path, function)
    return found


def _environment_reads_of(module_file: pathlib.Path, function_name: str) -> set[str]:
    """The ``envs.X`` names one function reads (the shared kernel gate's, below)."""
    tree = ast.parse(module_file.read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    return {node.attr for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "envs"}


def _scope_settings() -> set[str]:
    """The ``envs.X`` names the functions in :data:`SCOPE` read, folded or not."""
    names = set()
    for _path, function in _scoped_functions():
        names |= {node.attr for node in _own_nodes(function)
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id == "envs"}
    return names


# ---------------------------------------------------------------------------------------
# The compiled leg
# ---------------------------------------------------------------------------------------


def _indexer(cfg):
    import vllm_neuron.model.glm5_next.model_fp8 as model_fp8

    ape = (torch.randn(int(cfg.index_kpool), int(cfg.index_head_dim),
                       generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16)
    indexer = model_fp8.Glm5NextDSAIndexer(cfg)
    indexer.index_kpool_compress_ape = torch.nn.Parameter(ape, requires_grad=False)
    return indexer


def _operands(cfg, ctx: int, batch: int, seed: int) -> dict:
    """One layer's decode operands in the bank form, lengths ``ctx - 1 - b % 4``."""
    gen = torch.Generator().manual_seed(seed)
    heads, dim, pool = int(cfg.index_n_heads), int(cfg.index_head_dim), int(cfg.index_kpool)
    lengths = [ctx - 1 - b % 4 for b in range(batch)]
    slots_total = batch + 3
    rows = -(-(ctx // pool + 1) // 128) * 128
    return {
        "query_rows": torch.randn(batch * heads, dim, generator=gen).to(torch.bfloat16),
        "key": torch.randn(batch, dim, generator=gen).to(torch.bfloat16),
        "weights": torch.randn(batch, heads, generator=gen) * (dim ** -0.5) * (heads ** -0.5),
        "gate": torch.randn(batch, dim, generator=gen).to(torch.bfloat16),
        "pool_bank": (torch.randn(slots_total, rows, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "tail_bank": (torch.randn(slots_total, 2, pool, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "slots": torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32),
        "position": torch.tensor([n - 1 for n in lengths], dtype=torch.int64),
    }


def _next_step(ops: dict) -> dict:
    """The same requests one token later: new values, the same shapes."""
    return {**ops, "seq_lens": ops["seq_lens"] + 1, "position": ops["position"] + 1}


def _leg(indexer, ctx: int):
    """The decode indexer leg the attention layer runs, from the projections on."""

    def leg(query_rows, key, weights, gate, pool_bank, tail_bank, slots, seq_lens, position):
        batch = key.shape[0]
        query = KH.dsa_hadamard128(query_rows).reshape(
            batch, indexer.index_n_heads, indexer.index_head_dim)
        return indexer.forward_requests(
            key.new_zeros((batch, 1)), None, pool_bank, tail_bank, slots, seq_lens,
            position, max_seq_len=ctx, indices_wanted=True,
            projected=(query, key, weights, gate))

    return leg


class _Recorder:
    """A backend that keeps each graph and its inputs, and marks the graph's frames."""

    def __init__(self):
        self.graphs, self.inputs = [], []

    def __call__(self, graph_module, example_inputs):
        self.graphs.append(graph_module)
        self.inputs.append(list(example_inputs))
        forward = graph_module.forward

        def _decode_host_path_graph(*args):
            return forward(*args)

        return _decode_host_path_graph


def _outside_graph(call):
    """``(result, python frames, envs settings read)`` of one call, outside the graph."""
    frames, settings = collections.Counter(), collections.Counter()
    depth = [0]
    plugin = str(_plugin_root())

    def hook(frame, event, _arg):
        name = frame.f_code.co_name
        if name == "_decode_host_path_graph":
            depth[0] += 1 if event == "call" else -1 if event == "return" else 0
            return
        if depth[0] or event != "call":
            return
        filename = frame.f_code.co_filename
        if filename.startswith(plugin):
            frames[f"{pathlib.Path(filename).relative_to(plugin)}:{name}"] += 1
            if filename.endswith("envs.py") and name == "__getattr__":
                settings[frame.f_locals.get("name")] += 1

    sys.setprofile(hook)
    try:
        result = call()
    finally:
        sys.setprofile(None)
    return result, frames, settings


def _compiled_steps(monkeypatch, ctx: int, batch: int):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", SERVED_LNC)
    cfg = decode_config(hidden_size=512, q_lora_rank=256)
    indexer = _indexer(cfg)
    first = _operands(cfg, ctx, batch, seed=ctx + batch)
    second = _next_step(first)
    recorder = _Recorder()
    torch._dynamo.reset()
    compiled = torch.compile(_leg(indexer, ctx), backend=recorder, fullgraph=True)
    compiled(**first)
    graphs_after_first = len(recorder.graphs)
    _result, frames, settings = _outside_graph(lambda: compiled(**second))
    return indexer, first, recorder, graphs_after_first, frames, settings


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"


def test_the_decode_indexer_spells_no_host_work_on_its_traced_path():
    """The scan over :data:`SCOPE`: no data build, no unguarded value read, no unfolded
    environment read."""
    found = scan()
    assert not found, (
        "the decode indexer's traced path spells per-step host work:\n  "
        + "\n  ".join(found))
    missing = [str(path) for path in _scoped_paths() if not path.exists()]
    assert not missing, f"scoped modules missing from the imported plugin: {missing}"


@pytest.mark.parametrize("ctx", [SELECTING_CONTEXT, BOUND_CONTEXT],
                         ids=["selecting", "inside-the-bound"])
@pytest.mark.parametrize("batch", [1, 4])
def test_a_cached_decode_step_runs_no_indexer_python_and_binds_only_its_operands(
        monkeypatch, ctx, batch):
    """One graph, reused by the next step; per step, no python of :data:`SCOPE` and no read
    of a setting the indexer reads; the inputs are the operands and the parameter."""
    indexer, first, recorder, graphs_after_first, frames, settings = _compiled_steps(
        monkeypatch, ctx, batch)
    scope_files = {pathlib.Path(entry if isinstance(entry, str) else entry[0]).name
                   for entry in SCOPE}
    indexer_frames = {f: n for f, n in frames.items()
                      if pathlib.Path(f.split(":")[0]).name in scope_files}
    gate_settings = _environment_reads_of(pathlib.Path(neuron_utils.__file__),
                                          "can_run_kernel")
    indexer_settings = _scope_settings()
    read_per_step = set(settings)

    assert graphs_after_first == 1 and len(recorder.graphs) == 1, (
        f"{len(recorder.graphs)} graphs ({graphs_after_first} after the first step): the "
        f"next step's values recompiled the leg")
    assert not indexer_frames, (
        f"a cached step ran the indexer's python outside the graph: {indexer_frames}")
    assert not read_per_step & indexer_settings, (
        f"a cached step re-read {sorted(read_per_step & indexer_settings)} from the "
        f"environment: a guard on a setting the indexer read while tracing")
    # Whatever else a step reads is the shared kernel gate's (``can_run_kernel``, which
    # every kernel module calls), not the indexer's.
    assert read_per_step <= gate_settings, (read_per_step, gate_settings)

    graph = recorder.graphs[0].graph
    operands = [value for value in first.values() if torch.is_tensor(value)]
    allowed = operands + [indexer.index_kpool_compress_ape]
    foreign = [index for index, value in enumerate(recorder.inputs[0])
               if not any(value is known for known in allowed)]
    constants = [node.target for node in graph.nodes if node.op == "get_attr"]
    builds = [node.name for node in graph.nodes if node.op == "call_function"
              and getattr(node.target, "__name__", "") in DATA_BUILDERS]
    assert not foreign, (
        f"graph inputs {foreign} are neither an operand nor the indexer's parameter: "
        f"{[tuple(recorder.inputs[0][i].shape) for i in foreign]}")
    assert not constants and not builds, (constants, builds)


@pytest.mark.parametrize("setting, pair", [(None, False), ("1", False), ("2", True)])
def test_the_grid_follows_the_served_settings(monkeypatch, setting, pair):
    """Unset or 1: one program; 2: the LNC2 pair. Eagerly, every call reads the setting."""
    from vllm_neuron.functional.dsa import launch_grid

    if setting is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", setting)
    assert launch_grid.lnc_pair() is pair


@pytest.mark.parametrize("setting", ["4", "0", "two"])
def test_an_unserved_setting_is_refused_by_name(monkeypatch, setting):
    from vllm_neuron.functional.dsa import launch_grid

    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", setting)
    with pytest.raises(launch_grid.LaunchGridError, match="NEURON_LOGICAL_NC_CONFIG"):
        launch_grid.lnc_pair()
