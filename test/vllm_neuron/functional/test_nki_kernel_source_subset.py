"""Every shipped kernel keeps to the call forms the NKI front end accepts.

The front end reads a kernel's source under a restricted subset and refuses a call
that expands a mapping into keyword arguments. The simulator cannot see that refusal,
because it runs the same body as plain Python, so this test reads the shipped tree
with ``ast`` instead.
"""

from __future__ import annotations

import ast
import pathlib

import vllm_neuron.functional.moe.moe_blockwise_fp8 as live

_ROOT = pathlib.Path(live.__file__).resolve().parents[1]


def _kernel_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Every function a kernel entry of one module reaches, the entries included."""
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    entries = {
        argument.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "wrap_nki"
        for argument in node.args[:1]
        if isinstance(argument, ast.Name)
    }
    for name, node in functions.items():
        for decorator in node.decorator_list:
            marked = decorator.func if isinstance(decorator, ast.Call) else decorator
            named = (
                marked.attr
                if isinstance(marked, ast.Attribute)
                else getattr(marked, "id", "")
            )
            if named == "jit":
                entries.add(name)
    reached: set[str] = set()
    pending = [name for name in entries if name in functions]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending += [
            call.func.id
            for call in ast.walk(functions[name])
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") in functions
        ]
    return [functions[name] for name in sorted(reached)]


def _forms(root: pathlib.Path) -> tuple[tuple, tuple, tuple]:
    """The keyword expansions, star arguments and comprehensions under one tree."""
    expansions: list[str] = []
    stars: list[str] = []
    comprehensions: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _kernel_functions(tree):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    where = f"{path.name}:{inner.lineno}({node.name})"
                    expansions += [
                        where for keyword in inner.keywords if keyword.arg is None
                    ]
                    stars += [
                        where
                        for argument in inner.args
                        if isinstance(argument, ast.Starred)
                    ]
                elif isinstance(
                    inner,
                    (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp),
                ):
                    comprehensions.append(f"{path.name}:{inner.lineno}({node.name})")
    return tuple(expansions), tuple(stars), tuple(comprehensions)


def test_no_kernel_call_expands_a_mapping_into_keywords():
    """No call inside a shipped kernel expands a mapping into keyword arguments."""
    expansions, stars, _ = _forms(_ROOT)
    assert expansions == (), (
        f"kernel calls that expand a mapping into keywords, which the front end "
        f"refuses: {expansions}"
    )
    assert stars == (), (
        f"kernel calls that expand a sequence into arguments, which the front end "
        f"refuses as well: {stars}"
    )
