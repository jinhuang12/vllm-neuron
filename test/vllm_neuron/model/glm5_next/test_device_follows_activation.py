# SPDX-License-Identifier: Apache-2.0
"""An allocation on the KDA attention's traced path lives where its activation lives."""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: Each of these allocates rather than deriving, so each decides a device.
FACTORIES = frozenset({"empty", "zeros", "ones", "full", "arange", "tensor"})


def _require_cpu_mode() -> None:
    """The flag comes from the process environment; the seams read it at import."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise RuntimeError("VLLM_NEURON_CPU_MODE must be 1 in the process environment")


def _census(path: pathlib.Path) -> tuple[int, list[tuple[int, str, str]]]:
    """``(calls read, offenders)``; an offender is ``(line, factory, enclosing def)``.
    """
    calls, offenders = [], []

    def walk(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            func = getattr(child, "func", None)
            if (
                isinstance(func, ast.Attribute)
                and func.attr in FACTORIES
                and getattr(func.value, "id", None) == "torch"
            ):
                calls.append(child.lineno)
                keywords = {keyword.arg for keyword in child.keywords}
                if where != "__init__" and "device" not in keywords:
                    offenders.append((child.lineno, func.attr, where))
            inner = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            walk(child, child.name if inner else where)

    walk(ast.parse(path.read_text()), "<module>")
    return len(calls), offenders


def test_every_allocation_outside_init_names_its_device() -> None:
    """No allocation outside ``__init__`` takes the default device."""
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next import model_fp8

    path = pathlib.Path(model_fp8.__file__)
    _calls, offenders = _census(path)
    named = ",".join(f"{line}:torch.{factory}" for line, factory, _ in offenders)
    assert not offenders, (
        f"{len(offenders)} allocation(s) outside __init__ name no device= and take "
        f"the default device under capture: {named} in {path}"
    )
