# SPDX-License-Identifier: Apache-2.0
"""No allocation on the KDA path decides a device for itself."""

from __future__ import annotations

import ast
import inspect
import os
import pathlib

import pytest
import torch

from vllm_neuron.functional import kda as kda_package

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The package this file censuses, resolved off the imported module rather than typed.
KDA_PACKAGE = pathlib.Path(kda_package.__file__).parent


def _require_cpu_mode() -> None:
    """Both flags come from the process environment and are read at import."""
    missing = [
        name
        for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR")
        if os.environ.get(name) != "1"
    ]
    if missing:
        raise RuntimeError(f"{missing} must be 1 in the process environment")


def _package_modules() -> list[pathlib.Path]:
    """Every module of the package, any sub-directory included."""
    return sorted(KDA_PACKAGE.rglob("*.py"))


def _torch_calls(tree: ast.AST):
    """Each ``torch.<name>(...)`` call as ``(name, enclosing def, keyword names)``."""

    def walk(node: ast.AST, where: str):
        for child in ast.iter_child_nodes(node):
            func = getattr(child, "func", None)
            if isinstance(func, ast.Attribute) and getattr(func.value, "id", None) == "torch":
                yield func.attr, where, {keyword.arg for keyword in child.keywords}
            inner = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            yield from walk(child, child.name if inner else where)

    yield from walk(tree, "<module>")


def _device_sources(name: str) -> list[str]:
    """Which places say ``torch.<name>`` accepts a ``device=``, so a call omitting it picks
    one.
    """
    function = getattr(torch, name, None)
    if function is None:
        return []
    reads = []
    try:
        if "device" in inspect.signature(function).parameters:
            reads.append("signature")
    except (TypeError, ValueError):
        pass
    if "device=" in (getattr(function, "__text_signature__", None) or ""):
        reads.append("text_signature")
    doc = getattr(function, "__doc__", None) or ""
    if "device=" in next((line for line in doc.splitlines() if line.strip()), ""):
        reads.append("docstring")
    return reads


def _factory_set(modules: list[pathlib.Path]) -> tuple[list[str], list[str]]:
    """``(the names the package calls, those that take a device)``, off its own parse trees."""
    called = sorted(
        {
            name
            for path in modules
            for name, _where, _keywords in _torch_calls(ast.parse(path.read_text()))
        }
    )
    return called, [name for name in called if _device_sources(name)]


def _census(
    path: pathlib.Path, factories: frozenset[str]
) -> tuple[int, list[tuple[str, str, str]]]:
    """``(calls read, offenders)``; an offender is ``(module, enclosing def, factory)``.
    """
    calls, offenders = 0, []
    for factory, where, keywords in _torch_calls(ast.parse(path.read_text())):
        if factory not in factories:
            continue
        calls += 1
        if not factory.endswith("_like") and "device" not in keywords:
            offenders.append((path.name, where, factory))
    return calls, offenders


def test_no_allocation_in_the_kda_package_decides_a_device_for_itself() -> None:
    """No allocation in the KDA package omits ``device=``."""
    _require_cpu_mode()
    modules = _package_modules()
    called, factories = _factory_set(modules)
    kept = frozenset(factories)
    read = [_census(path, kept) for path in modules]
    calls = sum(count for count, _offenders in read)
    found = sorted(entry for _count, offenders in read for entry in offenders)
    # A predicate that keeps no name, or a walk that reads no call, reports zero offenders on
    # any tree at all, so both are asserted before the count below is believed.
    assert factories, (
        f"the device predicate kept NONE of the {len(called)} names this package calls: {called}"
    )
    assert calls, f"no call was read over the {len(modules)} modules with the kept set {factories}"
    assert not found, (
        f"{len(found)} allocation(s) in {KDA_PACKAGE.name}/ name no device= and take the "
        f"default device, which under capture is not where the activations are: {found}"
    )


def test_the_tiny_stacks_layer_schedule_is_a_dial_with_its_default() -> None:
    """The default schedule is what it was, and another schedule can now be asked for."""
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE, KDA_LAYER_TYPE

    default = tiny._stack_text_config()
    default_schedule = [DSA_LAYER_TYPE] * tiny.STACK_LAYERS
    assert list(default.layer_types) == default_schedule
    assert default.num_hidden_layers == tiny.STACK_LAYERS

    asked = [KDA_LAYER_TYPE] * (tiny.STACK_LAYERS - 1) + [DSA_LAYER_TYPE]
    mixed = tiny._stack_text_config(layer_types=asked)
    moved = [
        name
        for name in vars(default)
        if name != "layer_types" and getattr(mixed, name, None) != getattr(default, name)
    ]
    assert list(mixed.layer_types) == asked
    assert not moved, moved


