# SPDX-License-Identifier: Apache-2.0
"""A carrier operand the runner builds is cast on the host, never after it reaches the device."""

from __future__ import annotations

import ast
import os
import pathlib

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny_forward

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: Each of these converts dtype, which is the copy the eager backend refuses.
CASTS = frozenset({
    "to", "int", "long", "float", "half", "bfloat16", "double", "short", "bool", "type",
})

#: The dtype names a ``.to()`` may be given positionally.
DTYPES = frozenset({
    "int8", "int16", "int32", "int64", "uint8", "float16", "float32", "float64",
    "bfloat16", "bool", "half", "float", "double", "long", "int", "short",
})

#: The helper whose mappings are the carriers, read by name rather than by position.
CARRIER_HELPER = "_glm5next_layer_carriers"

#: The two keys the sparse carrier gained with the paged latent bank: the request's block
#: table, and each token's physical row of that bank.
PAGED_KEYS = ("block_table_row", "latent_slots")

#: The helper the physical rows come from. It is a ``_glm5next_`` name, so whatever it
#: builds is already inside the census above.
SLOTS_HELPER = "_glm5next_latent_slot_mapping"

#: A second prefill chunk: it starts past 0 and its pools close on the sequence's boundaries.
CHUNK_START = 33


def _require_cpu_mode() -> None:
    """The flag comes from the process environment; the seams read it at import."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise RuntimeError("VLLM_NEURON_CPU_MODE must be 1 in the process environment")


def _on_device(node: ast.AST, bound: set[str]) -> bool:
    """True when this expression's value lives on the device it was constructed with.
    """
    if isinstance(node, ast.Call):
        if any(keyword.arg == "device" for keyword in node.keywords):
            return True
        parts = list(node.args) + [keyword.value for keyword in node.keywords]
        if isinstance(node.func, ast.Attribute):
            parts.append(node.func.value)
        return any(_on_device(part, bound) for part in parts)
    if isinstance(node, ast.Name):
        return node.id in bound
    if isinstance(node, ast.BinOp):
        return _on_device(node.left, bound) or _on_device(node.right, bound)
    if isinstance(node, ast.Compare):
        return _on_device(node.left, bound) or any(
            _on_device(other, bound) for other in node.comparators
        )
    if isinstance(node, (ast.UnaryOp,)):
        return _on_device(node.operand, bound)
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _on_device(node.value, bound)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_on_device(element, bound) for element in node.elts)
    return False


def _device_locals(helper: ast.FunctionDef) -> set[str]:
    """Names this helper binds to a value that lives on the device, to a fixed point."""
    bound: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(helper):
            if not isinstance(node, ast.Assign) or not _on_device(node.value, bound):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in bound:
                    bound.add(target.id)
                    changed = True
    return bound


def _converts_dtype(call: ast.Call) -> bool:
    """True when this call changes the dtype of what it is given."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in CASTS:
        return False
    if func.attr == "type":
        return bool(call.args)
    if func.attr != "to":
        return True
    named = {
        argument.attr for argument in call.args if isinstance(argument, ast.Attribute)
    }
    if named & DTYPES:
        return True
    return any(keyword.arg == "dtype" for keyword in call.keywords)


def _census(path: pathlib.Path) -> tuple[int, list[tuple[int, str]]]:
    """``(helpers read, offenders)``; an offender is ``(line, helper name)``."""
    helpers, offenders = 0, []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("_glm5next_"):
            continue
        helpers += 1
        bound = _device_locals(node)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and _converts_dtype(inner):
                subject = inner.func.value
                if _on_device(subject, bound):
                    offenders.append((inner.lineno, node.name))
    return helpers, offenders


def _carrier_entries(path: pathlib.Path) -> dict[str, ast.AST]:
    """``key -> the expression built under it`` for every mapping the carrier helper
    builds.
    """
    entries: dict[str, ast.AST] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef) or node.name != CARRIER_HELPER:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Dict):
                continue
            for key, value in zip(inner.keys, inner.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    entries.setdefault(key.value, value)
    return entries


def _runner_path() -> pathlib.Path:
    """The file the runner under test was imported from."""
    import vllm_neuron.vllm.worker.neuron_model_runner as runner

    return pathlib.Path(runner.__file__)


def test_no_carrier_helper_converts_a_dtype_on_the_device() -> None:
    """No ``_glm5next_`` carrier helper converts a dtype after the move."""
    _require_cpu_mode()
    path = _runner_path()
    helpers, offenders = _census(path)
    named = ",".join(f"{line}:{helper}" for line, helper in offenders)
    assert helpers >= 8, f"read only {helpers} _glm5next_ helpers in {path}"
    assert not offenders, (
        f"{len(offenders)} carrier helper site(s) convert a dtype after the tensor "
        f"reached the device, which the eager backend refuses: {named} in {path}"
    )


def _pool_slots_by_the_rule(tokens: int, start: int, pool: int) -> list[int]:
    """The indexer's rule in plain python: the last token of a complete pool carries its id."""
    return [
        (position // pool) if (position + 1) % pool == 0 else -1
        for position in range(start, start + tokens)
    ]


def test_the_pool_slots_are_int32_and_hold_at_a_chunked_start() -> None:
    """Chunk 0 equals the operand; a later chunk equals the rule, both in int32."""
    _require_cpu_mode()
    device = torch.device("cpu")
    pool = int(tiny_forward.MLA_INDEX_KPOOL)
    tokens = int(tiny_forward.STACK_TOKENS)
    selection = tiny_forward._mla_selection_operands(tokens=tokens, pages=int(tiny_forward.STACK_PAGES))

    first = NeuronModelRunner._glm5next_pool_slot_mapping(
        tokens=tokens, start_position=0, index_kpool=pool, device=device
    )
    shifted = NeuronModelRunner._glm5next_pool_slot_mapping(
        tokens=tokens, start_position=CHUNK_START, index_kpool=pool, device=device
    )
    expected = torch.tensor(
        _pool_slots_by_the_rule(tokens, CHUNK_START, pool), dtype=torch.int32
    )

    assert first.dtype == torch.int32
    assert shifted.dtype == torch.int32
    assert first.device.type == device.type
    assert torch.equal(first, selection["slot_mapping"])
    assert torch.equal(shifted, expected), (
        "the chunked derivation left the indexer's rule: only the last token of each "
        "complete pool carries a pool id, and the pool closes on the sequence's positions"
    )
    assert not torch.equal(shifted, first), (
        "the derivation ignores start_position, so a second prefill chunk would pool on "
        "the chunk's own boundaries instead of the sequence's"
    )


def test_the_paged_carrier_operands_are_built_on_the_host_and_moved_once() -> None:
    """The two operands the paged latent bank added, read where they are constructed.
    """
    _require_cpu_mode()
    path = _runner_path()
    entries = _carrier_entries(path)
    present = [key for key in PAGED_KEYS if key in entries]

    assert list(present) == list(PAGED_KEYS), (
        f"the carrier names {present} of the paged operands {list(PAGED_KEYS)}; the layer "
        f"takes these as keywords, so a missing one is served as a default in {path}"
    )

    table = entries["block_table_row"]
    assert (
        isinstance(table, ast.Call)
        and isinstance(table.func, ast.Attribute)
        and table.func.attr == "to"
    ), "the block table does not end in a move, so it is not built on the host"
    built = table.func.value
    assert isinstance(built, ast.Call), (
        "the block table is moved from something that is not a construction"
    )
    keywords = {keyword.arg for keyword in built.keywords}
    assert "dtype" in keywords, (
        "the block table's dtype is not declared where the tensor is built, so it can "
        "only be reached by a cast -- and the cast would run after the move"
    )
    assert "device" not in keywords, (
        "the block table is constructed on the device it is then moved to; a converting "
        "copy of a tensor already on the device is what the eager backend refuses"
    )

    slots = entries["latent_slots"]
    assert isinstance(slots, ast.Call) and getattr(slots.func, "attr", "") == SLOTS_HELPER, (
        f"the physical rows do not come from {SLOTS_HELPER}, so they are outside the "
        f"census this file makes over every _glm5next_ helper"
    )
    assert any(keyword.arg == "device" for keyword in slots.keywords), (
        f"{SLOTS_HELPER} is called without a device, so the rows it builds on the host "
        f"never reach the one the carrier's other operands live on"
    )
