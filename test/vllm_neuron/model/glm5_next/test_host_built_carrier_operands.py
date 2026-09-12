# SPDX-License-Identifier: Apache-2.0
"""A carrier operand the runner builds is cast on the host, never after it reaches the device.

    VLLM_NEURON_CPU_MODE=1 python -m pytest -s -rA \\
      test/vllm_neuron/model/glm5_next/test_host_built_carrier_operands.py

The eager Neuron backend refuses a dtype-converting copy of a tensor that already lives on
the device: 63 of 64 ranks left prefill graph extraction with ``Expected self.dtype() ==
dst.dtype() to be true, but got false`` raised by ``_glm5next_pool_slot_mapping``, which
built its positions with ``torch.arange(..., device=device)`` and ended ``.to(torch.int32)``.
Graph extraction runs the runner's operand derivations EAGERLY -- they build the kwargs the
capture is then given -- so a cast there is executed, not traced.

B01 is an AST census of that mistake over every ``_glm5next_*`` helper. B02 drives the
repaired derivation at chunk start 0, where a landed operand pins it, and again at a chunked
start, where the indexer's own rule does. With ``GLM53F_130_EXPECT_BASE=1`` B01 asserts the
PRE-REPAIR reading instead -- one converting site in ``_glm5next_pool_slot_mapping`` -- which
is this file's control arm, and ``GLM53F_130_RUNNER_FILE`` points it at the source to read.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as landed

pytestmark = [pytest.mark.fast, pytest.mark.forked]

RUNNER_FILE_VAR = "GLM53F_130_RUNNER_FILE"
EXPECT_BASE_VAR = "GLM53F_130_EXPECT_BASE"

#: Each of these converts dtype, which is the copy the eager backend refuses.
CASTS = frozenset({
    "to", "int", "long", "float", "half", "bfloat16", "double", "short", "bool", "type",
})

#: The dtype names a ``.to()`` may be given positionally.
DTYPES = frozenset({
    "int8", "int16", "int32", "int64", "uint8", "float16", "float32", "float64",
    "bfloat16", "bool", "half", "float", "double", "long", "int", "short",
})

#: The one converting site the pre-repair tree carries, as ``(line, helper)``.
BASE_OFFENDERS = ((4821, "_glm5next_pool_slot_mapping"),)

#: A second prefill chunk: it starts past 0 and its pools close on the sequence's boundaries.
CHUNK_START = 33


def _require_cpu_mode() -> None:
    """The flag comes from the process environment; the seams read it at import."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise RuntimeError("VLLM_NEURON_CPU_MODE must be 1 in the process environment")


def _expect_base() -> bool:
    return os.environ.get(EXPECT_BASE_VAR) == "1"


def _on_device(node: ast.AST, bound: set[str]) -> bool:
    """True when this expression's value lives on the device it was constructed with.

    Device provenance flows the way the tensor does -- out of a constructor carrying
    ``device=``, through a receiver chain, through every operand of an operator and through
    the arguments of any call built from them -- so ``torch.where`` over a device tensor is
    on the device although it names no device of its own.
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


def _runner_path() -> pathlib.Path:
    """The file the runner under test was imported from, or the one asked for."""
    import vllm_neuron.vllm.worker.neuron_model_runner as runner

    return pathlib.Path(os.environ.get(RUNNER_FILE_VAR) or runner.__file__)


def test_b01_no_carrier_helper_converts_a_dtype_on_the_device() -> None:
    """The census, over the imported runner's own file or the one asked for."""
    _require_cpu_mode()
    path = _runner_path()
    helpers, offenders = _census(path)
    named = ",".join(f"{line}:{helper}" for line, helper in offenders)
    print(f"B01|census|file={path}|helpers={helpers}|offenders={len(offenders)}"
          f"|sites={named or 'none'}")
    assert helpers >= 8, f"read only {helpers} _glm5next_ helpers in {path}"
    if _expect_base():
        assert tuple(offenders) == BASE_OFFENDERS, (
            f"expected the pre-repair converting site at {BASE_OFFENDERS}; read "
            f"{named or 'none'} in {path}"
        )
    else:
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


def test_b02_the_pool_slots_are_int32_and_hold_at_a_chunked_start() -> None:
    """Chunk 0 equals the landed operand; a later chunk equals the rule, both in int32."""
    _require_cpu_mode()
    device = torch.device("cpu")
    pool = int(landed.MLA_INDEX_KPOOL)
    tokens = int(landed.STACK_TOKENS)
    selection = landed._mla_selection_operands(tokens=tokens, pages=int(landed.STACK_PAGES))

    first = NeuronModelRunner._glm5next_pool_slot_mapping(
        tokens=tokens, start_position=0, index_kpool=pool, device=device
    )
    shifted = NeuronModelRunner._glm5next_pool_slot_mapping(
        tokens=tokens, start_position=CHUNK_START, index_kpool=pool, device=device
    )
    expected = torch.tensor(
        _pool_slots_by_the_rule(tokens, CHUNK_START, pool), dtype=torch.int32
    )
    print(f"B02|operands|first={first.dtype}:{tuple(first.shape)}"
          f"|shifted={shifted.dtype}:{tuple(shifted.shape)}"
          f"|pooled_first={int((first >= 0).sum())}"
          f"|pooled_shifted={int((shifted >= 0).sum())}|start={CHUNK_START}")

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
