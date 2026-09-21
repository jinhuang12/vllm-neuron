# SPDX-License-Identifier: Apache-2.0
"""A precondition that reads a tensor's value must not run while a tracer is tracing."""

from __future__ import annotations

import ast
import os
import pathlib

import pytest
import torch

from vllm_neuron.functional.kda.chunked_recurrence import (
    ChunkedRecurrenceError,
    kda_intra_chunk,
)
from vllm_neuron.functional.kda.decode_state import DecodeStateError, kda_decode_step
from vllm_neuron.utils import neuron_utils

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The packages a traced forward reaches, relative to the installed plugin.
SCOPE = (
    "model/glm5_next/model_fp8.py",
    "functional/kda",
    "functional/attention",
    "functional/mhc",
)

PREDICATE = "values_are_readable"
#: Reads that are values whatever they are handed; a shape read never is.
VALUE_METHODS = frozenset({"test", "tolist"})
#: What answers for the tracer that traces the caller's own types.
TRACING_INDICATORS = ("is_compiling", "TracingContext")
#: A torch reference or an NKI kernel body: the first runs only where no kernel route
#: exists, the second traces under NKI and not under this tracer.
OFF_PATH_MARKS = ("_torch_oracle", "_impl", "_torch_reference", "_kernel")

REFUSAL = "Could not guard on data-dependent expression"

#: Chunks, chunk length, key width, value width. One tile each, chunk a power of two.
NC, CHUNK, KDIM, VDIM = 2, 64, 64, 64
#: Inside the cumulative limit, and past it.
GATE_OK, GATE_OVER = -0.01, -1.0
#: One token's gate, past the decode limit.
DECODE_GATE_OVER = -61.0


def require_cpu_mode() -> None:
    """The flag comes from the process environment; the seams read it at import."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise RuntimeError("VLLM_NEURON_CPU_MODE must be 1 in the process environment")


def scoped_files() -> list[pathlib.Path]:
    """Every python file in scope, under the plugin this process imported."""
    root = pathlib.Path(neuron_utils.__file__).parent.parent
    found: list[pathlib.Path] = []
    for entry in SCOPE:
        target = root / entry
        found += sorted(target.rglob("*.py")) if target.is_dir() else [target]
    return found


def value_reads(path: pathlib.Path) -> list[tuple[int, str, str]]:
    """``(line, function, state)`` per ``.item()`` or ``.tolist()`` call in one file."""
    found = []
    for function in [node for node in ast.walk(ast.parse(path.read_text()))
                     if isinstance(node, ast.FunctionDef)]:
        guarded = guarded_lines(function)
        off_path = any(mark in function.name for mark in OFF_PATH_MARKS)
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in VALUE_METHODS:
                continue
            state = ("off_path" if off_path
                     else "guarded" if node.lineno in guarded else "UNGUARDED")
            found.append((node.lineno, function.name, state))
    return found


def guarded_lines(function: ast.FunctionDef) -> set[int]:
    """Every line the predicate guards, as a statement test or a conditional expression."""
    lines: set[int] = set()

    def asks(node: ast.AST) -> bool:
        return any(isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                   and inner.func.id == PREDICATE for inner in ast.walk(node))

    for node in ast.walk(function):
        regions = []
        if isinstance(node, ast.If) and asks(node.test):
            regions = node.body
        elif isinstance(node, ast.IfExp) and asks(node.test):
            regions = [node.body]
        for region in regions:
            lines.update(inner.lineno for inner in ast.walk(region)
                         if hasattr(inner, "lineno"))
    return lines


def predicate_consults_a_tracer() -> tuple[bool, str]:
    """``(consults, indicators)`` read off the predicate's code, never its docstring."""
    source = pathlib.Path(neuron_utils.__file__).read_text()
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name == PREDICATE)
    statements = [node for node in function.body
                  if not (isinstance(node, ast.Expr)
                          and isinstance(node.value, ast.Constant))]
    body = "\n".join(ast.unparse(node) for node in statements)
    seen = [mark for mark in TRACING_INDICATORS if mark in body]
    return bool(seen), ",".join(seen) or "none"


def intra_operands(gate: float) -> tuple[torch.Tensor, ...]:
    """The prefill seam's five operands at one tile, with every gate set to ``gate``."""
    return (
        torch.zeros(NC, CHUNK, KDIM),
        torch.zeros(NC, CHUNK, KDIM),
        torch.zeros(NC, CHUNK, VDIM),
        torch.zeros(NC, CHUNK),
        torch.full((NC, CHUNK, KDIM), gate),
    )


def decode_operands(gate: float) -> tuple[torch.Tensor, ...]:
    """The decode seam's six operands for one token, with every gate set to ``gate``."""
    return (
        torch.zeros(VDIM, KDIM),
        torch.zeros(1, KDIM),
        torch.zeros(1, KDIM),
        torch.zeros(1, VDIM),
        torch.zeros(1, 1),
        torch.full((1, KDIM), gate),
    )


def traced_call(subject, operands, keywords=None) -> tuple[bool, str, str]:
    """``(raised, text, scalars)`` for one subject compiled as the worker compiles. """
    found = torch._dynamo.config.capture_scalar_outputs
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.reset()
    compiled = torch.compile(subject, backend="eager", fullgraph=True)
    scalars = f"found={found},used=True"
    try:
        compiled(*operands, **(keywords or {}))
    except Exception as error:  # the tracer's refusal is this file's reading
        return True, str(error), scalars
    return False, "", scalars


def test_every_value_read_on_the_traced_path_is_guarded() -> None:
    """The census, plus the predicate's own reading, over the imported plugin."""
    require_cpu_mode()
    sites, unguarded = [], []
    for path in scoped_files():
        for line, function, state in value_reads(path):
            sites.append(f"{path.name}:{line}:{function}:{state}")
            if state == "UNGUARDED":
                unguarded.append(f"{path}:{line} in {function}")
    consults, _indicators = predicate_consults_a_tracer()
    assert not unguarded, (
        f"a value read that no predicate guards runs under every tracer: {unguarded}"
    )
    assert consults, (
        "the predicate decides from the tensor alone, so it answers True under the "
        "tracer that traces the caller's own types and the guarded reads run there"
    )


def test_the_traced_seams_read_no_gate_value() -> None:
    """Both seams compile with the runner's flags."""
    require_cpu_mode()
    readings = {
        "prefill": traced_call(kda_intra_chunk, intra_operands(GATE_OK)),
        "decode": traced_call(kda_decode_step, decode_operands(GATE_OK)),
    }
    for phase, (raised, text, _) in readings.items():
        assert not raised, f"the {phase} seam did not compile: {text}"


def _kda_layer_case():
    """One KDA layer with the fixture's own weights, its bank, and its tokens.
    """
    from test.vllm_neuron.model.glm5_next import test_kda_layer as stack

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    config = Glm5NextTextConfig()
    hidden = int(config.hidden_size)
    heads = stack.KDA_NUM_HEADS // stack.TP_WORLD_SIZE
    torch.manual_seed(stack.SEED)
    weights = stack._make_weights(
        hidden, heads, stack.KDA_HEAD_SIZE, stack.KDA_CONV_KERNEL_SIZE
    )
    layer = stack._impl().Glm5NextKDALayer(config, 0, stack.TP_WORLD_SIZE)
    for name, tensor in weights.items():
        target = layer if name == "input_layernorm_weight" else layer.attention
        setattr(target, name, torch.nn.Parameter(tensor.clone(), requires_grad=False))
    attention = layer.attention
    bank = (
        torch.zeros(attention.kda_conv_state_shape, dtype=attention.kda_conv_state_dtype),
        torch.zeros(
            attention.kda_recurrent_state_shape, dtype=attention.kda_recurrent_state_dtype
        ),
    )
    torch.manual_seed(stack.SEED + 1)
    tokens = torch.randn(
        stack.DECLARED_PREFILL_TOKENS + 1, hidden, dtype=torch.float32
    )
    return layer, bank, tokens, stack.DECLARED_PREFILL_TOKENS, stack.DECLARED_CHUNK


def test_the_traced_layer_forward_reads_no_gate_value() -> None:
    """The layer forward the worker compiles, at prefill and at decode. """
    require_cpu_mode()
    layer, bank, tokens, prefill, chunk = _kda_layer_case()
    conv_state, recurrent_state = bank

    def drive(rows: torch.Tensor, is_prefill: bool):
        return traced_call(
            layer,
            (rows,),
            dict(conv_state=conv_state, recurrent_state=recurrent_state,
                 is_prefill=is_prefill, chunk_size=chunk),
        )

    readings = {
        "prefill": drive(tokens[:prefill], True),
        "decode": drive(tokens[prefill:prefill + 1], False),
    }
    for phase, (_, text, _) in readings.items():
        assert REFUSAL not in text, (
            f"the {phase} layer forward still branched on a traced value: {text}"
        )


def test_an_inadmissible_gate_still_names_its_cause_in_eager() -> None:
    """The message the precondition exists for is unchanged where values are readable."""
    require_cpu_mode()
    with pytest.raises(ChunkedRecurrenceError) as prefill:
        kda_intra_chunk(*intra_operands(GATE_OVER))
    with pytest.raises(DecodeStateError) as decode:
        kda_decode_step(*decode_operands(DECODE_GATE_OVER))
    assert "cumulative gate" in str(prefill.value)
    assert "per-token gate" in str(decode.value)
