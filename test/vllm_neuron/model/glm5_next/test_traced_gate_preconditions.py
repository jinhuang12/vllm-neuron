# SPDX-License-Identifier: Apache-2.0
"""A precondition that reads a tensor's VALUE must not run while a tracer is tracing.

    VLLM_NEURON_CPU_MODE=1 python -m pytest -s -rA \\
      test/vllm_neuron/model/glm5_next/test_traced_gate_preconditions.py

Dynamo traces the caller's own types, so a tensor under it is neither a ``FakeTensor``
instance nor on ``meta``: a serving run passed both of those clauses, read a gate value
with ``.item()``, and torch then refused to guard the unbacked symbol the read had made
(``Could not guard on data-dependent expression``). ``values_are_readable`` answers for
that tracer as well now.

A01 is a census of the value reads in the model file and the KDA, attention and MHC
packages, plus a reading of the predicate's own code. A02 compiles the two KDA seams the
serving run traces, at prefill and at decode shapes, under the runner's own
``fullgraph=True``, and A04 compiles the KDA layer forward that reaches them. A03 holds
the eager message: an inadmissible gate still names its cause where values ARE readable.
With ``GLM53F_131_EXPECT_BASE=1`` A01, A02 and A04 assert the PRE-REPAIR readings
instead, which is this file's control arm.
"""

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

SENT = "GATEPRE"
EXPECT_BASE_VAR = "GLM53F_131_EXPECT_BASE"

#: The packages a traced forward reaches, relative to the installed plugin.
SCOPE = (
    "model/glm5_next/model_fp8.py",
    "functional/kda",
    "functional/attention",
    "functional/mhc",
)

PREDICATE = "values_are_readable"
#: Reads that are values whatever they are handed; a shape read never is.
VALUE_METHODS = frozenset({"item", "tolist"})
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


def say(*fields) -> None:
    """Print one machine-readable row, prefixed so a launcher can anchor on it."""
    print(f"{SENT}|" + "|".join(str(field) for field in fields))


def expect_base() -> bool:
    return os.environ.get(EXPECT_BASE_VAR) == "1"


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
    """``(consults, indicators)`` read off the predicate's CODE, never its docstring."""
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
    """``(raised, text, scalars)`` for one subject compiled as the worker compiles.

    Scalar capture is SET rather than assumed: it is what makes ``.item()`` produce a
    symbol instead of breaking the graph, and the reading being reproduced is a serving
    run that produced one. Neither this fork nor the vLLM it serves under sets it, so the
    value found here is the torch default and both values are reported.
    """
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


def test_a01_every_value_read_on_the_traced_path_is_guarded() -> None:
    """The census, plus the predicate's own reading, over the imported plugin."""
    require_cpu_mode()
    sites, unguarded = [], []
    for path in scoped_files():
        for line, function, state in value_reads(path):
            sites.append(f"{path.name}:{line}:{function}:{state}")
            if state == "UNGUARDED":
                unguarded.append(f"{path}:{line} in {function}")
    consults, indicators = predicate_consults_a_tracer()
    say("census", f"files={len(scoped_files())}", f"reads={len(sites)}",
        f"unguarded={len(unguarded)}", f"sites={','.join(sites) or 'none'}")
    say("predicate", f"consults_a_tracer={consults}", f"indicators={indicators}")
    assert not unguarded, (
        f"a value read that no predicate guards runs under every tracer: {unguarded}"
    )
    if expect_base():
        assert not consults, (
            "the pre-repair predicate was expected to consult no tracer; it consults "
            f"{indicators}, so this is not the tree the control arm names"
        )
    else:
        assert consults, (
            "the predicate decides from the tensor alone, so it answers True under the "
            "tracer that traces the caller's own types and the guarded reads run there"
        )


def test_a02_the_traced_seams_read_no_gate_value() -> None:
    """Both seams compile with the runner's flags; the pre-repair tree refuses instead."""
    require_cpu_mode()
    readings = {
        "prefill": traced_call(kda_intra_chunk, intra_operands(GATE_OK)),
        "decode": traced_call(kda_decode_step, decode_operands(GATE_OK)),
    }
    for phase, (raised, text, scalars) in readings.items():
        say("traced", f"phase={phase}", f"raised={raised}",
            f"data_dependent={REFUSAL in text}", f"scalar_capture={scalars}",
            f"text={text.splitlines()[0] if text else 'none'}")
    if expect_base():
        for phase, (raised, text, _) in readings.items():
            assert raised and REFUSAL in text, (
                f"the {phase} seam was expected to refuse on the pre-repair tree with "
                f"{REFUSAL!r}; read raised={raised} text={text!r}"
            )
    else:
        for phase, (raised, text, _) in readings.items():
            assert not raised, f"the {phase} seam did not compile: {text}"


def _kda_layer_case():
    """One KDA layer with the landed fixture's own weights, its bank, and its tokens.

    The fixture is imported rather than restated: the layer stack that drives this
    forward on the CPU is already landed, and a second copy of its weight shapes would
    drift from it.
    """
    from test.vllm_neuron.model.glm5_next import test_kda_layer as stack

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    config = Glm5NextTextConfig()
    hidden = int(config.hidden_size)
    heads = stack.DECLARED_KDA_NUM_HEADS // stack.REGISTERED_TP_WORLD_SIZE
    torch.manual_seed(stack.SEED)
    weights = stack._make_weights(
        hidden, heads, stack.DECLARED_KDA_HEAD_SIZE, stack.DECLARED_KDA_CONV_KERNEL_SIZE
    )
    layer = stack._impl().Glm5NextKDALayer(config, 0, stack.REGISTERED_TP_WORLD_SIZE)
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


def test_a04_the_traced_layer_forward_reads_no_gate_value() -> None:
    """The layer forward the worker compiles, at prefill and at decode.

    The candidate arm asserts the absence of THIS block's refusal and not the absence of
    every dynamo limitation: a graph break elsewhere in the layer is a different claim,
    and the row prints whatever came back so the distinction stays visible.
    """
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
    for phase, (raised, text, scalars) in readings.items():
        say("layer", f"phase={phase}", f"raised={raised}",
            f"data_dependent={REFUSAL in text}", f"scalar_capture={scalars}",
            f"text={text.splitlines()[0] if text else 'none'}")
    if expect_base():
        for phase, (raised, text, _) in readings.items():
            assert raised and REFUSAL in text, (
                f"the {phase} layer forward was expected to refuse on the pre-repair "
                f"tree with {REFUSAL!r}; read raised={raised} text={text!r}"
            )
    else:
        for phase, (_, text, _) in readings.items():
            assert REFUSAL not in text, (
                f"the {phase} layer forward still branched on a traced value: {text}"
            )


def test_a03_an_inadmissible_gate_still_names_its_cause_in_eager() -> None:
    """The message the precondition exists for is unchanged where values are readable."""
    require_cpu_mode()
    with pytest.raises(ChunkedRecurrenceError) as prefill:
        kda_intra_chunk(*intra_operands(GATE_OVER))
    with pytest.raises(DecodeStateError) as decode:
        kda_decode_step(*decode_operands(DECODE_GATE_OVER))
    say("eager", f"prefill={str(prefill.value).splitlines()[0]}")
    say("eager", f"decode={str(decode.value).splitlines()[0]}")
    assert "cumulative gate" in str(prefill.value)
    assert "per-token gate" in str(decode.value)
