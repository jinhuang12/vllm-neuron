# SPDX-License-Identifier: Apache-2.0
"""What the tool-call constraint lets GLM-5.3-Flash generate, checked on CPU.

For ``tool_choice="required"``, a named tool, or ``auto`` with a ``strict`` tool, the GLM
parser (``--tool-call-parser glm47``) adds an xgrammar structural tag in the checkpoint's
own markup to the request (``Parser._apply_structural_tag``). A server with structured
outputs enabled compiles it with vLLM's ``XgrammarBackend`` and masks every step's logits
to the tokens it allows; a server without them refuses the request. These tests compile
the exact constraint with that backend over the checkpoint's tokenizer and vocabulary
size, then ask it which tokens it allows and which outputs it accepts.

These are facts about the compiled constraint alone. On a server started with
``--reasoning-parser``, vLLM's ``StructuredOutputManager`` fills an all-true mask and
does not advance the grammar until the reasoning has ended (``should_fill_bitmask``,
``should_advance``), so with thinking on the constraint binds only after ``</think>``.
It binds from the first token when the server counts the reasoning as already ended:
``include_reasoning: false``, or thinking turned off (``reasoning_effort: "none"``,
``enable_thinking: false``), which this checkpoint's template ignores, so the prompt
has still opened ``<think>``.
"""

from __future__ import annotations

import json
import types

import pytest
import torch
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

from glm_chat import (
    ARG_KEY_END,
    ARG_KEY_START,
    ARG_VALUE_END,
    ARG_VALUE_START,
    PING,
    REASONING,
    THINK_END,
    TOOL_START,
    WEATHER,
    encode,
    make_parser,
    make_request,
    parse,
    strict,
    tool_call,
)

pytestmark = [pytest.mark.fast]

#: Draft tokens per step on a server running MTP speculative decoding with ``k=3``; the
#: backend sizes the matcher's rollback window from it.
SPECULATIVE_TOKENS = 3


@pytest.fixture(scope="module")
def backend(tokenizer, checkpoint_dir):
    """vLLM's xgrammar backend as ``StructuredOutputManager`` builds it: the language
    model's vocabulary size (``text_config.vocab_size``, which pads the tokenizer's
    vocabulary), default whitespace handling."""
    config = types.SimpleNamespace(
        structured_outputs_config=types.SimpleNamespace(disable_any_whitespace=False),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=SPECULATIVE_TOKENS),
    )
    vocab_size = json.loads((checkpoint_dir / "config.json").read_text())["text_config"]["vocab_size"]
    return XgrammarBackend(vllm_config=config, tokenizer=tokenizer, vocab_size=vocab_size)


@pytest.fixture
def grammar(backend, tokenizer, parser_cls):
    """Compile the constraint the parser adds to a request with ``fields``."""

    def compile_for(**fields):
        request = make_request(**fields)
        adjusted = make_parser(parser_cls, tokenizer, request).adjust_request(request)
        return backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG, adjusted.structured_outputs.structural_tag)

    return compile_for


def allowed(backend, grammar) -> set[int]:
    """The token ids the constraint allows at the matcher's current position."""
    bitmask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    bits = (bitmask[0].view(-1, 1) >> torch.arange(32, dtype=torch.int32)) & 1
    return set(bits.view(-1)[: backend.vocab_size].nonzero().view(-1).tolist())


def accepts(grammar, tokenizer, text: str) -> bool:
    return grammar.accept_tokens("request", encode(tokenizer, text))


@pytest.fixture(scope="module")
def eos_ids(checkpoint_dir) -> set[int]:
    return set(json.loads((checkpoint_dir / "generation_config.json").read_text())["eos_token_id"])


def ends_turn(backend, grammar, eos_ids) -> bool:
    """The constraint lets the turn end here on one of the model's EOS ids."""
    return grammar.is_terminated() or bool(eos_ids & allowed(backend, grammar))


REQUIRED = {"tool_choice": "required"}
NAMED_PING = {"tool_choice": {"type": "function", "function": {"name": "ping"}}}
AUTO_STRICT = {"tools": [strict(WEATHER), PING]}


@pytest.mark.parametrize("fields", [REQUIRED, NAMED_PING], ids=["required", "named"])
def test_a_forced_call_must_open_at_the_first_token(backend, grammar, tokenizer, fields):
    """The constraint is built without reasoning, so its first token must start a call:
    no reasoning and no ``</think>`` before it."""
    first = allowed(backend, grammar(**fields))
    assert encode(tokenizer, TOOL_START)[0] in first
    assert encode(tokenizer, THINK_END)[0] not in first
    assert all(tokenizer.decode([token]) in ("<", "<t", TOOL_START) for token in first), (
        sorted(tokenizer.decode([token]) for token in first))


@pytest.mark.parametrize("fields", [REQUIRED, NAMED_PING], ids=["required", "named"])
def test_a_forced_call_rejects_reasoning_first(grammar, tokenizer, fields):
    assert not accepts(grammar(**fields), tokenizer, f"{REASONING}{THINK_END}" + tool_call("ping"))


def test_required_accepts_a_call_to_any_tool_then_ends_the_turn(backend, grammar, tokenizer, eos_ids):
    for output in (tool_call("get_weather", city="Paris", days="3"), tool_call("ping"),
                   tool_call("ping") + tool_call("get_weather", city="Rome")):
        constraint = grammar(**REQUIRED)
        assert accepts(constraint, tokenizer, output), output
        assert ends_turn(backend, constraint, eos_ids), output


def test_a_named_choice_rejects_a_call_to_another_tool(backend, grammar, tokenizer, eos_ids):
    assert not accepts(grammar(**NAMED_PING), tokenizer, tool_call("get_weather", city="Paris"))
    constraint = grammar(**NAMED_PING)
    assert accepts(constraint, tokenizer, tool_call("ping"))
    assert ends_turn(backend, constraint, eos_ids)


def test_the_constraint_holds_values_to_the_parameter_types(grammar, tokenizer):
    """``days`` is an integer in the tool schema: the constraint takes ``3`` and refuses
    ``three``, and it refuses a call that leaves out the required ``city``."""
    assert accepts(grammar(**REQUIRED), tokenizer, tool_call("get_weather", city="Paris", days="3"))
    assert not accepts(grammar(**REQUIRED), tokenizer,
                       tool_call("get_weather", city="Paris", days="three"))
    assert not accepts(grammar(**REQUIRED), tokenizer,
                       f"{TOOL_START}get_weather{ARG_KEY_START}days{ARG_KEY_END}"
                       f"{ARG_VALUE_START}3{ARG_VALUE_END}</tool_call>")


def test_a_constrained_call_parses_with_no_reasoning(grammar, tokenizer, parser_cls):
    """A call the constraint lets through parses into the call, with no reasoning and
    no content, when it is the whole output."""
    output = tool_call("get_weather", city="Paris", days="3")
    assert accepts(grammar(**REQUIRED), tokenizer, output)
    reasoning, content, tool_calls = parse(parser_cls, tokenizer, output, **REQUIRED)
    assert (reasoning, content) == (None, None)
    assert [(c.name, json.loads(c.arguments)) for c in tool_calls] == [
        ("get_weather", {"city": "Paris", "days": 3})]


def test_auto_with_a_strict_tool_still_allows_free_text_and_a_call(backend, grammar, tokenizer, eos_ids):
    constraint = grammar(**AUTO_STRICT)
    assert accepts(constraint, tokenizer, "Let me check. " + tool_call("get_weather", city="Paris"))
    assert ends_turn(backend, constraint, eos_ids)


@pytest.mark.xfail(
    strict=True,
    reason="Parser._apply_structural_tag asks for the constraint with reasoning=False "
    "(vllm/parser/abstract_parser.py:487-490), so xgrammar's glm_4_7 tag leaves out "
    "its reasoning prefix (xgrammar/builtin_structural_tag.py:1687-1696) and its free "
    "text excludes </think> (:1627). When the server counts the reasoning as ended "
    "from the start (include_reasoning=false, or thinking off, which the template "
    "ignores: chat_template.jinja:255-256 still opens <think>), the mask applies from "
    "the first token, and with a strict tool and tool_choice=auto the model can never "
    "close the <think> the prompt opened",
)
def test_auto_with_a_strict_tool_lets_the_model_end_its_reasoning(backend, grammar, tokenizer):
    assert encode(tokenizer, THINK_END)[0] in allowed(backend, grammar(**AUTO_STRICT))
