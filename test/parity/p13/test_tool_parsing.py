# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash tool calls and reasoning through vLLM's own parsers, on CPU.

The parser under test is the one ``vllm serve`` builds for this checkpoint when it is
started with ``--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser
glm45``: ``ParserManager.get_parser`` composes ``Glm47MoeParserReasoningAdapter`` and
``Glm47MoeModelToolParser``, both backed by ``vllm/parser/glm47_moe.py``. It reads the
checkpoint's own markup, which its chat template both teaches the model and writes for
earlier tool-call turns::

    <think>reasoning</think>content<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>

The generation prompt ends ``<|assistant|><think>``, so a completion starts inside the
reasoning and the model closes it with ``</think>``.

Model outputs here are either constructed in the template's markup or recorded from a
GLM-5.3-Flash server (``fixtures/recorded_chat_outputs.json``). Each is turned into token
ids with the checkpoint's tokenizer and, for streaming, back into the per-token text
vLLM's ``FastIncrementalDetokenizer`` produces, then handed to the parser the way
``OpenAIServingChat`` hands it engine output: ``parse`` for a whole response,
``parse_delta`` once per engine step for a stream.

Tests marked ``xfail(strict=True)`` state the behaviour an OpenAI client relies on and
that this parser does not give; each names the defect in its reason.
"""

from __future__ import annotations

import json
import pathlib
import random

import pytest
import tokenizers
from glm_chat import (
    ARG_KEY_END,
    ARG_KEY_START,
    ARG_VALUE_END,
    ARG_VALUE_START,
    MARKUP,
    MESSAGES,
    OBSERVATION,
    PING,
    REASONING,
    THINK_END,
    THINK_START,
    TOOL_CALL_ID_PREFIX,
    TOOL_END,
    TOOL_START,
    TOOLS,
    WEATHER,
    encode,
    make_parser,
    make_request,
    parse,
    prompt_ids,
    strict,
    template_kwargs,
    tool_call,
)

pytestmark = [pytest.mark.fast]

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# ── helpers ─────────────────────────────────────────────────────────────────────────


def detokenized(tokenizer, prompt: list[int], ids: list[int]) -> list[str]:
    """Per-token text exactly as ``FastIncrementalDetokenizer`` emits it for a request
    whose parser set ``skip_special_tokens=False``: a ``DecodeStream`` primed with the
    prompt, one ``step`` per generated token, an incomplete UTF-8 tail held back."""
    stream = tokenizers.decoders.DecodeStream(ids=prompt, skip_special_tokens=False)
    return [stream.step(tokenizer._tokenizer, token) or "" for token in ids]


def stream(parser_cls, tokenizer, ids: list[int], sizes, **fields) -> list:
    """The ``DeltaMessage`` list a streamed response builds when the engine hands the
    tokens over in steps of ``sizes`` tokens (cycled), the last step finishing."""
    request = make_request(**fields)
    parser = make_parser(parser_cls, tokenizer, request)
    request = parser.adjust_request(request)
    prompt = prompt_ids(tokenizer, request)
    texts = detokenized(tokenizer, prompt, ids)
    sizes = list(sizes)
    deltas, start, step = [], 0, 0
    while start < len(ids):
        end = min(len(ids), start + sizes[step % len(sizes)])
        delta = parser.parse_delta(
            "".join(texts[start:end]), ids[start:end], request,
            prompt_token_ids=prompt, finished=end == len(ids),
        )
        if delta is not None:
            deltas.append(delta)
        start, step = end, step + 1
    return deltas


def assemble(deltas) -> dict:
    """What a client accumulates from the deltas: reasoning, content, and per tool-call
    index the ids, names and argument text it was sent."""
    out = {"reasoning": "", "content": "", "calls": {}}
    for delta in deltas:
        out["reasoning"] += delta.reasoning or ""
        out["content"] += delta.content or ""
        for call in delta.tool_calls or []:
            slot = out["calls"].setdefault(call.index, {"ids": [], "names": [], "arguments": ""})
            if call.id is not None:
                slot["ids"].append(call.id)
            if call.function is not None:
                if call.function.name is not None:
                    slot["names"].append(call.function.name)
                slot["arguments"] += call.function.arguments or ""
    return out


def assert_stream_matches_parse(streamed: dict, reasoning, content, tool_calls) -> None:
    """A streamed response carries what the non-streamed response carries. When tools are
    called the whole response strips the whitespace around its content
    (``ParserEngine._strip_content_whitespace``) and a stream sends it as generated, so
    content is then compared stripped."""
    calls = tool_calls or []
    assert streamed["reasoning"] == (reasoning or "")
    if calls:
        assert streamed["content"].strip() == (content or "")
    else:
        assert streamed["content"] == (content or "")
    assert sorted(streamed["calls"]) == list(range(len(calls)))
    for index, call in enumerate(calls):
        slot = streamed["calls"][index]
        assert slot["names"] == [call.name], slot
        assert len(slot["ids"]) == 1 and slot["ids"][0].startswith(TOOL_CALL_ID_PREFIX), slot
        assert json.loads(slot["arguments"]) == json.loads(call.arguments), slot
    ids = [streamed["calls"][index]["ids"][0] for index in streamed["calls"]]
    assert len(set(ids)) == len(ids), ids


# Outputs used by several tests.
TWO_CALLS = (
    f"{REASONING}{THINK_END}"
    + tool_call("get_weather", city="Paris", days="3")
    + tool_call("get_weather", city="Rome", metric="false")
)
#: A string argument whose JSON form needs escapes, split by the tokenizer inside the value.
QUOTED = 'He said "bring an umbrella" \\ then left.\nLine two: 東京, café, ±5°'
QUOTED_CALL = f"{REASONING}{THINK_END}" + tool_call("get_weather", city=QUOTED, days="12345")


# ── the checkpoint's markup ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("markup", MARKUP)
def test_each_markup_string_is_one_non_special_token(tokenizer, markup):
    """One id per tag, and not a special token, so the tag survives the detokenizer's
    default ``skip_special_tokens=True`` on a server started without parsers."""
    ids = encode(tokenizer, markup)
    assert len(ids) == 1, ids
    assert tokenizer.decode(ids, skip_special_tokens=True) == markup


@pytest.mark.parametrize("effort", [None, "low", "high", "none"])
def test_the_generation_prompt_opens_the_reasoning(tokenizer, parser_cls, effort):
    """Whatever ``reasoning_effort`` is sent, the prompt ends inside ``<think>``, and the
    parser reads the prompt as reasoning not yet ended."""
    request = make_request(reasoning_effort=effort) if effort else make_request()
    prompt = prompt_ids(tokenizer, request)
    assert tokenizer.decode(prompt[-2:], skip_special_tokens=False) == "<|assistant|>" + THINK_START
    parser = make_parser(parser_cls, tokenizer, make_request())
    assert parser.is_reasoning_end(prompt) is False


def test_a_tool_call_turn_ends_on_an_eos_token(tokenizer, checkpoint_dir):
    """The template follows an assistant tool-call turn with ``<|observation|>``, and the
    generation config lists that token as an EOS id, so a turn that calls tools stops
    there with finish_reason ``stop``."""
    generation = json.loads((checkpoint_dir / "generation_config.json").read_text())
    (observation,) = encode(tokenizer, OBSERVATION)
    assert observation in generation["eos_token_id"]
    history = MESSAGES + [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
    ]
    text = tokenizer.apply_chat_template(history, tools=TOOLS, tokenize=False)
    assert tool_call("get_weather", city="Paris") + OBSERVATION in text


def test_a_tool_call_the_template_wrote_parses_back_to_the_same_call(tokenizer, parser_cls):
    """An earlier assistant turn rendered by the chat template is the markup the model is
    taught to write; parsed back it gives the same names and the schema-typed arguments."""
    calls = [("get_weather", {"city": "Paris", "days": 3}), ("ping", {})]
    history = MESSAGES + [{
        "role": "assistant", "content": "", "reasoning_content": REASONING,
        "tool_calls": [{"id": f"call_{i}", "type": "function",
                        "function": {"name": name, "arguments": args}}
                       for i, (name, args) in enumerate(calls)],
    }]
    text = tokenizer.apply_chat_template(history, tools=TOOLS, tokenize=False)
    turn = text.rsplit("<|assistant|>", 1)[1]
    assert turn.startswith(THINK_START)
    reasoning, content, tool_calls = parse(parser_cls, tokenizer, turn[len(THINK_START):])
    assert reasoning == REASONING
    assert content is None
    assert [(c.name, json.loads(c.arguments)) for c in tool_calls] == calls


# ── one response, parsed whole ──────────────────────────────────────────────────────


def test_one_call_is_split_from_the_reasoning(tokenizer, parser_cls):
    output = f"{REASONING}{THINK_END}" + tool_call("get_weather", city="Paris", days="3")
    reasoning, content, tool_calls = parse(parser_cls, tokenizer, output)
    assert reasoning == REASONING
    assert content is None
    (call,) = tool_calls
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Paris", "days": 3}
    assert call.id.startswith(TOOL_CALL_ID_PREFIX)


def test_several_calls_keep_their_order_and_get_distinct_ids(tokenizer, parser_cls):
    output = TWO_CALLS + tool_call("ping")
    _, content, tool_calls = parse(parser_cls, tokenizer, output)
    assert content is None
    assert [(c.name, json.loads(c.arguments)) for c in tool_calls] == [
        ("get_weather", {"city": "Paris", "days": 3}),
        ("get_weather", {"city": "Rome", "metric": False}),
        ("ping", {}),
    ]
    assert len({c.id for c in tool_calls}) == 3


@pytest.mark.parametrize(
    "between, content",
    [("Let me check.\n", "Let me check."), ("\n\n", None)],
    ids=["text", "whitespace"],
)
def test_content_before_a_call_is_kept_and_whitespace_is_dropped(
    tokenizer, parser_cls, between, content
):
    output = f"{REASONING}{THINK_END}{between}" + tool_call("get_weather", city="Paris")
    _, got, tool_calls = parse(parser_cls, tokenizer, output)
    assert got == content
    assert [c.name for c in tool_calls] == ["get_weather"]


def test_argument_values_take_the_types_the_tool_schema_declares(tokenizer, parser_cls):
    """The markup carries every value as text; the tool schema types it."""
    output = f"{REASONING}{THINK_END}" + tool_call(
        "get_weather", city="Paris", days="3", metric="true", opts='{"units": "C"}',
        tags='["rain", "wind"]', ratio="0.5", note="null",
    )
    (call,) = parse(parser_cls, tokenizer, output)[2]
    assert json.loads(call.arguments) == {
        "city": "Paris", "days": 3, "metric": True, "opts": {"units": "C"},
        "tags": ["rain", "wind"], "ratio": 0.5, "note": None,
    }


def test_a_string_argument_is_escaped_into_valid_json(tokenizer, parser_cls):
    (call,) = parse(parser_cls, tokenizer, QUOTED_CALL)[2]
    assert json.loads(call.arguments) == {"city": QUOTED, "days": 12345}


def test_values_that_do_not_fit_the_schema_stay_strings_in_valid_json(tokenizer, parser_cls):
    output = f"{REASONING}{THINK_END}" + tool_call(
        "get_weather", days="three", metric="yes", opts='{"units": "C"', ratio="half")
    (call,) = parse(parser_cls, tokenizer, output)[2]
    assert json.loads(call.arguments) == {
        "days": "three", "metric": "yes", "opts": '{"units": "C"', "ratio": "half"}


def test_a_call_without_arguments_has_empty_object_arguments(tokenizer, parser_cls):
    (call,) = parse(parser_cls, tokenizer, f"{REASONING}{THINK_END}" + tool_call("ping"))[2]
    assert (call.name, call.arguments) == ("ping", "{}")


def test_a_call_inside_unclosed_reasoning_ends_the_reasoning(tokenizer, parser_cls):
    reasoning, content, tool_calls = parse(
        parser_cls, tokenizer, REASONING + tool_call("get_weather", city="Paris"))
    assert reasoning == REASONING
    assert content is None
    assert [(c.name, json.loads(c.arguments)) for c in tool_calls] == [
        ("get_weather", {"city": "Paris"})]


def test_an_answer_without_a_call_is_split_into_reasoning_and_content(tokenizer, parser_cls):
    reasoning, content, tool_calls = parse(
        parser_cls, tokenizer, f"{REASONING}{THINK_END}It is sunny in Paris.")
    assert (reasoning, content) == (REASONING, "It is sunny in Paris.")
    assert tool_calls is None


def _recorded():
    return json.loads((FIXTURES / "recorded_chat_outputs.json").read_text())["outputs"]


@pytest.mark.parametrize("record", _recorded(), ids=lambda r: r["name"])
def test_recorded_completions_split_at_the_end_of_reasoning(tokenizer, parser_cls, record):
    """A recorded server answer splits at its ``</think>``; one cut off inside its
    reasoning is all reasoning. Tools in the request change nothing when none is called."""
    text = record["content"]
    reasoning, content, tool_calls = parse(parser_cls, tokenizer, text)
    if THINK_END in text:
        before, after = text.split(THINK_END, 1)
        assert (reasoning, content) == (before.rstrip(), after)
    else:
        assert (reasoning, content) == (text.rstrip(), None)
    assert tool_calls is None


# ── malformed calls ─────────────────────────────────────────────────────────────────


def test_a_call_to_an_unknown_tool_is_returned_as_content(tokenizer, parser_cls):
    markup = tool_call("get_time", zone="UTC")
    _, content, tool_calls = parse(parser_cls, tokenizer, f"{REASONING}{THINK_END}{markup}")
    assert tool_calls is None
    assert content == markup


def test_an_unknown_call_beside_a_known_one_is_dropped(tokenizer, parser_cls):
    output = (f"{REASONING}{THINK_END}" + tool_call("get_time", zone="UTC")
              + tool_call("get_weather", city="Paris"))
    _, content, tool_calls = parse(parser_cls, tokenizer, output)
    assert content is None
    assert [(c.name, json.loads(c.arguments)) for c in tool_calls] == [
        ("get_weather", {"city": "Paris"})]


@pytest.mark.parametrize(
    "body, arguments",
    [
        (f"{ARG_KEY_START}city{ARG_KEY_END}Paris", {}),
        (f"{ARG_KEY_START}city{ARG_KEY_END}{ARG_VALUE_START}Paris", {}),
        (f"{ARG_KEY_START}city{ARG_KEY_END}{ARG_VALUE_START}A{ARG_VALUE_END}"
         f"{ARG_KEY_START}city{ARG_KEY_END}{ARG_VALUE_START}B{ARG_VALUE_END}", {"city": "B"}),
    ],
    ids=["value-tags-missing", "value-unterminated", "key-repeated"],
)
def test_malformed_argument_markup_yields_valid_json_arguments(
    tokenizer, parser_cls, body, arguments
):
    """Missing or unterminated value tags lose that argument; a repeated key keeps the
    last value. The arguments field stays a JSON object either way."""
    output = f"{REASONING}{THINK_END}{TOOL_START}get_weather{body}{TOOL_END}"
    (call,) = parse(parser_cls, tokenizer, output)[2]
    assert json.loads(call.arguments) == arguments


def test_a_call_cut_off_by_max_tokens_is_still_returned_without_its_open_argument(
    tokenizer, parser_cls
):
    """No ``</tool_call>``: the parser closes the call at the end of the output and drops
    the argument whose value never closed. ``test_response_shaping`` shows the
    finish_reason the client then receives."""
    output = (f"{REASONING}{THINK_END}{TOOL_START}get_weather{ARG_KEY_START}days"
              f"{ARG_KEY_END}{ARG_VALUE_START}3{ARG_VALUE_END}{ARG_KEY_START}city"
              f"{ARG_KEY_END}{ARG_VALUE_START}Par")
    (call,) = parse(parser_cls, tokenizer, output)[2]
    assert json.loads(call.arguments) == {"days": 3}


# ── tool_choice ─────────────────────────────────────────────────────────────────────


def test_tool_choice_none_returns_the_markup_as_content(tokenizer, parser_cls):
    output = f"{REASONING}{THINK_END}" + tool_call("get_weather", city="Paris")
    reasoning, content, tool_calls = parse(parser_cls, tokenizer, output, tool_choice="none")
    assert reasoning == REASONING
    assert content == tool_call("get_weather", city="Paris")
    assert not tool_calls


def adjusted_constraint(parser_cls, tokenizer, **fields) -> dict | None:
    """The structured-output constraint the parser adds to a request, as a dict."""
    request = make_request(**fields)
    adjusted = make_parser(parser_cls, tokenizer, request).adjust_request(request)
    if adjusted.structured_outputs is None:
        return None
    assert adjusted.response_format is None
    so = adjusted.structured_outputs
    assert so.json is None and so.regex is None and so.grammar is None
    return json.loads(so.structural_tag)["format"]


def test_tool_choice_auto_adds_no_constraint_unless_a_tool_is_strict(tokenizer, parser_cls):
    """With ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` at its default the GLM parser builds an
    xgrammar structural tag in its own markup. For ``auto`` it does so only when a tool
    sets ``strict``; a server without structured outputs then refuses the request."""
    assert parser_cls.tool_parser_cls.structural_tag_model == "glm_4_7"
    assert adjusted_constraint(parser_cls, tokenizer) is None
    assert adjusted_constraint(parser_cls, tokenizer, tool_choice="none") is None
    tag = adjusted_constraint(parser_cls, tokenizer, tools=[strict(WEATHER), PING])
    assert tag["type"] == "triggered_tags" and tag["triggers"] == [TOOL_START]
    assert tag["at_least_one"] is False


def test_tool_choice_required_becomes_a_constraint_in_the_glm_markup(tokenizer, parser_cls):
    """``required`` becomes one or more calls in the template's markup, typed by each
    tool's parameter schema; not the JSON array ``get_json_schema_from_tools`` builds.
    The parser then reads ``required`` like ``auto``."""
    assert parser_cls.tool_parser_cls.supports_required_and_named is False
    tag = adjusted_constraint(parser_cls, tokenizer, tool_choice="required")
    assert tag["type"] == "tags_with_separator" and tag["at_least_one"] is True
    assert [(t["begin"], t["end"], t["content"]["style"], t["content"]["json_schema"])
            for t in tag["tags"]] == [
        (TOOL_START + "get_weather", TOOL_END, "glm_xml", WEATHER["function"]["parameters"]),
        (TOOL_START + "ping", TOOL_END, "glm_xml", PING["function"]["parameters"]),
    ]


def test_a_named_tool_choice_becomes_a_constraint_to_that_tool(tokenizer, parser_cls):
    tag = adjusted_constraint(parser_cls, tokenizer,
                              tool_choice={"type": "function", "function": {"name": "ping"}})
    assert (tag["type"], tag["begin"], tag["end"]) == ("tag", TOOL_START + "ping", TOOL_END)


def test_a_named_tool_choice_is_not_enforced_by_the_parser(tokenizer, parser_cls):
    """Parsing treats a named choice as ``auto``: a call to another tool is returned. Only
    the structured-output constraint, where a server serves it, would hold the model to
    the named tool."""
    output = f"{REASONING}{THINK_END}" + tool_call("get_weather", city="Paris")
    tool_calls = parse(parser_cls, tokenizer, output,
                       tool_choice={"type": "function", "function": {"name": "ping"}})[2]
    assert [c.name for c in tool_calls] == ["get_weather"]


@pytest.mark.xfail(
    strict=True,
    reason="reasoning_effort='none' sets enable_thinking=False, which turns the parser's "
    "reasoning off (vllm/parser/glm47_moe.py:184-191), but this chat template ignores "
    "enable_thinking and always opens <think> (chat_template.jinja:255-257), so the "
    "reasoning and its </think> come back as content",
)
def test_reasoning_effort_none_still_separates_the_reasoning(tokenizer, parser_cls):
    request = make_request(reasoning_effort="none")
    assert template_kwargs(request)["enable_thinking"] is False
    assert tokenizer.decode(prompt_ids(tokenizer, request)[-1:]) == THINK_START
    reasoning, content, _ = parse(parser_cls, tokenizer, f"{REASONING}{THINK_END}Sunny.",
                                  reasoning_effort="none")
    assert (reasoning, content) == (REASONING, "Sunny.")


# ── streaming: where the engine's steps cut the output ─────────────────────────────


def _check_stream(parser_cls, tokenizer, output: str, sizes, **fields) -> dict:
    ids = encode(tokenizer, output)
    streamed = assemble(stream(parser_cls, tokenizer, ids, sizes, **fields))
    assert_stream_matches_parse(streamed, *parse(parser_cls, tokenizer, output, **fields))
    return streamed


def test_streaming_one_token_per_step_matches_the_whole_response(tokenizer, parser_cls):
    streamed = _check_stream(parser_cls, tokenizer, TWO_CALLS, [1])
    assert len(streamed["calls"]) == 2


@pytest.mark.parametrize("sizes", [[2], [3], [4], [4, 1, 3, 2]], ids=str)
def test_streaming_multi_token_steps_matches_the_whole_response(tokenizer, parser_cls, sizes):
    """Speculative decoding with three draft tokens hands the detokenizer up to four
    tokens per step, so step boundaries fall anywhere in the markup."""
    _check_stream(parser_cls, tokenizer, TWO_CALLS, sizes)


def test_streaming_the_whole_output_in_one_finishing_step(tokenizer, parser_cls):
    ids = encode(tokenizer, TWO_CALLS)
    _check_stream(parser_cls, tokenizer, TWO_CALLS, [len(ids)])


def test_streaming_end_of_reasoning_and_call_start_in_one_step(tokenizer, parser_cls):
    """``</think>`` and ``<tool_call>`` land in the same engine step."""
    ids = encode(tokenizer, TWO_CALLS)
    cut = ids.index(encode(tokenizer, THINK_END)[0])
    sizes = [cut] + [2] + [1] * len(ids)
    streamed = _check_stream(parser_cls, tokenizer, TWO_CALLS, sizes)
    assert streamed["reasoning"] == REASONING


@pytest.mark.parametrize("together", [True, False], ids=["one-step", "two-steps"])
def test_streaming_the_boundary_between_two_calls(tokenizer, parser_cls, together):
    """``</tool_call><tool_call>`` in one step, or the step boundary between them; the
    second call gets index 1 and its own id, each id sent once."""
    ids = encode(tokenizer, TWO_CALLS)
    end = ids.index(encode(tokenizer, TOOL_END)[0])
    sizes = [end, 2] if together else [end + 1, 1]
    streamed = _check_stream(parser_cls, tokenizer, TWO_CALLS, sizes + [1] * len(ids))
    assert sorted(streamed["calls"]) == [0, 1]


def test_streaming_a_string_argument_cut_inside_its_json_string(tokenizer, parser_cls):
    """The value needs JSON escapes and is cut by the tokenizer (and inside a multi-byte
    character); every argument prefix a client holds is a prefix of the final JSON."""
    ids = encode(tokenizer, QUOTED_CALL)
    deltas = stream(parser_cls, tokenizer, ids, [1])
    (call,) = parse(parser_cls, tokenizer, QUOTED_CALL)[2]
    held = ""
    for delta in deltas:
        for part in delta.tool_calls or []:
            held += (part.function.arguments or "") if part.function else ""
            assert call.arguments.startswith(held), (held, call.arguments)
    assert json.loads(held) == {"city": QUOTED, "days": 12345}


def test_streaming_holds_a_typed_value_back_until_it_closes(tokenizer, parser_cls):
    """``days`` is an integer: no step sends part of it, so a client never sees a number
    that later changes."""
    ids = encode(tokenizer, QUOTED_CALL)
    pieces = [part.function.arguments for delta in stream(parser_cls, tokenizer, ids, [1])
              for part in delta.tool_calls or [] if part.function and part.function.arguments]
    assert [piece for piece in pieces if "1234" in piece] == ['12345}']


def test_streaming_random_step_sizes_match_the_whole_response(tokenizer, parser_cls):
    """Seeded random step sizes from one to five tokens over reasoning, content and two
    calls."""
    output = f"{REASONING}{THINK_END}Checking two cities.\n" + TWO_CALLS.split(THINK_END, 1)[1]
    rng = random.Random(13)
    for _ in range(25):
        _check_stream(parser_cls, tokenizer, output, [rng.randint(1, 5) for _ in range(64)])


def test_streaming_markup_spelled_in_ordinary_tokens_is_content(tokenizer, parser_cls):
    """``<tool_call>`` written as ordinary tokens (``<tool`` + ``_call>``) is text, not the
    tag token: a stream treats it as content, as a whole response does."""
    head = encode(tokenizer, f"{REASONING}{THINK_END}")
    spelled = encode(tokenizer, "<tool") + encode(tokenizer, "_call>")
    assert len(spelled) > 1 and encode(tokenizer, TOOL_START)[0] not in spelled
    ids = head + spelled + encode(tokenizer, "get_weather")
    streamed = assemble(stream(parser_cls, tokenizer, ids, [1]))
    assert streamed["calls"] == {}
    assert streamed["content"] == "<tool_call>get_weather"


def test_streaming_tool_choice_none_returns_the_markup_as_content(tokenizer, parser_cls):
    streamed = _check_stream(parser_cls, tokenizer, TWO_CALLS, [1], tool_choice="none")
    assert streamed["content"] == TWO_CALLS.split(THINK_END, 1)[1]


@pytest.mark.xfail(
    strict=True,
    reason="a stream drops a call to an unknown tool without a trace (no content, no "
    "tool_calls) while the whole response returns its markup as content: the stream "
    "never names the slot (parser_engine.py:781) and its arguments are never flushed",
)
def test_streaming_a_call_to_an_unknown_tool_matches_the_whole_response(tokenizer, parser_cls):
    _check_stream(parser_cls, tokenizer,
                  f"{REASONING}{THINK_END}" + tool_call("get_time", zone="UTC"), [1])


@pytest.mark.xfail(
    strict=True,
    reason="a stream cut off inside an argument value sends the open value "
    "('{\"city\": \"Par'), so the arguments a client holds are not JSON, while the "
    "whole response drops the open argument and returns '{}'",
)
def test_streaming_a_call_cut_off_by_max_tokens_sends_json_arguments(tokenizer, parser_cls):
    output = (f"{REASONING}{THINK_END}{TOOL_START}get_weather{ARG_KEY_START}city"
              f"{ARG_KEY_END}{ARG_VALUE_START}Par")
    streamed = assemble(stream(parser_cls, tokenizer, encode(tokenizer, output), [1]))
    json.loads(streamed["calls"][0]["arguments"])


@pytest.mark.xfail(
    strict=True,
    reason="text after the last call is sent by a stream but dropped by the whole "
    "response: it is deferred behind the call (parser_engine.py:697-699) and "
    "_single_pass_parse never flushes the deferred text (parser_engine.py:633-649)",
)
def test_text_after_a_call_matches_between_stream_and_whole_response(tokenizer, parser_cls):
    _check_stream(parser_cls, tokenizer, f"{REASONING}{THINK_END}"
                  + tool_call("get_weather", city="Paris") + "\nDone.", [1])
