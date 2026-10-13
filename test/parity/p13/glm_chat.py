# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash chat markup, tools and parser calls shared by this directory's tests.

The markup is the checkpoint's own (``chat_template.jinja``); the parser is vLLM's,
built by the ``parser_cls`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

#: The checkpoint's markup, in the order its tokenizer lists them.
THINK_START, THINK_END = "<think>", "</think>"
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
ARG_KEY_START, ARG_KEY_END = "<arg_key>", "</arg_key>"
ARG_VALUE_START, ARG_VALUE_END = "<arg_value>", "</arg_value>"
MARKUP = (
    THINK_START, THINK_END, TOOL_START, TOOL_END,
    ARG_KEY_START, ARG_KEY_END, ARG_VALUE_START, ARG_VALUE_END,
)
#: The token the template writes after an assistant turn that calls tools.
OBSERVATION = "<|observation|>"
#: The prefix vLLM's ``make_tool_call_id`` gives a random tool-call id.
TOOL_CALL_ID_PREFIX = "chatcmpl-tool-"

WEATHER_PROPERTIES = {
    "city": {"type": "string"},
    "days": {"type": "integer"},
    "metric": {"type": "boolean"},
    "opts": {"type": "object"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "ratio": {"type": "number"},
    "note": {"type": ["string", "null"]},
}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather forecast for a city.",
        "parameters": {"type": "object", "properties": WEATHER_PROPERTIES,
                       "required": ["city"]},
    },
}
PING = {
    "type": "function",
    "function": {"name": "ping", "description": "Liveness check.",
                 "parameters": {"type": "object", "properties": {}}},
}
TOOLS = [WEATHER, PING]
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]


def tool_call(name: str, **arguments: str) -> str:
    """One call in the markup the chat template writes (no whitespace between parts)."""
    body = "".join(
        f"{ARG_KEY_START}{key}{ARG_KEY_END}{ARG_VALUE_START}{value}{ARG_VALUE_END}"
        for key, value in arguments.items()
    )
    return f"{TOOL_START}{name}{body}{TOOL_END}"


def make_request(**fields) -> ChatCompletionRequest:
    fields.setdefault("messages", MESSAGES)
    fields.setdefault("tools", TOOLS)
    return ChatCompletionRequest(model="glm", **fields)


def template_kwargs(request: ChatCompletionRequest) -> dict:
    """The ``chat_template_kwargs`` ``OpenAIServingChat`` hands the parser."""
    return request.build_chat_params(None, "auto").with_defaults({}).chat_template_kwargs


def make_parser(parser_cls, tokenizer, request: ChatCompletionRequest):
    return parser_cls(tokenizer, request.tools, chat_template_kwargs=template_kwargs(request))


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def prompt_ids(tokenizer, request: ChatCompletionRequest) -> list[int]:
    """The generation prompt the chat template renders for ``request`` (its template
    kwargs carry ``add_generation_prompt=True``)."""
    tools = [tool.model_dump() for tool in request.tools] if request.tools else None
    text = tokenizer.apply_chat_template(
        request.messages, tools=tools, tokenize=False, **template_kwargs(request))
    return encode(tokenizer, text)


def parse(parser_cls, tokenizer, output: str, **fields):
    """``(reasoning, content, tool_calls)`` as the non-streamed chat response builds it."""
    request = make_request(**fields)
    parser = make_parser(parser_cls, tokenizer, request)
    request = parser.adjust_request(request)
    return parser.parse(output, request, enable_auto_tools=True,
                        model_output_token_ids=encode(tokenizer, output))


def strict(tool: dict) -> dict:
    """``tool`` with ``function.strict`` set."""
    return {**tool, "function": {**tool["function"], "strict": True}}


#: Reasoning text used by several tests' constructed outputs.
REASONING = "The user wants the forecast, so I call the weather tool."
