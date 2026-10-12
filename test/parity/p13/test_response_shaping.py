# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash chat responses over HTTP: what each request shape gets back, on CPU.

The app is vLLM's ``build_app`` + ``init_app_state`` for the served command line, driven
over HTTP by starlette's ``TestClient``, with this plugin's platform admission in place.
Its engine client is vLLM's ``AsyncLLM`` (renderer, ``InputProcessor``,
``OutputProcessor``, detokenizer) with only the engine-core socket replaced: the stand-in
records each request and plays back the token ids a test plans for it, a few per step as
speculative decoding hands them over. So every byte of each response is built by vLLM's
serving code from real token ids; only the model is absent.

Two servers are built, both from the served flags (tensor parallel 64 with expert
parallel, ``max_model_len`` 4096, one sequence, async scheduling, MTP with three draft
tokens, on-device greedy sampling, structured outputs off):

* ``served``: those flags alone, with no parser flags.
* ``parsers``: the same plus ``--enable-auto-tool-choice --tool-call-parser glm47
  --reasoning-parser glm45``.

Requests a server cannot honour must be refused with HTTP 400 before they reach the
engine; requests it admits must come back in the OpenAI shape.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest
from fastapi.testclient import TestClient
from glm_chat import (
    MESSAGES,
    OBSERVATION,
    PING,
    REASONING,
    THINK_END,
    TOOL_CALL_ID_PREFIX,
    TOOLS,
    WEATHER,
    encode,
    strict,
    tool_call,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

from vllm_neuron.vllm.platform import SO_DISABLED_MESSAGE, NeuronPlatform

from test.vllm_neuron.vllm.test_admission import KNOB, SERVED_BLOCK, STANDARD_LINE
from test.vllm_neuron.vllm.test_admission_http import TASKS, _AsyncLLM, _EngineCore

pytestmark = [pytest.mark.fast]

#: Tokens per engine step: one target token plus three accepted drafts.
STEP = 4
PARSER_FLAGS = ["--enable-auto-tool-choice", "--tool-call-parser", "glm47",
                "--reasoning-parser", "glm45"]


class _PlannedCore(_EngineCore):
    """Plays back, for each request, the next planned ``(token ids, finish reason)``.

    A turn that ends on an EOS id other than the tokenizer's own (``<|observation|>``,
    listed in ``generation_config.json``) finishes ``stop`` with that id as its stop
    reason, as the scheduler reports it."""

    def __init__(self) -> None:
        super().__init__()
        self.plans: list[tuple[list[int], FinishReason, int | None]] = []

    async def add_request_async(self, request) -> None:
        self.requests.append(request)
        ids, finish, stop_reason = self.plans.pop(0)
        steps = [ids[i:i + STEP] for i in range(0, len(ids), STEP)]
        for number, step in enumerate(steps, 1):
            last = number == len(steps)
            await self._queue().put(EngineCoreOutputs(outputs=[EngineCoreOutput(
                request_id=request.request_id, new_token_ids=step,
                finish_reason=finish if last else None,
                stop_reason=stop_reason if last else None)]))


class Server:
    def __init__(self, client, core, tokenizer, model, has_parsers: bool):
        self.client, self.core, self.tokenizer, self.model = client, core, tokenizer, model
        self.has_parsers = has_parsers

    def chat(self, output: str | None = None, *, finish=FinishReason.STOP, **body):
        """POST a chat request; the engine answers ``output`` and, for a ``stop``
        finish, the ``<|observation|>`` EOS id."""
        if output is not None:
            ids = encode(self.tokenizer, output)
            stop_reason = None
            if finish == FinishReason.STOP:
                (stop_reason,) = encode(self.tokenizer, OBSERVATION)
                ids.append(stop_reason)
            self.core.plans.append((ids, finish, stop_reason))
        body.setdefault("messages", MESSAGES)
        body.setdefault("max_tokens", 256)
        return self.client.post("/v1/chat/completions", json=dict(model=self.model, **body))

    def complete(self, **body):
        return self.client.post("/v1/completions", json=dict(
            model=self.model, prompt="Hello", max_tokens=8, **body))

    def refuses(self, *fragments: str, endpoint: str = "chat", **body) -> str:
        """POST a request that must be refused: HTTP 400 naming ``fragments``, and the
        engine never handed it."""
        reached = len(self.core.requests)
        response = (self.chat if endpoint == "chat" else self.complete)(**body)
        assert response.status_code == 400, response.text
        message = response.json()["error"]["message"]
        for fragment in fragments:
            assert fragment in message, message
        assert len(self.core.requests) == reached, "a refused request reached the engine"
        return message


def _cli(model_dir, *extra) -> list[str]:
    """The served command line. ``--enforce-eager`` and ``--no-enable-prefix-caching``
    are added only so no engine resources are planned on CPU; neither changes how a
    request is validated, rendered or shaped."""
    line = STANDARD_LINE
    neuron_config = dict(line["neuron_config"], on_device_sampling_config={"all_greedy": True})
    return [
        "--model", str(model_dir),
        "--max-model-len", str(line["max_model_len"]),
        "--max-num-seqs", str(line["max_num_seqs"]),
        "--max-num-batched-tokens", str(line["max_num_batched_tokens"]),
        "--tensor-parallel-size", "64", "--enable-expert-parallel",
        "--block-size", str(SERVED_BLOCK), "--async-scheduling",
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3}),
        "--additional-config", json.dumps({"neuron_config": neuron_config}),
        "--enforce-eager", "--no-enable-prefix-caching", *extra,
    ]


def _server(model_dir, monkeypatch, *extra) -> Server:
    monkeypatch.setenv(KNOB, "1")
    for name in ("_admission", "_enable_structured_outputs"):
        monkeypatch.setattr(NeuronPlatform, name, getattr(NeuronPlatform, name))
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(_cli(model_dir, *extra))
    config = AsyncEngineArgs.from_cli_args(args).create_engine_config()
    engine = _AsyncLLM(config)
    engine.engine_core = _PlannedCore()
    app = build_app(args, TASKS, config.model_config)
    asyncio.run(init_app_state(engine, app.state, args, TASKS))
    client = TestClient(app, raise_server_exceptions=False)
    return Server(client, engine.engine_core, engine.renderer.tokenizer, str(model_dir),
                  has_parsers="--tool-call-parser" in extra)


@pytest.fixture(scope="module")
def served(checkpoint_dir):
    with pytest.MonkeyPatch.context() as monkeypatch:
        server = _server(checkpoint_dir, monkeypatch)
        with server.client:
            yield server


@pytest.fixture(scope="module")
def parsers(checkpoint_dir):
    with pytest.MonkeyPatch.context() as monkeypatch:
        server = _server(checkpoint_dir, monkeypatch, *PARSER_FLAGS)
        with server.client:
            yield server


@pytest.fixture(params=["served", "parsers"])
def server(request):
    """Each of the two servers in turn, for behaviour the parser flags do not change."""
    return request.getfixturevalue(request.param)


@pytest.fixture(autouse=True)
def _no_leftover_plans(request):
    """Every planned answer is consumed: a refused request must not reach the engine."""
    yield
    for name in ("served", "parsers"):
        if name in request.fixturenames:
            server = request.getfixturevalue(name)
            leftover, server.core.plans[:] = list(server.core.plans), []
            assert leftover == [], "a planned answer was not consumed"


def events(response) -> list[dict]:
    assert response.status_code == 200, response.text
    lines = [line[len("data: "):] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]", lines[-3:]
    return [json.loads(line) for line in lines[:-1]]


def streamed_message(chunks: list[dict]) -> dict:
    """What a client accumulates from a stream's chunks, by field."""
    out = {"role": None, "reasoning": "", "content": "", "calls": {}, "finish_reason": None}
    for chunk in chunks:
        for choice in chunk["choices"]:
            delta = choice["delta"]
            out["role"] = out["role"] or delta.get("role")
            out["reasoning"] += delta.get("reasoning") or ""
            out["content"] += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                slot = out["calls"].setdefault(call["index"], {"ids": [], "name": "", "arguments": ""})
                if call.get("id"):
                    slot["ids"].append(call["id"])
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                assert out["finish_reason"] is None, chunks
                out["finish_reason"] = choice["finish_reason"]
    return out


def test_the_plugin_under_test_is_this_trees(request):
    """``python -I`` drops ``PYTHONPATH``; pytest's rootdir insertion must still make
    ``vllm_neuron`` this checkout's, not another installed copy."""
    import vllm_neuron

    root = request.config.rootpath.resolve()
    assert pathlib.Path(vllm_neuron.__file__).resolve().is_relative_to(root), vllm_neuron.__file__


ONE_CALL = f"{REASONING}{THINK_END}" + tool_call("get_weather", city="Paris", days="3")
ANSWER = f"{REASONING}{THINK_END}It is sunny in Paris."
NAMED_PING = {"type": "function", "function": {"name": "ping"}}


# ── served: the served flags, no parser flags ──────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_served_refuses_tools_with_the_default_auto_choice(served, stream):
    served.refuses('"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser',
                   tools=TOOLS, stream=stream)


@pytest.mark.parametrize("choice", ["required", NAMED_PING], ids=["required", "named"])
def test_served_refuses_a_forced_tool_choice(served, choice):
    served.refuses("requires --tool-call-parser to be set", tools=TOOLS, tool_choice=choice)


def test_served_returns_reasoning_and_tool_markup_inside_content(served):
    """With ``tool_choice="none"`` the tools are admitted and only describe themselves in
    the prompt; with no parser the answer, reasoning, ``</think>`` and any call markup
    included, is the content."""
    response = served.chat(ONE_CALL, tools=TOOLS, tool_choice="none")
    assert response.status_code == 200, response.text
    (choice,) = response.json()["choices"]
    assert choice["message"]["content"] == ONE_CALL
    assert choice["message"].get("reasoning") is None
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_served_returns_reasoning_inside_content_without_tools(served, stream):
    response = served.chat(ANSWER, stream=stream)
    if stream:
        message = streamed_message(events(response))
        assert (message["role"], message["reasoning"]) == ("assistant", "")
    else:
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message.get("reasoning") is None
    assert message["content"] == ANSWER
    assert message["finish_reason" if stream else "role"] == ("stop" if stream else "assistant")


@pytest.mark.xfail(
    strict=True,
    reason="the chat template never reads enable_thinking and renders any "
    "reasoning_effort other than low/high as 'Max' (chat_template.jinja:2-3); its "
    "generation prompt always opens <think> (:255-256), so asking for no reasoning "
    "renders the same prompt as the default and is admitted, then ignored",
)
@pytest.mark.parametrize("fields", [
    {"reasoning_effort": "none"},
    {"chat_template_kwargs": {"enable_thinking": False}},
], ids=["reasoning_effort-none", "enable_thinking-false"])
def test_turning_reasoning_off_closes_the_reasoning_in_the_prompt(server, fields):
    server.chat(ANSWER, **fields)
    prompt = server.tokenizer.decode(server.core.requests[-1].prompt_token_ids)
    assert prompt.endswith("<|assistant|><think></think>"), prompt[-80:]


# ── structured outputs are off on both servers ─────────────────────────────────────


SO_REQUESTS = {
    "json_object": {"response_format": {"type": "json_object"}},
    "json_schema": {"response_format": {"type": "json_schema", "json_schema": {
        "name": "forecast", "schema": {"type": "object", "properties": {"t": {"type": "number"}}}}}},
    "regex": {"structured_outputs": {"regex": "[a-z]+"}},
    "choice": {"structured_outputs": {"choice": ["yes", "no"]}},
}


@pytest.mark.parametrize("body", SO_REQUESTS.values(), ids=SO_REQUESTS.keys())
@pytest.mark.parametrize("endpoint", ["chat", "completions"])
def test_structured_output_requests_are_refused(server, endpoint, body):
    server.refuses(SO_DISABLED_MESSAGE, endpoint=endpoint, **body)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
@pytest.mark.parametrize("fields", [
    {"tool_choice": "required"},
    {"tool_choice": NAMED_PING},
    {"tools": [strict(WEATHER), PING]},
], ids=["required", "named", "auto-strict-tool"])
def test_parsers_refuse_tool_choices_that_need_a_constraint(parsers, fields, stream):
    """``required``, a named tool, and ``auto`` with a ``strict`` tool are served by a
    constraint in the tool markup (see ``test_tool_grammar``); structured outputs are
    off, so the request is refused rather than run unconstrained."""
    parsers.refuses(SO_DISABLED_MESSAGE, **{"tools": TOOLS, **fields}, stream=stream)


# ── sampling the on-device greedy sampler cannot serve ─────────────────────────────


@pytest.mark.parametrize("body, knob", [
    ({"n": 2}, "n=2"),
    ({"temperature": 0.7}, "temperature=0.7"),
    ({"logprobs": True, "top_logprobs": 2}, "logprobs"),
    ({"logprobs": True}, "logprobs"),
], ids=["n", "temperature", "top_logprobs", "logprobs"])
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_chat_sampling_the_sampler_cannot_serve_is_refused(server, body, knob, stream):
    """Platform admission refuses what the on-device greedy sampler cannot apply."""
    server.refuses(knob, stream=stream, **body)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_several_greedy_choices_are_refused_by_vllm(server, stream):
    """At ``temperature: 0`` vLLM itself refuses ``n > 1`` before the platform does."""
    server.refuses("n must be 1 when using greedy sampling, got 2", n=2, temperature=0, stream=stream)


LOGIT_BIAS = {"logit_bias": {"1000": -100}}
SPEC_DECODE_REFUSAL = "not yet supported with speculative decoding"


def test_logit_bias_is_refused_under_speculative_decoding(server):
    server.refuses(SPEC_DECODE_REFUSAL, **LOGIT_BIAS)


def test_a_streamed_logit_bias_is_refused_inside_the_stream(server):
    """vLLM validates ``logit_bias`` against speculative decoding in ``process_inputs``,
    after the stream has started, so the refusal arrives as an error event."""
    reached = len(server.core.requests)
    response = server.chat(stream=True, **LOGIT_BIAS)
    assert response.status_code == 200, response.text
    (error,) = [chunk["error"] for chunk in events(response) if "error" in chunk]
    assert error["code"] == 400 and SPEC_DECODE_REFUSAL in error["message"], error
    assert len(server.core.requests) == reached


@pytest.mark.xfail(
    strict=True,
    reason="the speculative-decoding check on logit_bias runs in vLLM's "
    "InputProcessor.process_inputs (vllm/sampling_params.py:848-860), inside the stream "
    "generator, and the platform's eager validation does not cover it, so a streamed "
    "request gets 200 and an error event instead of the 400 the other refusals get",
)
def test_a_streamed_logit_bias_is_a_400(server):
    server.refuses(SPEC_DECODE_REFUSAL, stream=True, **LOGIT_BIAS)


@pytest.mark.xfail(
    strict=True,
    reason="admission refuses only temperature/top_k/top_p/min_p/seed/n and logprobs "
    "(vllm_neuron/vllm/admission.py), while the async on-device path returns the device "
    "tokens without vLLM's Sampler (neuron_model_runner.py:11359-11366) and the input "
    "batch has no logits processors (neuron_model_runner.py:12097), so penalties, "
    "bad_words, allowed_token_ids and min_tokens are admitted and ignored",
)
@pytest.mark.parametrize("body", [
    {"presence_penalty": 1.0},
    {"frequency_penalty": 1.0},
    {"repetition_penalty": 1.2},
    {"min_tokens": 4},
    {"bad_words": ["Paris"]},
    {"allowed_token_ids": [1000, 1001]},
], ids=lambda body: next(iter(body)))
def test_sampling_controls_the_device_sampler_ignores_are_refused(server, body):
    server.core.plans.append((encode(server.tokenizer, ANSWER), FinishReason.LENGTH, None))
    try:
        server.refuses(**body)
    finally:
        server.core.plans.clear()


# ── parsers: tool calls and reasoning in the OpenAI shape ──────────────────────────


def test_a_tool_call_comes_back_as_tool_calls_with_its_reasoning(parsers):
    response = parsers.chat(ONE_CALL, tools=TOOLS)
    assert response.status_code == 200, response.text
    body = response.json()
    (choice,) = body["choices"]
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["reasoning"] == REASONING
    (call,) = message["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith(TOOL_CALL_ID_PREFIX)
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert choice["finish_reason"] == "tool_calls"
    (request,) = parsers.core.requests[-1:]
    generated = len(encode(parsers.tokenizer, ONE_CALL)) + 1
    assert body["usage"] == {
        "prompt_tokens": len(request.prompt_token_ids), "completion_tokens": generated,
        "total_tokens": len(request.prompt_token_ids) + generated, "prompt_tokens_details": None,
    }


def test_two_calls_get_their_own_ids_in_order(parsers):
    output = ONE_CALL + tool_call("ping")
    message = parsers.chat(output, tools=TOOLS).json()["choices"][0]["message"]
    calls = message["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_weather", "ping"]
    assert len({c["id"] for c in calls}) == 2


def test_an_answer_without_a_call_finishes_stop_with_reasoning_split(parsers):
    (choice,) = parsers.chat(ANSWER, tools=TOOLS).json()["choices"]
    assert choice["message"]["reasoning"] == REASONING
    assert choice["message"]["content"] == "It is sunny in Paris."
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == "stop"


def test_tool_choice_none_returns_the_call_markup_as_content(parsers):
    (choice,) = parsers.chat(ONE_CALL, tools=TOOLS, tool_choice="none").json()["choices"]
    assert choice["message"]["reasoning"] == REASONING
    assert choice["message"]["content"] == ONE_CALL.split(THINK_END, 1)[1]
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == "stop"


def test_include_reasoning_false_leaves_the_reasoning_out(parsers):
    message = parsers.chat(ANSWER, include_reasoning=False).json()["choices"][0]["message"]
    assert message.get("reasoning") is None
    assert message["content"] == "It is sunny in Paris."


@pytest.mark.parametrize("effort, rendered", [("low", "Low"), ("high", "High"), ("medium", "Max")])
def test_reasoning_effort_reaches_the_prompt_as_the_template_spells_it(server, effort, rendered):
    """The template knows ``low`` and ``high``; any other effort renders ``Max``. The
    parsers split the reasoning out; the served form returns it in the content."""
    message = server.chat(ANSWER, reasoning_effort=effort).json()["choices"][0]["message"]
    prompt = server.tokenizer.decode(server.core.requests[-1].prompt_token_ids)
    assert f"Reasoning Effort: {rendered}" in prompt
    split = (message.get("reasoning"), message["content"])
    assert split == ((REASONING, "It is sunny in Paris.") if server.has_parsers else (None, ANSWER))


def test_a_streamed_tool_call_arrives_in_the_openai_chunk_shape(parsers):
    chunks = events(parsers.chat(ONE_CALL, tools=TOOLS, stream=True,
                                 stream_options={"include_usage": True}))
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    *body, usage = chunks
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == len(
        encode(parsers.tokenizer, ONE_CALL)) + 1
    assert all(chunk["id"] == chunks[0]["id"] for chunk in chunks)
    message = streamed_message(body)
    assert message["reasoning"] == REASONING
    assert message["content"].strip() == ""
    assert message["finish_reason"] == "tool_calls"
    (call,) = message["calls"].values()
    assert len(call["ids"]) == 1 and call["ids"][0].startswith(TOOL_CALL_ID_PREFIX)
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": "Paris", "days": 3}


def test_streamed_calls_match_the_whole_response(parsers):
    output = ONE_CALL + tool_call("ping")
    whole = parsers.chat(output, tools=TOOLS).json()["choices"][0]["message"]["tool_calls"]
    streamed = streamed_message(events(parsers.chat(output, tools=TOOLS, stream=True)))
    assert sorted(streamed["calls"]) == [0, 1]
    assert [(streamed["calls"][i]["name"], json.loads(streamed["calls"][i]["arguments"]))
            for i in (0, 1)] == [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                                 for c in whole]


@pytest.mark.xfail(
    strict=True,
    reason="a call cut off by max_tokens finishes 'tool_calls' instead of 'length': "
    "OpenAIServingChat reports tool_calls whenever a call was parsed, whatever the "
    "engine's finish reason (vllm/entrypoints/openai/chat_completion/serving.py:702-703 "
    "streamed, :989-993 and :1009-1011 whole), so the client cannot tell the arguments "
    "are incomplete",
)
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_a_call_cut_off_by_max_tokens_finishes_length(parsers, stream):
    output = (f"{REASONING}{THINK_END}" + tool_call("get_weather", days="3")[:-len("</tool_call>")]
              + "<arg_key>city</arg_key><arg_value>Par")
    response = parsers.chat(output, finish=FinishReason.LENGTH, tools=TOOLS, stream=stream)
    if stream:
        finish = streamed_message(events(response))["finish_reason"]
    else:
        finish = response.json()["choices"][0]["finish_reason"]
    assert finish == "length"
