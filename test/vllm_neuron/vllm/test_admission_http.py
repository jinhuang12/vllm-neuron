# SPDX-License-Identifier: Apache-2.0
"""Refusals reach the client as HTTP 400 from vLLM's own OpenAI server, and the engine never sees them.

The app is vLLM's ``build_app`` + ``init_app_state``, driven over HTTP by starlette's
``TestClient``. Its engine client is vLLM's real ``AsyncLLM`` (``generate``,
``add_request``, ``InputProcessor.process_inputs``, ``OutputProcessor``) with one part
replaced: the engine-core client, which in a serve is the socket to the EngineCore
process. The stand-in records every request handed to it and answers each with three
greedy tokens, so "the engine was not touched" is a count of what crossed that socket.

Before this change the same harness returns 200 for a prompt past the window and for
``top_k=50`` (both reach the engine), 500 ``list index out of range`` for ``logprobs=5``
(the gate's server.log), and 200 with an error event for a streamed refusal.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD python -m pytest \\
        test/vllm_neuron/vllm/test_admission_http.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.renderers import renderer_from_config
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor

from vllm_neuron.vllm.platform import NeuronPlatform

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.vllm.test_admission import (
    KNOB,
    SERVED_BLOCK,
    SERVED_GENERATION_CONFIG,
    OLD_WINDOW,
    STANDARD_LINE,
    STANDARD_MAX,
)

pytestmark = [pytest.mark.fast]

# What the stand-in engine returns for every request it is handed.
ENGINE_TOKENS = [7, 8, 9]
TASKS = ("generate",)


class _EngineCore:
    """The engine-core client's surface AsyncLLM uses, with a record of what reached it."""

    class resources:
        engine_dead = False

    def __init__(self) -> None:
        self.requests: list = []
        self._outputs: asyncio.Queue | None = None

    def _queue(self) -> asyncio.Queue:
        if self._outputs is None:
            self._outputs = asyncio.Queue()
        return self._outputs

    async def add_request_async(self, request) -> None:
        self.requests.append(request)
        await self._queue().put(EngineCoreOutputs(outputs=[EngineCoreOutput(
            request_id=request.request_id, new_token_ids=list(ENGINE_TOKENS),
            finish_reason=FinishReason.LENGTH)]))

    async def get_output_async(self) -> EngineCoreOutputs:
        return await self._queue().get()

    async def abort_requests_async(self, request_ids) -> None:
        pass

    async def get_supported_tasks_async(self):
        return TASKS

    def shutdown(self, timeout=None) -> None:
        pass


class _AsyncLLM(AsyncLLM):
    """AsyncLLM as ``__init__`` builds it, minus the EngineCore process."""

    def __init__(self, vllm_config) -> None:  # noqa: D107 - mirrors AsyncLLM.__init__
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = True
        self.log_stats = False
        self.renderer = renderer_from_config(vllm_config)
        self.input_processor = InputProcessor(vllm_config, self.renderer)
        self.output_processor = OutputProcessor(
            self.renderer.tokenizer, log_stats=False,
            stream_interval=vllm_config.scheduler_config.stream_interval,
            tracing_enabled=False,
        )
        self.engine_core = _EngineCore()
        self.logger_manager = None
        self._client_count = 1
        self.output_handler = None
        self.profiler = None


def _cli(model_dir: pathlib.Path, line: dict, sampler) -> list[str]:
    neuron_config = dict(line["neuron_config"], on_device_sampling_config=sampler)
    async_flag = "--async-scheduling" if sampler is not None else "--no-async-scheduling"
    return [
        "--model", str(model_dir), "--skip-tokenizer-init",
        "--max-model-len", str(line["max_model_len"]),
        "--max-num-seqs", str(line["max_num_seqs"]),
        "--max-num-batched-tokens", str(line["max_num_batched_tokens"]),
        "--block-size", str(SERVED_BLOCK), "--enforce-eager", "--no-enable-prefix-caching",
        async_flag, "--additional-config", json.dumps({"neuron_config": neuron_config}),
    ]


@pytest.fixture(scope="module")
def served_model_dir():
    root = pathlib.Path(tempfile.mkdtemp(prefix="glm53f-admission-http-"))
    shutil.copy(fr.FIXTURE / "config.json", root / "config.json")
    (root / "generation_config.json").write_text(json.dumps(SERVED_GENERATION_CONFIG))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _server(model_dir, monkeypatch, sampler):
    """vLLM's OpenAI app for the standard line; yields (client, engine core stand-in)."""
    monkeypatch.setenv(KNOB, "1")
    monkeypatch.setattr(NeuronPlatform, "_admission", NeuronPlatform._admission)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        _cli(model_dir, STANDARD_LINE, sampler))
    config = AsyncEngineArgs.from_cli_args(args).create_engine_config()
    engine = _AsyncLLM(config)
    app = build_app(args, TASKS, config.model_config)
    asyncio.run(init_app_state(engine, app.state, args, TASKS))
    return TestClient(app, raise_server_exceptions=False), engine.engine_core, str(model_dir)


@pytest.fixture
def greedy_server(served_model_dir, monkeypatch):
    client, core, model = _server(served_model_dir, monkeypatch, {"all_greedy": True})
    with client:
        yield client, core, model


@pytest.fixture
def host_server(served_model_dir, monkeypatch):
    client, core, model = _server(served_model_dir, monkeypatch, None)
    with client:
        yield client, core, model


def _complete(server, prompt_tokens: int, **body):
    client, _, model = server
    return client.post("/v1/completions", json=dict(
        model=model, prompt=[11] * prompt_tokens, max_tokens=3, **body))


def _error(response) -> str:
    return response.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_a_prompt_past_max_model_len_is_a_400_the_engine_never_sees(greedy_server, stream):
    """A 4096-token prompt plus the requested tokens exceeds max_model_len 4096."""
    response = _complete(greedy_server, STANDARD_MAX, temperature=0.0, stream=stream)
    assert response.status_code == 400, response.text
    message = _error(response)
    assert str(STANDARD_MAX) in message, message
    assert greedy_server[1].requests == []


@pytest.mark.parametrize("tokens", [OLD_WINDOW + 1, 3000, STANDARD_MAX - 3])
def test_a_prompt_past_the_old_2048_token_window_reaches_the_engine(greedy_server, tokens):
    """Before this change the standard line answered 400 here (window 2048 < 4096)."""
    response = _complete(greedy_server, tokens, temperature=0.0, return_token_ids=True)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["token_ids"] == ENGINE_TOKENS
    assert len(greedy_server[1].requests) == 1
    assert len(greedy_server[1].requests[0].prompt_token_ids) == tokens


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_logprobs_under_on_device_sampling_are_a_400_not_a_500(greedy_server, stream):
    response = _complete(greedy_server, 10, temperature=0.0, logprobs=5, stream=stream)
    assert response.status_code == 400, response.text
    assert "logprobs=5" in _error(response)
    assert greedy_server[1].requests == []


def test_top_k_under_all_greedy_is_a_400_the_engine_never_sees(greedy_server):
    response = _complete(greedy_server, 10, top_k=50)
    assert response.status_code == 400, response.text
    assert "top_k=50" in _error(response)
    assert greedy_server[1].requests == []


def test_a_valid_greedy_request_after_each_refusal_is_served(greedy_server):
    """Refused, refused, refused, then served: the engine sees exactly the one request it
    can serve, and returns its tokens."""
    for body in ({"prompt_tokens": STANDARD_MAX, "temperature": 0.0},
                 {"prompt_tokens": 10, "top_k": 50},
                 {"prompt_tokens": 10, "temperature": 0.0, "logprobs": 5}):
        tokens = body.pop("prompt_tokens")
        assert _complete(greedy_server, tokens, **body).status_code == 400
    response = _complete(greedy_server, 3000, temperature=0.0, return_token_ids=True)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["token_ids"] == ENGINE_TOKENS
    assert len(greedy_server[1].requests) == 1
    assert len(greedy_server[1].requests[0].prompt_token_ids) == 3000


def test_a_streamed_valid_request_after_a_refusal_streams_its_tokens(greedy_server):
    assert _complete(greedy_server, STANDARD_MAX, temperature=0.0,
                     stream=True).status_code == 400
    response = _complete(greedy_server, 10, temperature=0.0, stream=True,
                         return_token_ids=True)
    assert response.status_code == 200, response.text
    events = [json.loads(line[len("data: "):]) for line in response.text.splitlines()
              if line.startswith("data: {")]
    streamed = [t for event in events for t in event["choices"][0].get("token_ids") or []]
    assert streamed == ENGINE_TOKENS, events
    assert len(greedy_server[1].requests) == 1


def test_host_sampling_admits_top_k_to_the_engine(host_server):
    response = _complete(host_server, 10, top_k=50, return_token_ids=True)
    assert response.status_code == 200, response.text
    assert len(host_server[1].requests) == 1
