# SPDX-License-Identifier: Apache-2.0
"""FastAPI entrypoint for the EPD Router.

Thin process/HTTP wiring around Router. Builds the renderer, an httpx client for
the VE pool (running vLLM with --mm-encoder-only + ec_producer role) and a
second httpx client for the PD pool (additional_config
{"mm_language_model_only": true} + ec_consumer role) at startup, and exposes:

* POST /v1/chat/completions -- preprocess once, HRW-route each media item to a
  VE, wait for ready acks (which carry the per-mm_hash EC locator), then drive
  a chosen PD's /inference/v1/generate with token_ids + placeholders +
  kv_transfer_params.ec_locator, and return an OpenAI ChatCompletion-shaped
  JSON body. When stream=True, PD's tokens-only SSE stream is
  incrementally detokenized on the router and re-emitted as OpenAI
  chat.completion.chunk frames.
* GET /healthcheck.

Example (co-located 1VE + 1PD on one host)::

    python examples/vllm_neuron/vllm/disaggregated_encoder/server.py \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --ve-endpoint 127.0.0.1:8300 \\
        --pd-endpoint 127.0.0.1:8100 \\
        --port 8000

Serving a quantized checkpoint additionally needs --quantization, even though
the router loads no weights -- see the --quantization help text in parse_args::

    python examples/vllm_neuron/vllm/disaggregated_encoder/server.py \\
        --model /path/to/qwen3-vl-mxfp8 \\
        --quantization mxfp8 \\
        --ve-endpoint 127.0.0.1:8300 \\
        --pd-endpoint 127.0.0.1:8100 \\
        --port 8000
"""

from __future__ import annotations

import argparse
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import msgspec
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_neuron import envs
from vllm_neuron.vllm.disaggregated_encoder.codec import build_renderer
from vllm_neuron.vllm.disaggregated_encoder.router import (
    Router,
    classify_error,
    error_payload,
    post_encode,
    post_generate,
    post_generate_stream,
)
from vllm_neuron.vllm.disaggregated_encoder.routing import (
    Endpoint,
    PdRegistry,
    VeRegistry,
)

logger = logging.getLogger(__name__)


def _map_router_error(exc: BaseException, request_id: str) -> JSONResponse:
    """Router / upstream exception → OpenAI-shaped JSONResponse."""
    status, err_type, message = classify_error(exc)
    logger.warning(
        "router error %s (%s): %s",
        request_id,
        status,
        exc,
        exc_info=(status == 500),
    )
    return JSONResponse(
        status_code=status,
        content=error_payload(message, err_type, request_id, code=status),
    )


@dataclass
class RouterConfig:
    """Static configuration for the Router process.

    Args:
        model: HF model id / path (must match VE + PD).
        ve_instances: (host, port) pairs for the Vision Encoders.
        pd_instances: (host, port) pairs for Prefill/Decode servers.
        host: Bind host for the Router's own HTTP server.
        port: Bind port for the Router's own HTTP server.
        ve_timeout_s: Per-VE /inference/v1/generate request timeout (seconds).
        pd_timeout_s: Per-PD /inference/v1/generate request timeout (seconds).
        quantization: Neuron quantization of the checkpoint (e.g. "mxfp8"), or
            None for an unquantized one. Declared only so the renderer's
            VllmConfig passes the Neuron platform's quantization validator; the
            router loads no weights.
    """

    model: str
    ve_instances: list[tuple[str, int]]
    pd_instances: list[tuple[str, int]]
    host: str = "127.0.0.1"
    port: int = 8000
    ve_timeout_s: float = 120.0
    pd_timeout_s: float = 600.0
    quantization: str | None = None

    def ve_endpoints(self) -> list[Endpoint]:
        """VE endpoints with stable ids ve-0, ve-1, ... (HRW candidates)."""
        return [Endpoint(f"ve-{i}", h, p) for i, (h, p) in enumerate(self.ve_instances)]

    def pd_endpoints(self) -> list[Endpoint]:
        """PD endpoints with stable ids pd-0, pd-1, ... (round-robin pool)."""
        return [Endpoint(f"pd-{i}", h, p) for i, (h, p) in enumerate(self.pd_instances)]


def build_app(config: RouterConfig) -> FastAPI:
    """Construct the FastAPI app for config.

    Args:
        config: The Router configuration.

    Returns:
        A FastAPI app whose lifespan builds the renderer, the VE + PD client
        pools, and the Router; and which serves /v1/chat/completions and
        /healthcheck.
    """
    ve_registry = VeRegistry(config.ve_endpoints())
    pd_registry = PdRegistry(config.pd_endpoints())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ve_client = httpx.AsyncClient(
            timeout=config.ve_timeout_s,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        pd_client = httpx.AsyncClient(
            timeout=config.pd_timeout_s,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        renderer = build_renderer(config.model, quantization=config.quantization)

        async def _post_encode(base_url, items, request_id):
            return await post_encode(ve_client, base_url, items, request_id)

        async def _post_generate(base_url, body):
            return await post_generate(pd_client, base_url, body)

        def _post_generate_stream(base_url, body):
            return post_generate_stream(pd_client, base_url, body)

        app.state.ve_client = ve_client
        app.state.pd_client = pd_client
        app.state.router = Router(
            renderer=renderer,
            ve_registry=ve_registry,
            pd_registry=pd_registry,
            post_encode=_post_encode,
            post_generate=_post_generate,
            post_generate_stream=_post_generate_stream,
        )
        logger.info(
            "EPD Router ready: model=%s ves=%s pds=%s",
            config.model,
            ve_registry.ids(),
            pd_registry.ids(),
        )
        try:
            yield
        finally:
            await ve_client.aclose()
            await pd_client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            body = await request.json()
            if not isinstance(body, dict) or not body.get("messages"):
                raise ValueError("body must be an object with non-empty 'messages'")
            req = ChatCompletionRequest.model_validate(body)
            max_tokens = req.max_completion_tokens or req.max_tokens
            sp = req.to_sampling_params(
                max_tokens=max_tokens, default_sampling_params={}
            )
            sampling = msgspec.to_builtins(sp)
        except (ValueError, ValidationError) as exc:
            return _map_router_error(exc, request_id)
        messages = body["messages"]
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options")
        # Validation-only teacher-forcing override: drive PD with these exact
        # prompt tokens (image still encoded from messages). Gated behind
        # VLLM_NEURON_DEBUG_MODE so a client can't bypass tokenization on a
        # deployed instance; ignored (None) in production.
        debug_prompt_token_ids = (
            body.get("prompt_token_ids") if envs.VLLM_NEURON_DEBUG_MODE else None
        )
        chat_template_kwargs = {
            "add_generation_prompt": body.get("add_generation_prompt", True)
        }
        model_name = body.get("model") or config.model

        router: Router = request.app.state.router
        if stream:
            return StreamingResponse(
                router.stream_generate_request(
                    messages,
                    sampling,
                    model=model_name,
                    stream_options=stream_options,
                    chat_template_kwargs=chat_template_kwargs,
                    request_id=request_id,
                ),
                media_type="text/event-stream",
            )

        try:
            result = await router.generate_request(
                messages,
                sampling,
                model=model_name,
                chat_template_kwargs=chat_template_kwargs,
                request_id=request_id,
                debug_prompt_token_ids=debug_prompt_token_ids,
            )
        except Exception as exc:
            return _map_router_error(exc, request_id)
        return JSONResponse(content=result)

    @app.get("/healthcheck")
    async def healthcheck():
        return {
            "status": "ok",
            "ve_instances": len(ve_registry.ids()),
            "pd_instances": len(pd_registry.ids()),
        }

    return app


def _parse_endpoint(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"expected host:port, got {spec!r}")
    host, _, port = spec.rpartition(":")
    return host, int(port)


def parse_args(argv: list[str] | None = None) -> RouterConfig:
    """Parse CLI args into a RouterConfig."""
    parser = argparse.ArgumentParser(description="EPD Router")
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local path. Must be the same checkpoint the VE "
        "and PD engines were launched with -- the router tokenizes and "
        "detokenizes with it.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the router's own HTTP server (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for the router's own HTTP server (default: %(default)s).",
    )
    parser.add_argument(
        "--ve-endpoint",
        dest="ve_endpoints",
        type=_parse_endpoint,
        action="append",
        required=True,
        help="host:port of a Vision Encoder engine. Repeat once per VE; media "
        "items are spread over the pool by rendezvous hashing.",
    )
    parser.add_argument(
        "--ve-timeout-s",
        type=float,
        default=120.0,
        help="Per-VE encode request timeout in seconds (default: %(default)s). "
        "Raise it if a cold VE is still compiling vision buckets.",
    )
    parser.add_argument(
        "--pd-endpoint",
        dest="pd_endpoints",
        type=_parse_endpoint,
        action="append",
        required=True,
        help="host:port of a Prefill+Decode engine. Repeat once per PD; "
        "requests are assigned round-robin.",
    )
    parser.add_argument(
        "--pd-timeout-s",
        type=float,
        default=600.0,
        help="Per-PD generate request timeout in seconds (default: %(default)s). "
        "Must cover the full generation, not just time-to-first-token.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Neuron quantization of the checkpoint, e.g. mxfp8. Required for a "
        "quantized checkpoint even though the router loads no weights: the "
        "renderer builds a VllmConfig, and the Neuron platform rejects the "
        "checkpoint's quantization_config unless the CPU-dequant "
        "quantization is declared here.",
    )

    args = parser.parse_args(argv)
    return RouterConfig(
        model=args.model,
        ve_instances=list(args.ve_endpoints),
        pd_instances=list(args.pd_endpoints),
        host=args.host,
        port=args.port,
        ve_timeout_s=args.ve_timeout_s,
        pd_timeout_s=args.pd_timeout_s,
        quantization=args.quantization,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: parse args, build the app, and serve with uvicorn."""
    import uvicorn

    config = parse_args(argv)
    uvicorn.run(build_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
