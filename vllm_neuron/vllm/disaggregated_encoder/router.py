# SPDX-License-Identifier: Apache-2.0
"""EPD Router orchestration: preprocess once → HRW-route each item → dispatch to VE → drive PD."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.disagg import mm_serde
from vllm_neuron.vllm.disaggregated_encoder.codec import (
    Detokenizer,
    align_placeholders_to_tokens,
    render_request,
)
from vllm_neuron.vllm.disaggregated_encoder.protocol import (
    EncodeResult,
    PreprocessedItem,
    PreprocessedRequest,
    VeReady,
)
from vllm_neuron.vllm.disaggregated_encoder.routing import hrw_pick
from vllm_neuron.utils.vision_utils import get_epd_kwargs


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from transformers import PretrainedConfig
    from vllm.renderers.base import BaseRenderer
    from vllm_neuron.vllm.disaggregated_encoder.routing import PdRegistry, VeRegistry

logger = logging.getLogger(__name__)


async def _sse_data_frames(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Yield each SSE frame payload, buffering across chunks to the \\n\\n boundary
    (httpx does not guarantee one chunk per frame)."""
    buf = ""
    async for chunk in chunks:
        buf += chunk.decode("utf-8")
        while "\n\n" in buf:
            frame, buf = buf.split("\n\n", 1)
            if frame.startswith("data: "):
                yield frame[len("data: ") :]


def error_payload(
    message: str, err_type: str, request_id: str, *, code: int
) -> dict[str, Any]:
    """Build the router's error envelope: a vLLM ErrorInfo (code = HTTP status)
    plus a router-level request_id so clients can correlate the failure."""
    envelope = ErrorResponse(
        error=ErrorInfo(message=message, type=err_type, code=code)
    ).model_dump()
    envelope["request_id"] = request_id
    return envelope


def classify_error(exc: BaseException) -> tuple[int, str, str]:
    """Map a router/upstream exception to (http_code, err_type, message)."""
    if isinstance(exc, httpx.HTTPError):
        return 502, "upstream_error", "upstream request failed"
    if isinstance(exc, (ValidationError, ValueError)):
        return 400, "invalid_request_error", "bad request"
    return 500, "internal_error", "internal error"


def build_pd_request(
    pre: PreprocessedRequest,
    ready: Mapping[str, VeReady],
    sampling: Mapping[str, Any],
    hf_config: "PretrainedConfig",
    *,
    request_id: str | None = None,
    stream: bool = False,
    stream_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Router→PD GenerateRequest body for /inference/v1/generate.

    Args:
        pre: The preprocessed request (token ids + items).
        ready: Map of mm_hash to VeReady (the VE locators to pull from).
        sampling: SamplingParams dict (max_tokens, temperature, ...).
        hf_config: Model config used to slim each item's mm kwargs to the
            LM-relevant subset (see get_epd_kwargs).
        request_id: Optional request_id to propagate through PD.

    Returns:
        A GenerateRequest-shaped dict.

    Notes:
        features.kwargs_data ships only the LM-relevant mm kwargs; encoder-only
        tensors (e.g. pixel_values) are pulled over NIXL by PD.
    """
    mm_hashes: list[str] = []
    placeholders: list[dict[str, int]] = []
    kwargs_encoded: list[str] = []
    for it in pre.items:
        mm_hashes.append(it.mm_hash)
        placeholders.append({"offset": it.offset, "length": it.length})
        item = get_epd_kwargs(hf_config, it.mm_kwargs_item)
        kwargs_encoded.append(mm_serde.encode_mm_kwargs_item(item))

    body: dict[str, Any] = {
        "token_ids": list(pre.prompt_token_ids),
        "features": {
            "mm_hashes": {"image": mm_hashes},
            "mm_placeholders": {"image": placeholders},
            "kwargs_data": {"image": kwargs_encoded},
        },
        "sampling_params": {
            **sampling,
            "extra_args": {
                **(sampling.get("extra_args") or {}),
                "kv_transfer_params": {
                    "ec_locator": {h: ready[h].locator for h in mm_hashes},
                },
            },
        },
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = dict(stream_options or {"include_usage": True})
    if request_id is not None:
        body["request_id"] = request_id
    return body


class Router:
    """Full EPD orchestration: preprocess → route → dispatch → wait → drive PD.

    Args:
        renderer: A vLLM renderer from build_renderer.
        ve_registry: The static set of Vision Encoders.
        pd_registry: The static set of Prefill/Decode servers (round-robin).
        post_encode: Async callable (base_url, [PreprocessedItem], request_id) ->
            dict that POSTs a batched multi-image GenerateRequest to a VE and
            returns the decoded JSON (ec_locator carries one locator per mm_hash).
        post_generate: Async callable (base_url, body) -> dict that POSTs the full
            request body to a PD and returns the decoded GenerateResponse.
        post_generate_stream: Async-iterator callable (base_url, body) yielding raw
            SSE bytes from PD when body["stream"] is True.
    """

    def __init__(
        self,
        renderer: "BaseRenderer",
        ve_registry: "VeRegistry",
        pd_registry: "PdRegistry",
        post_encode: "Callable[[str, Sequence[PreprocessedItem], str], Awaitable[dict[str, Any]]]",
        post_generate: "Callable[[str, Mapping[str, Any]], Awaitable[dict[str, Any]]]",
        post_generate_stream: "Callable[[str, Mapping[str, Any]], AsyncIterator[bytes]]",
    ):
        self._renderer = renderer
        self._ve_registry = ve_registry
        self._pd_registry = pd_registry
        self._post_encode = post_encode
        self._post_generate = post_generate
        self._post_generate_stream = post_generate_stream

    async def encode_request(
        self,
        messages: "Sequence[Mapping[str, Any]]",
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        request_id: str = "",
        debug_prompt_token_ids: "Sequence[int] | None" = None,
    ) -> EncodeResult:
        """Preprocess a request, route each media item to a VE, and await ready acks.

        Args:
            messages: OpenAI chat messages for one conversation.
            chat_template_kwargs: Extra chat-template kwargs.
            request_id: Router-level request id. Propagated to each per-item VE
                call as "{request_id}/enc/{index}" so VE-side logs correlate
                back to the originating Router request.
            debug_prompt_token_ids: Validation-only override. When provided, the
                media items (and their placeholders) are still extracted from
                messages via the renderer -- so encode fires exactly as in
                production -- but the prompt token ids driven to PD are replaced
                with these. Used by logit-validation teacher forcing, where the
                caller supplies a grown [rendered prompt][forced suffix]
                sequence whose prefix matches the rendered prompt.

        Returns:
            An EncodeResult with the preprocessed request and the per-item VE ready acks.
            Text-only requests return an empty ready map.

        Batching: images are HRW-routed to a VE per mm_hash, then all images landing on the same VE are
        sent in ONE /encode POST. The VE returns one EC locator per mm_hash in the
        batch, so a batched response carries them all.
        """
        pre = await render_request(
            self._renderer, messages, chat_template_kwargs=chat_template_kwargs
        )
        if debug_prompt_token_ids is not None:
            # Validation-only teacher-forcing override. Swap the prompt tokens
            # driven to PD, then realign placeholder offsets onto the new stream
            # (the renderer's offsets were computed against its own layout, which
            # can differ -- see align_placeholders_to_tokens).
            pre = align_placeholders_to_tokens(pre, debug_prompt_token_ids)
        if not pre.items:
            return EncodeResult(pre=pre, ready={})

        candidates = self._ve_registry.ids()

        # Dispatch once per distinct mm_hash: duplicate media in one request shares a
        # cache entry, so re-encoding it is wasted work. Group the unique items by
        # their HRW-chosen VE so each VE gets one batched /encode POST.
        unique: dict[str, PreprocessedItem] = {}
        for item in pre.items:
            unique.setdefault(item.mm_hash, item)
        by_ve: dict[str, list[PreprocessedItem]] = {}
        for item in unique.values():
            ve_id = hrw_pick(item.mm_hash, candidates)
            by_ve.setdefault(ve_id, []).append(item)

        async def _encode_batch(
            ve_id: str, items: list[PreprocessedItem]
        ) -> list[VeReady]:
            base_url = self._ve_registry.get(ve_id).base_url
            prefix = f"{request_id}/" if request_id else ""
            resp = await self._post_encode(base_url, items, f"{prefix}enc/{ve_id}")
            ec_locator = (resp.get("kv_transfer_params") or {}).get("ec_locator") or {}
            readies: list[VeReady] = []
            for it in items:
                locator = ec_locator.get(it.mm_hash)
                if not locator:
                    raise httpx.HTTPError(
                        f"VE {ve_id} response for {it.mm_hash} missing EC locator "
                        f"(got {len(ec_locator)} locators for {len(items)} items): {resp!r}"
                    )
                readies.append(VeReady(mm_hash=it.mm_hash, locator=dict(locator)))
            return readies

        batches = await asyncio.gather(
            *(_encode_batch(ve_id, items) for ve_id, items in by_ve.items())
        )
        acks = [ack for batch in batches for ack in batch]
        return EncodeResult(pre=pre, ready={ack.mm_hash: ack for ack in acks})

    async def generate_request(
        self,
        messages: "Sequence[Mapping[str, Any]]",
        sampling: Mapping[str, Any],
        *,
        model: str,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        request_id: str = "",
        debug_prompt_token_ids: "Sequence[int] | None" = None,
    ) -> dict[str, Any]:
        """Full pipeline: encode media → build PD body → post to PD.

        Args:
            messages: OpenAI chat messages for one conversation.
            sampling: SamplingParams dict for PD (max_tokens, temperature, ...).
            model: Model name to stamp on the OpenAI response body.
            chat_template_kwargs: Extra chat-template kwargs.
            request_id: Router-level request id, propagated to VE calls and PD.
            debug_prompt_token_ids: Validation-only override forwarded to
                encode_request -- drive PD with these exact tokens (teacher
                forcing) while still encoding the image from messages.

        Returns:
            An OpenAI ChatCompletion-shaped dict. PD's raw token_ids are
            detokenized on the router using the renderer's tokenizer. Per-token
            logprobs are passed through whenever PD returns them (i.e. the client
            requested them via sampling).
        """
        encode = await self.encode_request(
            messages,
            chat_template_kwargs=chat_template_kwargs,
            request_id=request_id,
            debug_prompt_token_ids=debug_prompt_token_ids,
        )
        body = build_pd_request(
            encode.pre,
            encode.ready,
            sampling,
            request_id=request_id or None,
            hf_config=self._renderer.model_config.hf_config,
        )
        pd_endpoint = self._pd_registry.next()
        logger.debug(
            "EPD Router → PD: request_id=%s pd=%s tokens=%d items=%d",
            request_id or "<none>",
            pd_endpoint.id,
            len(encode.pre.prompt_token_ids),
            len(encode.pre.items),
        )
        pd_response = await self._post_generate(pd_endpoint.base_url, body)
        return self._detokenize_pd_response(
            pd_response,
            prompt_token_count=len(encode.pre.prompt_token_ids),
            request_id=request_id,
            model=model,
            skip_special_tokens=sampling.get("skip_special_tokens", True),
        )

    async def stream_generate_request(
        self,
        messages: "Sequence[Mapping[str, Any]]",
        sampling: Mapping[str, Any],
        model: str,
        *,
        stream_options: Mapping[str, Any] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        request_id: str = "",
    ) -> AsyncIterator[bytes]:
        """Drive one request end-to-end and re-emit PD's tokens-only SSE as
        OpenAI chat.completion.chunk frames"""
        tokenizer = self._renderer.get_tokenizer()
        skip_special_tokens = sampling.get("skip_special_tokens", True)
        spaces_between_special_tokens = sampling.get(
            "spaces_between_special_tokens", True
        )
        detokenizers: dict[int, Detokenizer] = {}
        first_role_sent: set[int] = set()
        completion_id = f"chatcmpl-{request_id or 'epd'}"
        created = int(time.time())

        def _frame(
            choices: list[ChatCompletionResponseStreamChoice],
            *,
            usage: UsageInfo | None = None,
        ) -> bytes:
            frame = ChatCompletionStreamResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=choices,
            )
            # Only set usage on the final frame; leaving it unset keeps it off
            # the per-token frames (exclude_unset), matching upstream.
            if usage is not None:
                frame.usage = usage
            # Match upstream serving_chat: exclude_unset keeps explicit
            # finish_reason=None on per-token frames while dropping fields we
            # never set (logprobs, stop_reason, token_ids, ...).
            data = frame.model_dump_json(exclude_unset=True)
            return f"data: {data}\n\n".encode()

        try:
            encode = await self.encode_request(
                messages,
                chat_template_kwargs=chat_template_kwargs,
                request_id=request_id,
            )
            body = build_pd_request(
                encode.pre,
                encode.ready,
                sampling,
                request_id=request_id or None,
                stream=True,
                stream_options=stream_options,
                hf_config=self._renderer.model_config.hf_config,
            )
            pd_endpoint = self._pd_registry.next()
            logger.debug(
                "EPD Router → PD (stream): request_id=%s pd=%s tokens=%d items=%d",
                request_id or "<none>",
                pd_endpoint.id,
                len(encode.pre.prompt_token_ids),
                len(encode.pre.items),
            )
            async for payload in _sse_data_frames(
                self._post_generate_stream(pd_endpoint.base_url, body)
            ):
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("PD stream: unparsable payload %r", payload)
                    continue
                # PD error frames come as `{"error": ..., ...}` without `choices`.
                if data.get("error") is not None and "choices" not in data:
                    yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()
                    continue

                out_choices: list[ChatCompletionResponseStreamChoice] = []
                for c in data.get("choices") or []:
                    idx = c.get("index", 0)
                    delta_ids = list(c.get("token_ids") or [])
                    finish_reason = c.get("finish_reason")
                    det = detokenizers.get(idx)
                    if det is None:
                        det = detokenizers[idx] = Detokenizer(
                            tokenizer,
                            skip_special_tokens=skip_special_tokens,
                            spaces_between_special_tokens=spaces_between_special_tokens,
                        )
                    delta_text = det.decode_delta(delta_ids)
                    delta_kwargs: dict[str, Any] = {}
                    if idx not in first_role_sent:
                        delta_kwargs["role"] = "assistant"
                        first_role_sent.add(idx)
                    if delta_text:
                        delta_kwargs["content"] = delta_text
                    delta = DeltaMessage(**delta_kwargs)
                    out_choices.append(
                        ChatCompletionResponseStreamChoice(
                            index=idx, delta=delta, finish_reason=finish_reason
                        )
                    )

                if out_choices:
                    yield _frame(out_choices)
                elif data.get("usage"):
                    yield _frame([], usage=UsageInfo(**data["usage"]))
        except Exception as exc:
            logger.exception("streaming router failed for %s", request_id)
            code, err_type, message = classify_error(exc)
            payload = error_payload(message, err_type, request_id, code=code)
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()

        yield b"data: [DONE]\n\n"

    def _detokenize_pd_response(
        self,
        pd_response: dict,
        *,
        prompt_token_count: int,
        request_id: str,
        model: str,
        skip_special_tokens: bool,
    ) -> dict:
        """Detokenize PD's tokens-only GenerateResponse into OpenAI ChatCompletion shape.

        PD's per-token logprobs are passed through whenever present -- PD only
        returns them when the client requested logprobs via sampling -- so no
        separate opt-in flag is needed.
        """
        tokenizer = self._renderer.get_tokenizer()
        choices: list[ChatCompletionResponseChoice] = []
        completion_tokens = 0
        for c in pd_response.get("choices") or []:
            token_ids = list(c.get("token_ids") or [])
            completion_tokens += len(token_ids)
            content = Detokenizer(
                tokenizer, skip_special_tokens=skip_special_tokens
            ).decode_delta(token_ids)
            choices.append(
                ChatCompletionResponseChoice(
                    index=c.get("index", 0),
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=c.get("finish_reason", "stop"),
                    token_ids=token_ids,
                )
            )
        response = ChatCompletionResponse(
            id=pd_response.get("request_id") or request_id or "epd-router",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_tokens,
                total_tokens=prompt_token_count + completion_tokens,
            ),
        )
        body = response.model_dump(exclude_unset=True, exclude_none=True)
        # Pass through PD's per-token logprobs when present (grafted after the
        # pydantic dump so the response shape is untouched when PD omits them).
        for out_choice, pd_choice in zip(
            body["choices"], pd_response.get("choices") or []
        ):
            logprobs = pd_choice.get("logprobs")
            if logprobs is not None:
                out_choice["logprobs"] = logprobs
        return body


async def post_encode(
    client: "httpx.AsyncClient",
    base_url: str,
    items: "Sequence[PreprocessedItem]",
    request_id: str,
) -> dict[str, Any]:
    """POST a batched multi-image GenerateRequest to {base_url}/inference/v1/generate.

    Args:
        client: An httpx.AsyncClient.
        base_url: The VE's http://host:port base URL.
        items: The preprocessed items to encode together on this VE.
        request_id: Router-side id to embed in the VE request for log
            correlation.

    Returns:
        The decoded JSON GenerateResponse from the VE (its kv_transfer_params.
        ec_locator carries one locator per item mm_hash).
    """
    mm_hashes: list[str] = []
    placeholders: list[dict[str, int]] = []
    kwargs_encoded: list[str] = []
    offset = 0
    for it in items:
        mm_hashes.append(it.mm_hash)
        placeholders.append({"offset": offset, "length": it.length})
        kwargs_encoded.append(mm_serde.encode_mm_kwargs_item(it.mm_kwargs_item))
        offset += it.length
    body: dict[str, Any] = {
        "request_id": request_id,
        "token_ids": [0] * offset,
        "features": {
            "mm_hashes": {"image": mm_hashes},
            "mm_placeholders": {"image": placeholders},
            "kwargs_data": {"image": kwargs_encoded},
        },
        "sampling_params": {"max_tokens": 1},
    }
    resp = await client.post(f"{base_url}/inference/v1/generate", json=body)
    resp.raise_for_status()
    return resp.json()


async def post_generate(
    client: "httpx.AsyncClient",
    base_url: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """POST a full GenerateRequest body to a PD and return the parsed JSON."""
    resp = await client.post(f"{base_url}/inference/v1/generate", json=dict(body))
    resp.raise_for_status()
    return resp.json()


async def post_generate_stream(
    client: "httpx.AsyncClient",
    base_url: str,
    body: Mapping[str, Any],
) -> AsyncIterator[bytes]:
    """POST a streaming GenerateRequest and yield raw SSE bytes.

    The httpx.stream context is held open for the generator's lifetime; the
    caller must fully iterate (or aclose) to release the pooled connection.
    """
    async with client.stream(
        "POST", f"{base_url}/inference/v1/generate", json=dict(body)
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk
