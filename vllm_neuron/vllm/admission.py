# SPDX-License-Identifier: Apache-2.0
"""Request admission: refuse, on the request path, what this server cannot serve.

``NeuronPlatform.check_and_update_config`` resolves an :class:`AdmissionPolicy` from the
served config, in the API server process. ``NeuronPlatform.validate_request`` calls
:meth:`AdmissionPolicy.check` for every request, from vLLM's
``InputProcessor.process_inputs``, before the request becomes an ``EngineCoreRequest``.
A refusal is a ``ValueError``, which vLLM's OpenAI server returns as HTTP 400; the engine
never sees the request.

Five request classes are refused:

* **A prompt that leaves no room for a generated token** (``prompt + 1 > max_model_len``).
  Every sampling request generates at least one token (``max_tokens >= 1``), and the KV
  block table, the DSA side caches and the decode graphs are sized for ``max_model_len``
  positions, so the longest prompt is ``max_model_len - 1`` tokens. vLLM's own check
  (``prompt > max_model_len``) admits a prompt of exactly ``max_model_len``, whose first
  decode step would write past the request's pages. Nothing shorter is enforced: the
  GLM-5.3-Flash runner reads a prefill chunk's KV through a window of
  ``kv_segment_size + query bucket`` tokens, picks the KV segment per request from
  ``kv_segment_size_buckets`` and refuses at startup a list whose largest segment does
  not cover ``max_model_len`` (``bucket_utils.validate_kv_segment_size_buckets``), so
  every server that starts prefills every prompt vLLM admits. A pooling request
  generates nothing and may fill ``max_model_len``.
* **A sampling knob the ``all_greedy`` on-device sampler cannot apply.** That sampler is
  ``torch.argmax`` for every row (``functional/full_vocab_sampling.py``), so
  ``temperature > 0``, ``top_k``, ``top_p``, ``min_p``, ``seed`` and ``n > 1`` would be
  ignored in silence. A request whose knobs all equal the server defaults is admitted
  and served greedy: vLLM fills unset knobs from the model's generation config before
  this check runs, so "left unset" and "set to the default" look the same here.
* **A sampling control the on-device sampler does not apply**, set to a value other than
  its default: ``presence_penalty``, ``frequency_penalty``, ``repetition_penalty``,
  ``logit_bias``, ``bad_words`` and ``allowed_token_ids``. The device sampler reads one
  ``[top_k, top_p, temperature]`` row per request and the structured-output mask
  (``functional/full_vocab_sampling.py``); with it, the runner returns the device tokens
  and calls vLLM's ``Sampler`` only for logprobs, and its input batch has no logits
  processors (vLLM applies ``logit_bias`` and ``min_tokens`` in its built-in ones). So
  none of these reaches the tokens, greedy or not.
* **``min_tokens`` while a stop token id applies** (``ignore_eos`` false, so EOS is a
  stop, or ``stop_token_ids`` set). vLLM's scheduler does not stop a request before
  ``min_tokens`` and its detokenizer holds stop strings until then, but it is vLLM's
  min-tokens logits processor that keeps the EOS and stop ids out of those first tokens,
  and the device sampler has none. So the output can hold a stop token with more text
  after it. With ``ignore_eos`` and no ``stop_token_ids`` the request asks for text past
  EOS, and ``min_tokens`` is admitted. The gap left: with ``ignore_eos`` an EOS id may
  appear inside the first ``min_tokens`` tokens where a masking sampler would have
  excluded it.
* **logprobs / prompt_logprobs when the device sampler returns token ids only.** The
  runner then has no logits on the host; vLLM's completions endpoint indexes an empty
  logprobs list and answers HTTP 500.

Beyond the token and sampling rules above, one more request class is refused: an image
with more embeddings than the compiled vision encoder holds, that is, more than the
largest ``num_vision_tokens_buckets`` entry divided by the model's merge factor
(``vision_utils.get_vision_embed_limit``, which also caps the scheduler's encoder
budget). Such an image fits no compiled bucket, and the scheduler can never schedule it.
The refusal is a ``VLLMValidationError`` naming the content part, so the 400 carries it
as ``param``.

A media URL the client got wrong is answered with HTTP 400 too, not 500
(:func:`install_client_media_errors`): a ``file://`` URL on a server that loads no local
files, and an ``http(s)`` URL that cannot be fetched.

``validate_request`` alone answers a refused *streamed* request with HTTP 200 and an
error event: vLLM's ``AsyncLLM.generate`` is an async generator, so its body (where
``process_inputs`` runs) starts only once the response stream is iterated. vLLM's own
``SamplingParams.verify`` checks (for example ``logit_bias`` under speculative decoding)
run there too, before ``validate_request``. :func:`install_eager_validation` runs both,
in that order, when ``generate`` is called, before the stream begins, so streamed and
plain requests get the same 400.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_neuron.utils.vision_utils import VisionEmbedLimit

logger = logging.getLogger(__name__)

#: Architectures whose on-device sampler returns token ids and no logits, under
#: synchronous scheduling too (the GLM root hands its logits to ``sample_full_vocab``
#: and returns only its int32 tokens).
TOKEN_ONLY_SAMPLER_ARCHS = ("Glm5NextForConditionalGeneration",)

#: vLLM's own greedy threshold (``vllm.sampling_params._SAMPLING_EPS``).
_GREEDY_TEMPERATURE = 1e-5

#: The value each sampling knob takes when neither the client nor the model's
#: generation config sets it (``CompletionRequest._DEFAULT_SAMPLING_PARAMS``, the same
#: in the chat protocol and in ``SamplingParams``).
_PROTOCOL_DEFAULTS: dict[str, float] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}

#: Warnings already logged in this process, so each is logged once.
_WARNED: set[str] = set()

#: Marks the wrapped ``AsyncLLM.generate`` so a second install is a no-op.
_EAGER_MARK = "_vllm_neuron_eager_validation"

#: Marks the wrapped ``MediaConnector`` loaders so a second install is a no-op.
_MEDIA_MARK = "_vllm_neuron_client_media_errors"

#: The content part each of vLLM's media readers loads, for a refusal's ``param``.
_MEDIA_PARAMETERS = {
    "ImageMediaIO": "image_url",
    "VideoMediaIO": "video_url",
    "AudioMediaIO": "audio_url",
}


# ── prompt length ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromptLength:
    """The longest prompt this server generates from: ``max_model_len - 1`` tokens."""

    max_model_len: int

    @property
    def tokens(self) -> int:
        return int(self.max_model_len) - 1

    def refusal(self, prompt_len: int) -> str:
        return (
            f"The prompt has {prompt_len} tokens and this server's max_model_len is "
            f"{self.max_model_len}: the prompt and at least one generated token must fit "
            f"in max_model_len, so a prompt can have at most {self.tokens} tokens. "
            f"Shorten the prompt, or restart the server with a larger --max-model-len."
        )


def prompt_length(processed_inputs: Any) -> int | None:
    """Token count of a processed prompt (``EngineInput``), or None if it carries none."""
    if not isinstance(processed_inputs, Mapping):
        return None
    if processed_inputs.get("type") == "enc_dec":
        processed_inputs = processed_inputs.get("decoder_prompt") or {}
    token_ids = processed_inputs.get("prompt_token_ids")
    if token_ids is not None:
        return len(token_ids)
    embeds = processed_inputs.get("prompt_embeds")
    if embeds is not None:
        return len(embeds)
    return None


# ── sampling ─────────────────────────────────────────────────────────────────────


def _same(value: float, default: float) -> bool:
    return abs(float(value) - float(default)) <= 1e-6


@dataclass(frozen=True)
class SamplingAdmission:
    """What the served sampler can honour.

    ``on_device``: the tokens come from the on-device sampler. ``all_greedy``: every
    token is the argmax. ``logprobs``: the host receives logits to compute logprobs from.
    ``prompt_logprobs``: likewise for the prompt. ``defaults``: the knob values a request
    carries when the client sets none of them.
    """

    on_device: bool = False
    all_greedy: bool = False
    logprobs: bool = True
    prompt_logprobs: bool = True
    defaults: Mapping[str, float] = field(default_factory=lambda: dict(_PROTOCOL_DEFAULTS))

    def refusals(self, params) -> list[str]:
        refusals = []
        missing = []
        if not self.logprobs and params.logprobs is not None:
            missing.append(f"logprobs={params.logprobs}")
        if not self.logprobs and getattr(params, "logprob_token_ids", None) is not None:
            missing.append("logprob_token_ids")
        if not self.prompt_logprobs and params.prompt_logprobs is not None:
            missing.append(f"prompt_logprobs={params.prompt_logprobs}")
        if missing:
            refusals.append(
                f"This server samples on device (on_device_sampling_config is set) and "
                f"its sampler returns token ids only, so it cannot return "
                f"{', '.join(missing)}. Leave logprobs and prompt_logprobs unset, or "
                f"restart the server with on_device_sampling_config: null to sample on "
                f"the host."
            )
        knobs = self._knobs_argmax_ignores(params)
        if knobs:
            refusals.append(
                f"This server samples on device with on_device_sampling_config."
                f"all_greedy=true, so it returns only the greedy (argmax) token and "
                f"cannot apply {', '.join(knobs)}. Send temperature=0, or leave the "
                f"sampling parameters at the server defaults, or restart the server "
                f"without all_greedy to sample."
            )
        controls = self._controls_the_device_ignores(params)
        if controls:
            refusals.append(
                f"This server samples on device (on_device_sampling_config is set), and "
                f"the device sampler does not apply {', '.join(controls)}. Leave these "
                f"parameters unset."
            )
        if self._min_tokens_a_stop_token_can_break(params):
            refusals.append(
                f"This server samples on device (on_device_sampling_config is set), and "
                f"the device sampler does not keep the EOS and stop token ids out of the "
                f"first min_tokens={params.min_tokens} tokens, so the output can hold a "
                f"stop token with more text after it. Leave min_tokens unset; it is "
                f"admitted with ignore_eos=true and no stop_token_ids."
            )
        return refusals

    def _controls_the_device_ignores(self, params) -> list[str]:
        if not self.on_device:
            return []
        controls = []
        if not _same(params.presence_penalty, 0.0):
            controls.append(f"presence_penalty={params.presence_penalty:g}")
        if not _same(params.frequency_penalty, 0.0):
            controls.append(f"frequency_penalty={params.frequency_penalty:g}")
        if not _same(params.repetition_penalty, self.defaults["repetition_penalty"]):
            controls.append(f"repetition_penalty={params.repetition_penalty:g}")
        if params.logit_bias:
            controls.append("logit_bias")
        if params.bad_words:
            controls.append("bad_words")
        if params.allowed_token_ids is not None:
            controls.append("allowed_token_ids")
        return controls

    def _min_tokens_a_stop_token_can_break(self, params) -> bool:
        return (
            self.on_device
            and params.min_tokens > 0
            and (not params.ignore_eos or bool(params.stop_token_ids))
        )

    def _knobs_argmax_ignores(self, params) -> list[str]:
        if not self.all_greedy:
            return []
        # Argmax is what was asked for, whatever else is set.
        if params.temperature < _GREEDY_TEMPERATURE or params.top_k == 1:
            return []
        defaults = self.defaults
        knobs = []
        if not _same(params.temperature, defaults["temperature"]):
            knobs.append(f"temperature={params.temperature:g}")
        if params.top_k not in (0, -1) and params.top_k != defaults["top_k"]:
            knobs.append(f"top_k={params.top_k}")
        if params.top_p < 1.0 and not _same(params.top_p, defaults["top_p"]):
            knobs.append(f"top_p={params.top_p:g}")
        if params.min_p > 0.0 and not _same(params.min_p, defaults["min_p"]):
            knobs.append(f"min_p={params.min_p:g}")
        if params.seed is not None:
            knobs.append(f"seed={params.seed}")
        if params.n > 1:
            knobs.append(f"n={params.n}")
        if not knobs and "default-temperature" not in _WARNED:
            _WARNED.add("default-temperature")
            logger.warning(
                "This server samples on device with all_greedy=true. A request whose "
                "sampling parameters are all at the server defaults (temperature=%g, "
                "top_p=%g, top_k=%d, min_p=%g) is served greedy (argmax), not sampled; "
                "send temperature=0 to ask for greedy explicitly. Logged once.",
                defaults["temperature"], defaults["top_p"], defaults["top_k"],
                defaults["min_p"],
            )
        return knobs


# ── image size ───────────────────────────────────────────────────────────────────


#: The ``mm_processor_kwargs`` key that caps an image's placeholder tokens, for each
#: architecture whose image processor reads one. transformers' GLM-5.3-Flash image
#: processor shrinks an image to at most ``max_image_tokens`` placeholder tokens, the unit
#: of the per-image limit, and does not read ``max_pixels``.
IMAGE_TOKEN_CAP_KWARGS = {"Glm5NextForConditionalGeneration": "max_image_tokens"}


def image_resize_kwarg(hf_config) -> str | None:
    """The ``mm_processor_kwargs`` key a client can send to shrink an image on this
    model, or None when the model reads none."""
    architectures = tuple(getattr(hf_config, "architectures", None) or ())
    for arch in architectures:
        if arch in IMAGE_TOKEN_CAP_KWARGS:
            return IMAGE_TOKEN_CAP_KWARGS[arch]
    from vllm_neuron.model.interfaces import SupportsMaxPixels
    from vllm_neuron.model.registry import get_models

    models = dict(get_models())
    for arch in architectures:
        model_cls = models.get(arch)
        if model_cls is not None and issubclass(model_cls, SupportsMaxPixels):
            return "max_pixels"
    return None


@dataclass(frozen=True)
class ImageEmbeds:
    """The image rule: the most embeddings one multimodal item may produce, and the key
    a client can send to shrink an image."""

    #: The per-image limit; the scheduler's encoder budget reads the same value.
    embed_limit: VisionEmbedLimit
    #: The ``mm_processor_kwargs`` key that shrinks an image on this model, if any.
    resize_kwarg: str | None = None

    @property
    def limit(self) -> int:
        return self.embed_limit.embeds

    def refusal(self, modality: str, num_embeds: int) -> str:
        fixes = [f"Send a smaller {modality}"]
        if self.resize_kwarg == "max_pixels":
            fixes.append("send a smaller max_pixels in mm_processor_kwargs")
        elif self.resize_kwarg is not None:
            fixes.append(
                f'send mm_processor_kwargs {{"{self.resize_kwarg}": {self.limit}}} to have '
                f"the server shrink it"
            )
        return (
            f"The {modality} in this request produces {num_embeds} embedding tokens, and "
            f"this server admits at most {self.limit} per {modality}: the largest "
            f"num_vision_tokens_buckets entry in vision_neuron_config "
            f"({self.embed_limit.max_bucket} vision tokens) divided by the merge factor "
            f"{self.embed_limit.merge_factor}. "
            f"{', or '.join(fixes)}."
        )

    def first_refusal(self, processed_inputs: Any):
        """``(message, parameter, num_embeds)`` for the first item above the limit, or
        None. ``parameter`` is the content part that carried the item (``image_url``)."""
        if not isinstance(processed_inputs, Mapping):
            return None
        placeholders = processed_inputs.get("mm_placeholders") or {}
        for modality, positions in placeholders.items():
            for position in positions:
                # get_num_embeds, not length: a video placeholder interleaves timestamp
                # text tokens with the embeddings (is_embed), so length overcounts what
                # the vision encoder produces. For an image the two are equal.
                num_embeds = position.get_num_embeds()
                if num_embeds > self.limit:
                    return self.refusal(modality, num_embeds), f"{modality}_url", num_embeds
        return None


def image_embeds_from_config(vllm_config) -> ImageEmbeds | None:
    """The image rule for this served config, or None when it has no vision buckets."""
    from vllm_neuron.utils.vision_utils import get_vision_embed_limit

    limit = get_vision_embed_limit(vllm_config)
    if limit is None:
        return None
    return ImageEmbeds(
        embed_limit=limit,
        resize_kwarg=image_resize_kwarg(vllm_config.model_config.hf_config),
    )


# ── the policy ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdmissionPolicy:
    prompt: PromptLength | None = None
    sampling: SamplingAdmission = field(default_factory=SamplingAdmission)
    image: ImageEmbeds | None = None

    def check(self, processed_inputs: Any, params: Any) -> None:
        """Raise ``ValueError`` naming every reason this request cannot be served.

        A request refused for one oversized image alone gets a ``VLLMValidationError``
        that names the content part, so vLLM's 400 carries it as ``param``.
        """
        refusals = []
        # PoolingParams carry no sampling knobs and generate no token, so neither the
        # length rule nor the sampling rules apply to them.
        if hasattr(params, "temperature"):
            if self.prompt is not None:
                tokens = prompt_length(processed_inputs)
                if tokens is not None and tokens > self.prompt.tokens:
                    refusals.append(self.prompt.refusal(tokens))
            refusals.extend(self.sampling.refusals(params))
        image = None if self.image is None else self.image.first_refusal(processed_inputs)
        if image is not None and not refusals:
            from vllm.exceptions import VLLMValidationError

            message, parameter, num_embeds = image
            raise VLLMValidationError(message, parameter=parameter, value=num_embeds)
        if image is not None:
            refusals.append(image[0])
        if refusals:
            raise ValueError(" ".join(refusals))


def _server_defaults(model_config) -> dict[str, float]:
    """The knob values vLLM fills in for a client that sets none (the model's generation
    config over the protocol defaults), as the serving layer computes them."""
    try:
        diff = model_config.get_diff_sampling_param()
    except Exception as exc:  # a broken generation config must not stop the server
        logger.warning(
            "Could not read the model's default sampling parameters (%s); admission "
            "compares requests against vLLM's protocol defaults %s",
            exc, _PROTOCOL_DEFAULTS,
        )
        diff = {}
    return {name: diff.get(name, value) for name, value in _PROTOCOL_DEFAULTS.items()}


def policy_from_config(vllm_config) -> AdmissionPolicy:
    """The policy for this served config. Called after the platform has settled
    ``on_device_sampling_config``."""
    from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig

    model_config = vllm_config.model_config
    architectures = tuple(getattr(model_config.hf_config, "architectures", None) or ())
    neuron_config = vllm_config.additional_config.get("neuron_config", {})

    prompt = PromptLength(int(model_config.max_model_len))

    # The runner samples on device exactly when NeuronConfig.from_dict yields a sampler
    # config: absent means the default config, an explicit null means none.
    if "on_device_sampling_config" in neuron_config:
        raw = neuron_config["on_device_sampling_config"]
        sampler = None if raw is None else OnDeviceSamplingConfig(**dict(raw))
    else:
        sampler = OnDeviceSamplingConfig()
    if sampler is None:
        sampling = SamplingAdmission()
    else:
        # Under async scheduling the runner returns no logprobs for any model; a
        # token-only sampler returns none under synchronous scheduling either.
        logprobs = not (
            vllm_config.scheduler_config.async_scheduling
            or any(arch in TOKEN_ONLY_SAMPLER_ARCHS for arch in architectures)
        )
        sampling = SamplingAdmission(
            on_device=True,
            all_greedy=sampler.all_greedy,
            logprobs=logprobs,
            # The runner never fills prompt logprobs, and with the sampler on the
            # device there are no prompt logits on the host to fill them from.
            prompt_logprobs=False,
            defaults=_server_defaults(model_config),
        )
    image = image_embeds_from_config(vllm_config)
    policy = AdmissionPolicy(prompt=prompt, sampling=sampling, image=image)
    logger.info(
        "Request admission: prompts up to %d tokens (max_model_len %d less one generated "
        "token); on-device sampling %s%s",
        prompt.tokens,
        prompt.max_model_len,
        "off" if sampler is None
        else f"on (all_greedy={sampler.all_greedy}, logprobs={sampling.logprobs})",
        # A model class without multimodal input gets no image, so its image rule is idle.
        ""
        if image is None or model_config.multimodal_config is None
        else f"; images up to {image.limit} embeddings",
    )
    return policy


# ── eager validation for streamed requests ───────────────────────────────────────


def _verify_sampling_params(engine, params) -> None:
    """vLLM's ``SamplingParams.verify``, with the configs ``process_inputs`` passes it.

    The structured-outputs config is passed as None, so the grammar check (which also
    records the backend on the request) stays in ``process_inputs`` and runs once; every
    other check in ``verify`` only reads the request."""
    from vllm.sampling_params import SamplingParams

    if not isinstance(params, SamplingParams):
        return
    processor = engine.input_processor
    params.verify(
        processor.model_config, processor.speculative_config, None, processor.tokenizer
    )


def install_eager_validation() -> None:
    """Run the request checks of ``process_inputs`` when ``AsyncLLM.generate`` is called.

    ``generate`` is an async generator: ``process_inputs`` (and so vLLM's
    ``SamplingParams.verify`` and the platform's ``validate_request``) runs on the first
    iteration, which for a streamed response is after vLLM has sent HTTP 200. Validating
    at the call instead makes the ``ValueError`` propagate out of the endpoint handler,
    where vLLM's exception handler returns 400, for streamed and plain requests alike.
    The two checks run in the order ``process_inputs`` runs them, so the first refusal is
    the same either way. ``process_inputs`` validates again as before. Only a processed
    prompt (``EngineInput``, what every OpenAI endpoint passes) is checked here; anything
    else is left to ``process_inputs``. Idempotent.
    """
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except ImportError:  # an engine without the v1 async client has nothing to wrap
        return
    if getattr(AsyncLLM.generate, _EAGER_MARK, False):
        return
    lazy_generate = AsyncLLM.generate

    @functools.wraps(lazy_generate)
    def generate(self, *args, **kwargs):
        prompt = args[0] if args else kwargs.get("prompt")
        params = args[1] if len(args) > 1 else kwargs.get("sampling_params")
        if isinstance(prompt, Mapping) and "type" in prompt and params is not None:
            from vllm.platforms import current_platform

            _verify_sampling_params(self, params)
            current_platform.validate_request(prompt, params)
        return lazy_generate(self, *args, **kwargs)

    setattr(generate, _EAGER_MARK, True)
    AsyncLLM.generate = generate


# ── client media errors ──────────────────────────────────────────────────────────


def _fetch_errors() -> tuple[type[BaseException], ...]:
    """What a failed ``http(s)`` media fetch raises: ``aiohttp`` on the async path,
    ``requests`` on the sync path, and the built-in timeout and connection errors."""
    import aiohttp
    import requests

    return (
        aiohttp.ClientError,
        requests.exceptions.RequestException,
        TimeoutError,
        ConnectionError,
    )


def client_media_error(connector, url: str, media_io, exc: BaseException):
    """The HTTP 400 refusal for a media URL the client got wrong, or None to let
    ``exc`` propagate as it is.

    Two cases are the client's: a ``file://`` URL on a server started without
    ``--allowed-local-media-path`` (vLLM raises ``RuntimeError``), and an ``http(s)``
    URL that cannot be fetched (connection refused, an HTTP error status, a timeout).
    Both fall to the internal-error branch of vLLM's error handler, which answers HTTP
    500. A ``ValueError`` is already a 400 and is left alone, as is every other error,
    which keeps the status vLLM gives it.
    """
    if isinstance(exc, ValueError):
        return None
    from urllib3.util import parse_url
    from vllm.exceptions import VLLMValidationError

    try:
        url_spec = parse_url(url)
    except Exception:  # an unparsable URL raised before the fetch; leave its error
        return None
    scheme = (url_spec.scheme or "").lower()
    parameter = _MEDIA_PARAMETERS.get(type(media_io).__name__)
    if (
        scheme == "file"
        and isinstance(exc, RuntimeError)
        and connector.allowed_local_media_path is None
    ):
        return VLLMValidationError(
            f"{exc} This server loads no local files: send the media as an http(s) URL "
            f"or a base64 data URL.",
            parameter=parameter,
        )
    if scheme.startswith("http") and isinstance(exc, _fetch_errors()):
        reason = type(exc).__name__ + (f": {exc}" if str(exc) else "")
        return VLLMValidationError(
            f"The media URL could not be fetched from {url_spec.host} ({reason}). Send "
            f"a URL this server can reach, or the media as a base64 data URL.",
            parameter=parameter,
        )
    return None


def install_client_media_errors() -> None:
    """Answer a media URL the client got wrong with HTTP 400 instead of 500.

    Wraps vLLM's ``MediaConnector.load_from_url`` and ``load_from_url_async``, which
    every image, video and audio URL goes through, so that :func:`client_media_error`
    re-raises the client's two cases as a ``VLLMValidationError`` naming the content
    part. vLLM's error handler answers that with 400 and the part as ``param``.
    Idempotent.
    """
    try:
        from vllm.multimodal.media import MediaConnector
    except ImportError:  # a vLLM without the media connector fetches no media
        return
    if getattr(MediaConnector.load_from_url_async, _MEDIA_MARK, False):
        return
    load = MediaConnector.load_from_url
    load_async = MediaConnector.load_from_url_async

    @functools.wraps(load)
    def load_from_url(self, url, media_io, *args, **kwargs):
        try:
            return load(self, url, media_io, *args, **kwargs)
        except Exception as exc:
            refusal = client_media_error(self, url, media_io, exc)
            if refusal is None:
                raise
            raise refusal from exc

    @functools.wraps(load_async)
    async def load_from_url_async(self, url, media_io, *args, **kwargs):
        try:
            return await load_async(self, url, media_io, *args, **kwargs)
        except Exception as exc:
            refusal = client_media_error(self, url, media_io, exc)
            if refusal is None:
                raise
            raise refusal from exc

    setattr(load_from_url, _MEDIA_MARK, True)
    setattr(load_from_url_async, _MEDIA_MARK, True)
    MediaConnector.load_from_url = load_from_url
    MediaConnector.load_from_url_async = load_from_url_async
