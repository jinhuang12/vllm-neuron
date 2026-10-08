# SPDX-License-Identifier: Apache-2.0
"""Request admission: refuse, on the request path, what this server cannot serve.

``NeuronPlatform.check_and_update_config`` resolves an :class:`AdmissionPolicy` from the
served config, in the API server process. ``NeuronPlatform.validate_request`` calls
:meth:`AdmissionPolicy.check` for every request, from vLLM's
``InputProcessor.process_inputs``, before the request becomes an ``EngineCoreRequest``.
A refusal is a ``ValueError``, which vLLM's OpenAI server returns as HTTP 400; the engine
never sees the request.

Three request classes are refused:

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
* **logprobs / prompt_logprobs when the device sampler returns token ids only.** The
  runner then has no logits on the host; vLLM's completions endpoint indexes an empty
  logprobs list and answers HTTP 500.

``validate_request`` alone answers a refused *streamed* request with HTTP 200 and an
error event: vLLM's ``AsyncLLM.generate`` is an async generator, so its body (where
``process_inputs`` runs) starts only once the response stream is iterated.
:func:`install_eager_validation` runs the same validation when ``generate`` is called,
before the stream begins, so streamed and plain requests both get a 400.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
}

#: Warnings already logged in this process, so each is logged once.
_WARNED: set[str] = set()

#: Marks the wrapped ``AsyncLLM.generate`` so a second install is a no-op.
_EAGER_MARK = "_vllm_neuron_eager_validation"


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

    ``all_greedy``: every token is the argmax. ``logprobs``: the host receives logits to
    compute logprobs from. ``prompt_logprobs``: likewise for the prompt. ``defaults``: the
    knob values a request carries when the client sets none of them.
    """

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
        return refusals

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


# ── the policy ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdmissionPolicy:
    prompt: PromptLength | None = None
    sampling: SamplingAdmission = field(default_factory=SamplingAdmission)

    def check(self, processed_inputs: Any, params: Any) -> None:
        """Raise ``ValueError`` naming every reason this request cannot be served."""
        refusals = []
        # PoolingParams carry no sampling knobs and generate no token, so neither the
        # length rule nor the sampling rules apply to them.
        if hasattr(params, "temperature"):
            if self.prompt is not None:
                tokens = prompt_length(processed_inputs)
                if tokens is not None and tokens > self.prompt.tokens:
                    refusals.append(self.prompt.refusal(tokens))
            refusals.extend(self.sampling.refusals(params))
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
            all_greedy=sampler.all_greedy,
            logprobs=logprobs,
            # The runner never fills prompt logprobs, and with the sampler on the
            # device there are no prompt logits on the host to fill them from.
            prompt_logprobs=False,
            defaults=_server_defaults(model_config),
        )
    policy = AdmissionPolicy(prompt=prompt, sampling=sampling)
    logger.info(
        "Request admission: prompts up to %d tokens (max_model_len %d less one generated "
        "token); on-device sampling %s",
        prompt.tokens,
        prompt.max_model_len,
        "off" if sampler is None
        else f"on (all_greedy={sampler.all_greedy}, logprobs={sampling.logprobs})",
    )
    return policy


# ── eager validation for streamed requests ───────────────────────────────────────


def install_eager_validation() -> None:
    """Run the platform's request validation when ``AsyncLLM.generate`` is called.

    ``generate`` is an async generator: ``process_inputs`` (and so ``validate_request``)
    runs on the first iteration, which for a streamed response is after vLLM has sent
    HTTP 200. Validating at the call instead makes the ``ValueError`` propagate out of
    the endpoint handler, where vLLM's exception handler returns 400, for streamed and
    plain requests alike. ``process_inputs`` validates again as before. Only a processed
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

            current_platform.validate_request(prompt, params)
        return lazy_generate(self, *args, **kwargs)

    setattr(generate, _EAGER_MARK, True)
    AsyncLLM.generate = generate
