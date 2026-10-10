# SPDX-License-Identifier: Apache-2.0
"""On-device sampling over full-vocabulary logits.

The GLM-5.3-Flash root produces the FULL ``[B, vocab]`` logits on every rank (the head
is replicated today, and a vocab-sharded head all-gathers on device before it returns).
So this sampler reads no process group: each rank samples the same rows it holds. It is
the GLM root's logits -> token hand-off; it returns ``[B]`` int32 token ids, the form the
async runner feeds back as the next step's ``input_ids``.

B is any batch the decode graph is compiled for (up to 64). Each request brings
its own ``[top_k, top_p, temperature]`` row, the layout
``build_sampling_params_tensor`` builds.

Greedy rows are ``torch.argmax`` over the full row, not the top of a top-k: on a tie at
the maximum ``torch.argmax`` names the first index, which is what vLLM's host sampler
returns, while a top-k orders equal values in no promised way. bf16 logits tie often at
154,880 entries, so this keeps greedy output identical to the host sampler.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vllm_neuron.functional.sampling import _SAMPLING_EPS, sample


def sample_full_vocab(
    logits: Tensor,
    sampling_params: Tensor,
    sampling_config,
    logit_mask: Tensor | None = None,
) -> Tensor:
    """Sample one token per row of full-vocabulary ``logits``.

    Args:
        logits: ``[B, vocab]`` logits, every vocabulary entry present on this rank.
        sampling_params: ``[B, 3]`` float rows of ``[top_k, top_p, temperature]``.
        sampling_config: the ``OnDeviceSamplingConfig`` the server was started with;
            ``all_greedy``, ``max_top_k`` and ``deterministic`` are read from it.
        logit_mask: optional ``[B, vocab]`` bool mask, True where a token is allowed.

    Returns:
        ``[B]`` int32 token ids.

    Raises:
        ValueError: no sampling config, parameters for another batch, or a mask that
            does not cover the full vocabulary.
    """
    if sampling_config is None:
        raise ValueError(
            "full-vocabulary sampling was handed sampling parameters but the model has "
            "no on_device_sampling_config; serve it with an on-device sampling config "
            "or keep sampling on the host"
        )
    rows = int(logits.shape[0])
    if tuple(sampling_params.shape) != (rows, 3):
        raise ValueError(
            f"sampling parameters must be one row per request, [{rows}, 3] of "
            f"[top_k, top_p, temperature]; got {tuple(sampling_params.shape)}"
        )
    if logit_mask is not None:
        if tuple(logit_mask.shape) != tuple(logits.shape):
            raise ValueError(
                f"a full-vocabulary sampler needs a full-vocabulary mask of "
                f"{tuple(logits.shape)}; got {tuple(logit_mask.shape)}"
            )
        logits = logits.masked_fill(~logit_mask, float("-inf"))

    greedy = torch.argmax(logits, dim=-1).to(torch.int32)
    if sampling_config.all_greedy:
        return greedy

    temperature = sampling_params[:, 2]
    sampled = sample(
        logits,
        temperature=temperature,
        top_k=sampling_params[:, 0].to(torch.int32),
        top_p=sampling_params[:, 1],
        deterministic=sampling_config.deterministic,
        all_greedy=False,
        tp_group=None,
        max_top_k=sampling_config.max_top_k,
        capture_topk=sampling_config.capture_topk,
    )
    # Against a tensor, not the Python float: a float scalar lowers to an f64 compare,
    # which neuronx-cc refuses (NCC_ESPP004).
    is_greedy = temperature < torch.full_like(temperature, _SAMPLING_EPS)
    return torch.where(is_greedy, greedy, sampled)


def model_samples_on_device(model) -> bool:
    """True when the model was built with an on-device sampling config.

    Read off the model's own text config, through a ``torch.compile`` wrapper, because
    the model is what samples: a runner whose config asks for on-device sampling hands
    sampling parameters only to a model that was built to consume them.
    """
    model = getattr(model, "_orig_mod", model)
    text_config = getattr(model, "text_config", None)
    neuron_config = getattr(text_config, "neuron_config", None)
    return getattr(neuron_config, "on_device_sampling_config", None) is not None


def device_sampling_config(text_config):
    """The ``OnDeviceSamplingConfig`` a root was built with, or a named refusal.

    The root calls this before its stack runs when it is handed sampling parameters, so
    a root built without a sampler config refuses with nothing dispatched.
    """
    neuron_config = getattr(text_config, "neuron_config", None)
    config = getattr(neuron_config, "on_device_sampling_config", None)
    if config is None:
        raise ValueError(
            "this root was handed sampling parameters but was built without an "
            "on_device_sampling_config, so it has no sampler to hand its logits to"
        )
    return config


def device_sampling_kwargs(kwargs: dict, *, runner_samples_on_device: bool, model) -> dict:
    """The root keywords that carry on-device sampling, from the runner's generic ones.

    Empty unless the runner samples on device AND the model was built to: the root
    names these keywords itself (``device_sampling_params``, ``device_logit_mask``) so
    the generic ``sampling_params`` / ``logit_mask`` keys stay refused at its call.
    """
    if not (runner_samples_on_device and model_samples_on_device(model)):
        return {}
    params = kwargs.get("sampling_params")
    if params is None:
        raise ValueError(
            "on-device sampling is on and this call site handed no sampling_params; "
            "every model call site builds them when the runner samples on device"
        )
    out = {"device_sampling_params": params}
    if kwargs.get("logit_mask") is not None:
        out["device_logit_mask"] = kwargs["logit_mask"]
    return out
