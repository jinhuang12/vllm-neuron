"""Tests for the multimodal-input path of ``logit_validation``.

One synthetic image-bearing sample is driven through the entry point with a
caller-supplied ``generate_fn`` fake, so the dispatch is exercised on CPU
without a checkpoint or a device. Covered: the sample reaches ``generate_fn``
unchanged, the entry point answers with a plain ``bool``, and a text-only call
passes no multimodal keyword at all.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_neuron.accuracy.logit_validation import logit_validation


def _synthetic_sample() -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.zeros(1, 3, 4, 4)}


def test_multimodal_sample_reaches_generate_fn_unchanged() -> None:
    """The sample is handed to ``generate_fn`` as the same object it was passed as."""
    torch.manual_seed(0)
    vocab = 1024
    expected_logits = torch.randn(1, 1, vocab)
    sample = _synthetic_sample()
    calls: list[dict[str, Any]] = []

    def fake_generate_fn(input_ids, multimodal_inputs=None):
        calls.append(
            {
                "mm_kwarg_seen": multimodal_inputs is not None,
                "mm_is_the_same_object": multimodal_inputs is not None
                and multimodal_inputs[0]["pixel_values"] is sample["pixel_values"],
            }
        )
        return expected_logits.clone()

    returned = logit_validation(
        [[1, 2, 3]],
        fake_generate_fn,
        expected_logits,
        multimodal_inputs=[sample],
        test_device="cpu",
        colorize=False,
        visualize=False,
    )

    returned_type = f"{type(returned).__module__}.{type(returned).__qualname__}"

    assert calls and calls[0]["mm_kwarg_seen"] is True, (
        f"the multimodal branch was not taken: generate_fn calls={calls}"
    )
    assert calls[0]["mm_is_the_same_object"] is True, (
        "the synthetic sample did not reach generate_fn unchanged"
    )
    assert returned_type == "builtins.bool", (
        f"logit_validation returned {returned_type}, not a plain bool"
    )


def test_text_only_input_passes_no_multimodal_keyword() -> None:
    """Without multimodal inputs, ``generate_fn`` is called with no keywords at all."""
    torch.manual_seed(0)
    expected_logits = torch.randn(1, 1, 1024)
    seen: list[dict[str, Any]] = []

    def fake_generate_fn(input_ids, **kwargs):
        seen.append(dict(kwargs))
        return expected_logits.clone()

    logit_validation(
        [[1, 2, 3]],
        fake_generate_fn,
        expected_logits,
        test_device="cpu",
        colorize=False,
        visualize=False,
    )

    assert seen == [{}], (
        f"text-only path passed keyword arguments to generate_fn: {seen}"
    )
