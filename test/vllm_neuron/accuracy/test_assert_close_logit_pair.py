"""Tests for ``assert_close_logit_pair``, the logit-pair comparison adapter.

Covered here: the result type the adapter returns, adapting a pair that
``logit_validation`` hands to its ``logit_pair_sink`` callback, where the
default tolerances come from, and what an uncomparable pair raises.

Everything runs on CPU over synthetic tensors: no checkpoint, no device.
"""

from typing import Dict, Tuple

import pytest
import torch

import vllm_neuron.accuracy.testing as _testing
from vllm_neuron.accuracy.logit_validation import logit_validation
from vllm_neuron.accuracy.testing import (
    AssertCloseResult,
    assert_close_logit_pair,
    resolve_dtype_tolerance,
)

#: Vocabulary width for the synthetic logits. Small enough to stay instant.
VOCAB = 1024

#: The one sample index a single-element batch exposes.
SAMPLE_INDEX = 0


def _image_bearing_sample() -> Dict[str, torch.Tensor]:
    return {"pixel_values": torch.zeros(1, 3, 4, 4)}


def _exposed_logit_pairs() -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Run one image-bearing sample through ``logit_validation`` and collect its pairs."""
    torch.manual_seed(0)
    expected_logits = torch.randn(1, 1, VOCAB)
    exposed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def fake_generate_fn(input_ids, **kwargs):
        return expected_logits.clone()

    def collect(sample_index, actual, expected):
        exposed[sample_index] = (actual, expected)

    logit_validation(
        [[1, 2, 3]],
        fake_generate_fn,
        expected_logits,
        multimodal_inputs=[_image_bearing_sample()],
        logit_pair_sink=collect,
        test_device="cpu",
        colorize=False,
        visualize=False,
    )
    return exposed


def test_an_exposed_logit_pair_adapts_to_a_result() -> None:
    """The pair the sink receives converts into an ``AssertCloseResult``."""
    exposed = _exposed_logit_pairs()

    assert len(exposed) == 1, (
        f"expected the pair of exactly one sample to be exposed; got {sorted(exposed)}"
    )

    joined = assert_close_logit_pair(*exposed[SAMPLE_INDEX], name="exposed-pair")

    assert isinstance(joined, AssertCloseResult), (
        f"the adapter returned {type(joined)!r}, not an AssertCloseResult"
    )
    # Class identity, not a name match: a same-named look-alike must not pass.
    assert type(joined) is _testing.AssertCloseResult, (
        f"the returned class {type(joined)!r} is not the class testing.py defines"
    )
    assert joined.allclose is True, (
        "the pair is a tensor against its own clone and must compare equal; got "
        f"allclose={joined.allclose!r}"
    )


def test_default_tolerances_resolve_from_the_expected_dtype() -> None:
    """Omitted ``rtol``/``atol`` behave exactly like ``resolve_dtype_tolerance``."""
    torch.manual_seed(0)
    expected = torch.randn(4, VOCAB)
    # Not a clone: a bit-equal pair compares equal before any tolerance is read,
    # which would make the readings below agree for any value.
    actual = expected + torch.ones_like(expected)
    resolved_rtol, resolved_atol = resolve_dtype_tolerance(expected.dtype)

    defaulted = assert_close_logit_pair(actual, expected, name="defaulted")
    explicit = assert_close_logit_pair(
        actual,
        expected,
        rtol=resolved_rtol,
        atol=resolved_atol,
        name="explicit",
    )
    other_rtol = assert_close_logit_pair(
        actual, expected, rtol=1.0e3, atol=resolved_atol, name="other-rtol"
    )
    other_atol = assert_close_logit_pair(
        actual, expected, rtol=resolved_rtol, atol=2.0, name="other-atol"
    )

    assert defaulted.allclose == explicit.allclose, (
        "the defaulted rtol does not match the resolved value for "
        f"{expected.dtype}: defaulted allclose={defaulted.allclose}, "
        f"explicit={explicit.allclose}"
    )
    assert defaulted.max_rel_error == explicit.max_rel_error, (
        "the defaulted atol does not match the resolved value for "
        f"{expected.dtype}: defaulted max_rel_error={defaulted.max_rel_error}, "
        f"explicit={explicit.max_rel_error}"
    )
    assert other_rtol.allclose != defaulted.allclose, (
        "a different rtol produced the same verdict, so the agreement above "
        "holds for any tolerance and proves nothing"
    )
    assert other_atol.max_rel_error != defaulted.max_rel_error, (
        "a different atol produced the same max_rel_error, so the agreement "
        "above holds for any tolerance"
    )


def test_an_uncomparable_pair_raises() -> None:
    """A pair that cannot be compared raises; one that compares badly does not."""
    prefix = "uncomparable logit pair"
    torch.manual_seed(0)
    reference = torch.randn(4, VOCAB)

    with pytest.raises(TypeError) as not_a_tensor:
        assert_close_logit_pair([reference], reference, name="not-a-tensor")
    with pytest.raises(ValueError) as shape_mismatch:
        assert_close_logit_pair(reference[:2], reference, name="shape-mismatch")
    with pytest.raises(ValueError) as dtype_mismatch:
        assert_close_logit_pair(
            reference.to(torch.bfloat16), reference, name="dtype-mismatch"
        )

    raised = {
        "not_a_tensor": str(not_a_tensor.value),
        "shape_mismatch": str(shape_mismatch.value),
        "dtype_mismatch": str(dtype_mismatch.value),
    }
    for arm, message in raised.items():
        assert message.startswith(prefix), (
            f"the {arm} case raised without the {prefix!r} prefix: {message!r}"
        )
        assert "offending_type=" in message, (
            f"the {arm} case raised without naming the offending type: {message!r}"
        )
    assert "<class 'list'>" in raised["not_a_tensor"], (
        f"the non-tensor case did not carry the offending type: {raised['not_a_tensor']!r}"
    )
    assert "torch.bfloat16" in raised["dtype_mismatch"], (
        f"the dtype case did not carry the offending dtype: {raised['dtype_mismatch']!r}"
    )

    comparable_but_failing = assert_close_logit_pair(
        reference + torch.full_like(reference, float(VOCAB)),
        reference,
        name="far-apart",
    )
    assert isinstance(comparable_but_failing, AssertCloseResult), (
        "a comparable pair must still return a result, even when it compares "
        f"badly; got {type(comparable_but_failing)!r}"
    )
    assert comparable_but_failing.allclose is False, (
        "the far-apart pair was supposed to compare badly, so the raise-versus-"
        f"return distinction is untested; got allclose={comparable_but_failing.allclose!r}"
    )
