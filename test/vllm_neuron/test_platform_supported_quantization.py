# SPDX-License-Identifier: Apache-2.0
"""The platform's quantization allowlist admits ``fp8``.

``Platform.verify_quantization`` reads ``NeuronPlatform.supported_quantization``
at config time, so an ``fp8`` checkpoint has to be admitted here before the
fork's own validator ever sees it. These tests read the class attribute and the
fixture checkpoint's config only: nothing selects a quantization method, builds
a ``QuantizationConfig`` or constructs an engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_neuron.vllm.platform import NeuronPlatform

ADMITTED = "fp8"
ALREADY_SUPPORTED = ("neuron_quant", "compressed-tensors", "modelopt")
EXPECTED_METHODS = ALREADY_SUPPORTED + (ADMITTED,)

# A method the allowlist must still refuse: the fork neither lists nor
# implements it.
REFUSED = "awq"

# Resolved off ``__file__`` so the read cannot depend on the invocation's cwd.
FIXTURE_CONFIG = (
    Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures" / "config.json"
)


def test_fp8_is_admitted_by_the_platform_allowlist() -> None:
    """``verify_quantization("fp8")`` returns without raising."""
    try:
        returned = NeuronPlatform.verify_quantization(ADMITTED)
    except ValueError as exc:
        pytest.fail(f"{ADMITTED!r} is still refused by the allowlist: {exc}")

    assert returned is None


def test_an_unlisted_method_is_still_refused() -> None:
    """The allowlist still refuses, and names the method it refused."""
    assert bool(NeuronPlatform.supported_quantization) is True, (
        "an empty supported_quantization disables the vLLM gate for EVERY "
        "method, which would pass the fp8 admission test vacuously"
    )

    with pytest.raises(ValueError) as excinfo:
        NeuronPlatform.verify_quantization(REFUSED)
    assert REFUSED in str(excinfo.value), (
        f"the refusal must name the method; message was {str(excinfo.value)!r}"
    )


def test_the_previously_supported_methods_are_still_listed() -> None:
    """Membership, never length: a custom registration may append to the list."""
    allowlist = NeuronPlatform.supported_quantization

    for method in EXPECTED_METHODS:
        assert method in allowlist, (
            f"expected method absent: {method!r}; allowlist={list(allowlist)}"
        )


def test_fixture_checkpoint_method_is_the_admitted_one() -> None:
    """The fixture checkpoint's ``quant_method`` is the admitted one."""
    assert FIXTURE_CONFIG.is_file(), f"fixture unreachable: {FIXTURE_CONFIG}"

    config = json.loads(FIXTURE_CONFIG.read_text())
    quant_method = config["quantization_config"]["quant_method"]

    assert quant_method == ADMITTED
    assert quant_method in NeuronPlatform.supported_quantization
