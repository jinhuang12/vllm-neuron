# SPDX-License-Identifier: Apache-2.0
"""CONFIG-TIME quantisation allowlist, so an ``fp8`` checkpoint reaches the
fork's OWN validator (``inc-glm53f-075``).

Four items, one per declared conjunct, **no** ``parametrize`` -- the declared
count is 4 test-function definitions and stays derivable before the run by
``grep -c '^def test_' test/vllm_neuron/test_platform_supported_quantization.py``.

The subject is ``NeuronPlatform.supported_quantization``
(``vllm_neuron/vllm/platform.py`` L128-132) as read by
``Platform.verify_quantization`` (``vllm/platforms/interface.py``, guard at
L828) -- the exact line pair the chain in ``increments/evidence-074.md`` §10
raises at. This file authors admission at the vLLM platform allowlist and
nothing else: it selects no quantisation method, builds no
``QuantizationConfig``, and touches neither ``_validate_quantization_config``
nor ``_cpu_dequant_quantizations``.

**NO ITEM CONSTRUCTS AN ENGINE.** Every reading is taken at the class object or
at the pinned fixture bytes, so this acceptance can never depend on
``inc-glm53f-074`` and the two blocks cannot deadlock. No network, no
checkpoint, no device. Nothing under any compile cache or ``VLLM_CACHE_ROOT``
is read, written, deleted or relocated (P2).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from vllm_neuron.vllm.platform import NeuronPlatform

# The member this increment authors, and the pin's three it must not disturb.
ADMITTED = "fp8"
PIN_MEMBERS = ("neuron_quant", "compressed-tensors", "modelopt")
DECLARED_MEMBERS = PIN_MEMBERS + (ADMITTED,)

# A method the allowlist must still REFUSE, so item 2 is a differential rather
# than a second hope: the fork neither lists nor implements it.
REFUSED = "awq"

# The campaign's pinned checkpoint config -- blob
# 5d54bb5de98074e0ff8db6a455cb87adcf85501d, landed by inc-glm53f-008. Resolved
# off ``__file__`` so the read cannot depend on the invocation's cwd.
FIXTURE_CONFIG = (
    Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures" / "config.json"
)


def _record(label: str, value: object) -> None:
    """Emit an instrument-world reading into the run's own transcript.

    For values this increment does NOT author and therefore may not declare --
    see item 3, where the list LENGTH is an instrument-world number because
    ``vllm/model_executor/layers/quantization/__init__.py`` L95-96 appends to
    this very list at custom-registration time. Recorded, never asserted.
    """
    warnings.warn(f"RECORDED {label}={value!r}", stacklevel=2)


def test_fp8_is_admitted_by_the_platform_allowlist() -> None:
    """Item 1 -- the gate evidence-074 measured raising now admits ``fp8``."""
    try:
        returned = NeuronPlatform.verify_quantization(ADMITTED)
    except ValueError as exc:  # the L828-830 guard, i.e. the -074 blocker
        pytest.fail(f"{ADMITTED!r} is still refused by the allowlist: {exc}")

    # "Returns None and raises nothing": the raise would have failed above.
    assert returned is None


def test_the_allowlist_is_still_an_allowlist() -> None:
    """Item 2 -- the false-pass differential; an emptied list fails HERE."""
    assert bool(NeuronPlatform.supported_quantization) is True, (
        "an empty supported_quantization disables the vLLM gate for EVERY "
        "method (interface.py L828 short-circuits on the falsy list), which "
        "would pass item 1 vacuously"
    )

    with pytest.raises(ValueError) as excinfo:
        NeuronPlatform.verify_quantization(REFUSED)
    assert REFUSED in str(excinfo.value), (
        f"the refusal must name the method; message was {str(excinfo.value)!r}"
    )


def test_the_three_pin_members_survive() -> None:
    """Item 3 -- 4/4 DECLARED members present by membership; LENGTH recorded."""
    allowlist = NeuronPlatform.supported_quantization
    # Instrument-world readings: recorded here, asserted nowhere.
    _record("supported_quantization", list(allowlist))
    _record("len(supported_quantization)", len(allowlist))

    for member in DECLARED_MEMBERS:
        assert member in allowlist, (
            f"declared member absent: {member!r}; allowlist={list(allowlist)}"
        )

    # 4/4 counts THIS increment's declared members (repo-world authorship),
    # never ``len(allowlist)`` -- a custom registration may append to the list.
    present = [member for member in DECLARED_MEMBERS if member in allowlist]
    assert len(present) == 4


def test_the_campaign_fixture_method_is_the_admitted_one() -> None:
    """Item 4 -- the join: the pinned fixture's method IS the admitted one."""
    assert FIXTURE_CONFIG.is_file(), f"pinned fixture unreachable: {FIXTURE_CONFIG}"

    config = json.loads(FIXTURE_CONFIG.read_text())
    quant_method = config["quantization_config"]["quant_method"]
    _record("fixture quant_method", quant_method)

    assert quant_method == ADMITTED
    assert quant_method in NeuronPlatform.supported_quantization
