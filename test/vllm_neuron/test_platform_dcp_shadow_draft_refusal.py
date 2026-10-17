# SPDX-License-Identifier: Apache-2.0
"""The platform admits the GLM-5.3-Flash draft forms under DCP only on their capabilities.

``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT=k`` (k >= 1) makes the model build its MTP head and
run ``k`` draft iterations a step with no speculative config. A draft iteration past the
first computed the MTP layer's KV slot as ``table[position // page] * page + position %
page`` over the request's block table (``Glm5NextMultiTokenPredictor._latent_slots_at``),
as it still does at DCP 1.
Under decode context parallelism a rank's table lists only the pages it owns, so that
index names the wrong page. ``NeuronPlatform._refuse_speculative_decoding_under_dcp`` refused a
speculative config at ``decode_context_parallel_size > 1``; with no speculative config
it returned before the knob was seen. These tests pin, in the platform's DCP validation:

1. GLM-5.3-Flash at DCP 2, 4 and 8 with the knob at 1 or more and no speculative config:
   refused when the head does not say its draft slots follow the DCP ownership map
   (``mtp.DCP_DRAFT_SLOTS`` False), and the message names the knob, its value and the DCP
   size; admitted when it does;
2. DCP 2, 4 and 8 with the knob unset or 0: not refused;
3. DCP 1 with the knob set: the DCP rules do not run, so it is not refused;
4. another architecture at DCP 2 with the knob set: not refused, as no other
   architecture reads the knob;
5. DCP 2, 4 and 8 with a speculative config (method "mtp", 3 tokens), with the knob unset
   and set and with ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC`` unset and set: admitted when every
   capability the form counts is True (the T-row attention entry, the indexer's T rows,
   the draft head's ownership slots and, with the async drafter, its latent slots);
   refused with the speculative message, naming the capability, when one of them is
   False; the async drafter's latent slots are not counted without the async drafter.

No test constructs an engine, loads a checkpoint or touches a device.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from vllm_neuron.vllm.platform import NeuronPlatform

KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
GLM5_NEXT_ARCH = "Glm5NextForConditionalGeneration"
OTHER_ARCH = "LlamaForCausalLM"
DCP_SIZES = (2, 4, 8)
SPECULATIVE_REFUSAL = "Speculative decoding is not supported with decode_context_parallel_size > 1"
ASYNC_KNOB = "VLLM_NEURON_GLM5NEXT_MTP_ASYNC"
#: The capabilities the platform admits a speculative config on under DCP: (module,
#: attribute, counted only with the async drafter).
SPECULATIVE_CAPABILITIES = (
    ("vllm_neuron.functional.attention.mla_decode", "DCP_SPECULATIVE_ROWS", False),
    ("vllm_neuron.functional.dsa.decode_trow", "DCP_SPECULATIVE_ROWS", False),
    ("vllm_neuron.model.glm5_next.mtp", "DCP_DRAFT_SLOTS", False),
    ("vllm_neuron.functional.mtp.async_step", "DCP_LATENT_SLOTS", True),
)


def _config(
    dcp: int, num_speculative_tokens: int | None = None, architecture: str = GLM5_NEXT_ARCH
) -> SimpleNamespace:
    """The fields the refusal reads: the DCP size, the speculative config and the
    architecture."""
    speculative = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(method="mtp", num_speculative_tokens=num_speculative_tokens)
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        speculative_config=speculative,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
    )


@pytest.mark.parametrize("draft_k", [1, 3])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_shadow_draft_under_dcp_is_refused_by_name(monkeypatch, dcp, draft_k):
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.setenv(KNOB, str(draft_k))
    monkeypatch.setattr(mtp, "DCP_DRAFT_SLOTS", False)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp))
    message = str(refused.value)
    assert f"{KNOB}={draft_k}" in message
    assert f"decode_context_parallel_size={dcp}" in message


def test_shadow_draft_under_dcp_is_admitted_when_its_slots_follow_ownership(monkeypatch):
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.setenv(KNOB, "3")
    monkeypatch.setattr(mtp, "DCP_DRAFT_SLOTS", True)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2)) is None


@pytest.mark.parametrize("value", [None, "0"])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_dcp_without_the_shadow_draft_is_not_refused(monkeypatch, dcp, value):
    if value is None:
        monkeypatch.delenv(KNOB, raising=False)
    else:
        monkeypatch.setenv(KNOB, value)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp)) is None


def test_the_shadow_draft_at_dcp_1_is_not_refused(monkeypatch):
    monkeypatch.setenv(KNOB, "3")
    assert NeuronPlatform._validate_dcp_config(_config(1)) is None


def test_the_shadow_draft_of_another_architecture_is_not_refused(monkeypatch):
    monkeypatch.setenv(KNOB, "3")
    config = _config(2, architecture=OTHER_ARCH)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(config) is None


def _env(monkeypatch, name: str, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def _capabilities(monkeypatch, missing: int | None = None) -> None:
    """Every speculative capability True, but the one at index ``missing`` False."""
    for index, (module, attribute, _async_only) in enumerate(SPECULATIVE_CAPABILITIES):
        monkeypatch.setattr(importlib.import_module(module), attribute, index != missing)


@pytest.mark.parametrize("mtp_async", [None, "1"])
@pytest.mark.parametrize("value", [None, "3"])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_speculative_decoding_under_dcp_is_admitted_on_every_capability(
    monkeypatch, dcp, value, mtp_async
):
    _env(monkeypatch, KNOB, value)
    _env(monkeypatch, ASYNC_KNOB, mtp_async)
    _capabilities(monkeypatch)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp, 3)) is None


@pytest.mark.parametrize("missing", range(len(SPECULATIVE_CAPABILITIES)))
@pytest.mark.parametrize("mtp_async", [None, "1"])
@pytest.mark.parametrize("value", [None, "3"])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_speculative_decoding_under_dcp_stays_refused_without_a_capability(
    monkeypatch, dcp, value, mtp_async, missing
):
    _env(monkeypatch, KNOB, value)
    _env(monkeypatch, ASYNC_KNOB, mtp_async)
    _capabilities(monkeypatch, missing)
    module, attribute, async_only = SPECULATIVE_CAPABILITIES[missing]
    if async_only and mtp_async is None:
        # Without the async drafter its latent slots are not counted.
        assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp, 3)) is None
        return
    with pytest.raises(ValueError, match=SPECULATIVE_REFUSAL) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp, 3))
    message = str(refused.value)
    named = [f"{m}.{a}" for m, a, _ in SPECULATIVE_CAPABILITIES if f"{m}.{a}" in message]
    assert named == [f"{module}.{attribute}"], message
