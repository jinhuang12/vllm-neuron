# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding and the shadow draft under decode context parallelism: the admission.

``NeuronPlatform._refuse_speculative_decoding_under_dcp`` runs at
``decode_context_parallel_size > 1`` and admits two forms only when the paths they need are in
the tree, each read as a module attribute that ``is True``:

* the GLM-5.3-Flash shadow draft (``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT >= 1``, no speculative
  config) on ``mtp.DCP_DRAFT_SLOTS``: the draft iterations past the first take their latent
  slot from the DCP ownership map;
* a speculative config (method "mtp", GLM-5.3-Flash) on the conjunction of three: the T-row
  DCP partial-attention entry (``mla_decode.DCP_SPECULATIVE_ROWS``), the DSA indexer's T rows
  under DCP (``decode_trow.DCP_SPECULATIVE_ROWS``) and, with
  ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC`` set, the async step's DCP latent slots
  (``async_step.DCP_LATENT_SLOTS``).

Where a form is not admitted it is refused with today's message, byte for byte, and the
speculative refusal adds one sentence naming what is missing. Every test sets every
capability it reads, so the result does not depend on which members the tree holds. No test
builds an engine or touches a device.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from vllm_neuron.vllm.platform import NeuronPlatform

pytestmark = [pytest.mark.fast]

KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
ASYNC_KNOB = "VLLM_NEURON_GLM5NEXT_MTP_ASYNC"
GLM5_NEXT_ARCH = "Glm5NextForConditionalGeneration"
DCP_SIZES = (2, 4, 8)

DRAFT_SLOTS = ("vllm_neuron.model.glm5_next.mtp", "DCP_DRAFT_SLOTS")
ATTENTION_ROWS = ("vllm_neuron.functional.attention.mla_decode", "DCP_SPECULATIVE_ROWS")
INDEXER_ROWS = ("vllm_neuron.functional.dsa.decode_trow", "DCP_SPECULATIVE_ROWS")
ASYNC_SLOTS = ("vllm_neuron.functional.mtp.async_step", "DCP_LATENT_SLOTS")
SPECULATIVE = (ATTENTION_ROWS, INDEXER_ROWS, ASYNC_SLOTS)


def shadow_message(k: int, dcp: int) -> str:
    """The shadow-draft refusal as b5b516b3 words it (platform.py :1302-:1310)."""
    return (
        f"VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT={k} is not supported with "
        f"decode_context_parallel_size={dcp}: from its second draft iteration on, the "
        f"shadow draft computes the MTP layer's KV slots from the sequence's page index, "
        f"and a decode context parallel rank holds only its own pages. Unset "
        f"VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT or set decode_context_parallel_size=1."
    )


def speculative_message(k: int, dcp: int) -> str:
    """The speculative-config refusal as b5b516b3 words it (platform.py :1316-:1324)."""
    return (
        f"Speculative decoding is not supported with decode_context_parallel_size > 1 "
        f"(dcp={dcp}, num_speculative_tokens={k}): spec decode under DCP (item 5a T > 1 "
        f"follow-up) is not implemented. Disable speculative decoding or set "
        f"decode_context_parallel_size=1."
    )


def _config(dcp: int, k: int | None = None, *, method: str = "mtp",
            architecture: str = GLM5_NEXT_ARCH) -> SimpleNamespace:
    """The fields the refusal reads: the DCP size, the speculative config, the architecture."""
    speculative = None if k is None else SimpleNamespace(method=method, num_speculative_tokens=k)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        speculative_config=speculative,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
    )


def _set(monkeypatch, capability: tuple[str, str], present: bool) -> None:
    module = importlib.import_module(capability[0])
    if present:
        monkeypatch.setattr(module, capability[1], True, raising=False)
    else:
        monkeypatch.delattr(module, capability[1], raising=False)


def _speculative_tree(monkeypatch, *, attention: bool, indexer: bool, slots: bool) -> None:
    for capability, present in zip(SPECULATIVE, (attention, indexer, slots)):
        _set(monkeypatch, capability, present)


def _label(capability: tuple[str, str]) -> str:
    return f"{capability[0]}.{capability[1]}"


# --------------------------------------------------------------------------- #
# the shadow draft.


@pytest.mark.parametrize("draft_k", [1, 3])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_the_shadow_draft_is_admitted_when_its_slots_follow_ownership(monkeypatch, dcp, draft_k):
    monkeypatch.setenv(KNOB, str(draft_k))
    _set(monkeypatch, DRAFT_SLOTS, True)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp)) is None


@pytest.mark.parametrize("value", [False, 1, "True", None])
def test_only_a_true_draft_slot_attribute_admits_the_shadow_draft(monkeypatch, value):
    monkeypatch.setenv(KNOB, "3")
    module = importlib.import_module(DRAFT_SLOTS[0])
    if value is None:
        monkeypatch.delattr(module, DRAFT_SLOTS[1], raising=False)
    else:
        monkeypatch.setattr(module, DRAFT_SLOTS[1], value, raising=False)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2))
    assert str(refused.value) == shadow_message(3, 2)


def test_the_tree_publishes_the_draft_slot_attribute() -> None:
    module = importlib.import_module(DRAFT_SLOTS[0])
    assert getattr(module, DRAFT_SLOTS[1]) is True


# --------------------------------------------------------------------------- #
# a speculative config: the capability conjunction.


@pytest.mark.parametrize("async_knob", [None, "1"])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_speculative_mtp_is_admitted_when_every_capability_is_present(monkeypatch, dcp, async_knob):
    if async_knob is None:
        monkeypatch.delenv(ASYNC_KNOB, raising=False)
    else:
        monkeypatch.setenv(ASYNC_KNOB, async_knob)
    _speculative_tree(monkeypatch, attention=True, indexer=True, slots=True)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp, 3)) is None


def test_the_async_slots_are_not_needed_without_the_async_knob(monkeypatch):
    monkeypatch.delenv(ASYNC_KNOB, raising=False)
    _speculative_tree(monkeypatch, attention=True, indexer=True, slots=False)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3)) is None


@pytest.mark.parametrize(
    "attention, indexer, slots",
    [
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (True, True, False),
        (True, False, True),
    ],
)
def test_a_missing_capability_is_refused_with_todays_message_and_its_name(
    monkeypatch, attention, indexer, slots
):
    monkeypatch.setenv(ASYNC_KNOB, "1")
    _speculative_tree(monkeypatch, attention=attention, indexer=indexer, slots=slots)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3))
    message = str(refused.value)
    assert message.startswith(speculative_message(3, 2))
    for capability, present in zip(SPECULATIVE, (attention, indexer, slots)):
        assert (_label(capability) in message) is (not present), (capability, message)


def test_the_tree_alone_names_what_it_lacks(monkeypatch):
    """No capability is set here: the refusal reads the tree as it is, and names exactly
    the members' attributes that are not ``True`` in it."""
    monkeypatch.setenv(ASYNC_KNOB, "1")
    absent = [
        capability for capability in SPECULATIVE
        if getattr(importlib.import_module(capability[0]), capability[1], None) is not True
    ]
    if not absent:
        assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3)) is None
        return
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3))
    for capability in SPECULATIVE:
        assert (_label(capability) in str(refused.value)) is (capability in absent)


@pytest.mark.parametrize(
    "method, architecture",
    [("eagle3", GLM5_NEXT_ARCH), ("mtp", "LlamaForCausalLM"), ("mtp", "DeepseekV3ForCausalLM")],
)
def test_other_methods_and_architectures_stay_refused(monkeypatch, method, architecture):
    monkeypatch.setenv(ASYNC_KNOB, "1")
    _speculative_tree(monkeypatch, attention=True, indexer=True, slots=True)
    config = _config(2, 3, method=method, architecture=architecture)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(config)
    message = str(refused.value)
    assert message.startswith(speculative_message(3, 2))
    assert f"'{method}'" in message and architecture in message


def test_no_speculative_tokens_is_not_a_speculative_step(monkeypatch):
    _speculative_tree(monkeypatch, attention=False, indexer=False, slots=False)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 0)) is None


# --------------------------------------------------------------------------- #
# the message phrase the startup log is searched for.


def test_each_refusal_holds_its_phrase_on_one_source_line() -> None:
    """``grep 'is not supported with decode_context_parallel_size' platform.py`` finds both
    messages: the phrase sits in one string literal in each."""
    import inspect

    source = inspect.getsource(NeuronPlatform._refuse_speculative_decoding_under_dcp)
    lines = [line for line in source.splitlines()
             if "is not supported with decode_context_parallel_size" in line]
    assert len(lines) == 2, lines
