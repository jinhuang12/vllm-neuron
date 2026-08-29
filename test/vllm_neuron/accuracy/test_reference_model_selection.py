"""``inc-glm53f-005`` item 1 and extraction-fidelity guard A.

Item 1 is a single equality: for the architecture string
``Glm5NextForConditionalGeneration`` the reference-model selector returns the
conditional-generation ``AutoModel`` class. The predicate is settled by
**calling** :func:`~vllm_neuron.accuracy.goldens.reference_model.select_hf_model_cls`,
never by re-implementing its branch here -- a test that re-implemented the
branch would pass whatever the module did.

Guard A is the extraction-fidelity check for that selector. The branch used to
live inline inside ``init_hf_model``, which cannot return without loading a
checkpoint from disk or the hub on every path, so the behaviour was not
observable at all in a CPU-mode unit test. The guard asserts the extracted
function still maps the pin's documented contract:
``"ForConditionalGeneration" in arch`` selects ``AutoModelForImageTextToText``
and everything else selects ``AutoModelForCausalLM``, over a declared table
exercising both sides.

Nothing here loads a checkpoint, opens a socket, or reads a config: the
selector's whole input is a string.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from vllm_neuron.accuracy.goldens.reference_model import (
    init_hf_model,
    select_hf_model_cls,
)

#: The architecture string this campaign registers. The pin's own branch
#: predicate already admits it: ``"ForConditionalGeneration" in
#: "Glm5NextForConditionalGeneration"`` is true.
GLM5NEXT_ARCH = "Glm5NextForConditionalGeneration"

#: Guard A's declared table, image-text-to-text side. The class this side must
#: select is the one ``reference_model``'s own class-choice comment names, so
#: the table states the module's documented contract rather than an invented
#: one.
CONDITIONAL_GENERATION_ARCHS = (
    GLM5NEXT_ARCH,
    "Qwen3VLForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
)

#: Guard A's declared table, causal-LM side. ``""`` is the case
#: ``init_hf_model`` reaches when a config declares no architectures at all: it
#: leaves ``arch`` empty and must still select a class.
CAUSAL_LM_ARCHS = (
    "LlamaForCausalLM",
    "GptOssForCausalLM",
    "Qwen3ForCausalLM",
    "MixtralForCausalLM",
    "",
)

_SINK = Path(
    os.environ.get("VLLM_NEURON_INC005_SELECTION_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc005_selection_readings.json"
)
_RECORD: dict[str, Any] = {}
_SINK.write_text("{}\n")  # truncate a stale run's values


def _rec(**values: Any) -> None:
    """Persist as we go, so a failing conjunct still leaves its readings behind."""
    _RECORD.update(values)
    _SINK.write_text(json.dumps(_RECORD, indent=2, sort_keys=True, default=str) + "\n")


def test_item1_glm5next_arch_selects_conditional_generation() -> None:
    """Item 1 -- 1/1. The selector is CALLED; its branch is not re-implemented."""
    from transformers import AutoModelForImageTextToText

    selected = select_hf_model_cls(GLM5NEXT_ARCH)
    _rec(
        interpreter=sys.executable,
        item1_arch=GLM5NEXT_ARCH,
        item1_selected=f"{selected.__module__}.{selected.__qualname__}",
        item1_expected=(
            f"{AutoModelForImageTextToText.__module__}."
            f"{AutoModelForImageTextToText.__qualname__}"
        ),
    )

    assert selected is AutoModelForImageTextToText, (
        f"item 1: select_hf_model_cls({GLM5NEXT_ARCH!r}) returned {selected!r}, "
        f"not AutoModelForImageTextToText"
    )


def test_guard_a_selector_maps_the_declared_arch_table() -> None:
    """Guard A -- 1/1 extraction fidelity for ``select_hf_model_cls``.

    Both sides of the declared table are exercised. A selector that always
    returned the conditional-generation class would satisfy item 1 alone; it
    fails here.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    mismatches: list[str] = []
    resolved: dict[str, str] = {}

    for arch in CONDITIONAL_GENERATION_ARCHS:
        got = select_hf_model_cls(arch)
        resolved[arch] = f"{got.__module__}.{got.__qualname__}"
        if got is not AutoModelForImageTextToText:
            mismatches.append(
                f"{arch!r} -> {got!r}, want AutoModelForImageTextToText"
            )

    for arch in CAUSAL_LM_ARCHS:
        got = select_hf_model_cls(arch)
        resolved[arch] = f"{got.__module__}.{got.__qualname__}"
        if got is not AutoModelForCausalLM:
            mismatches.append(f"{arch!r} -> {got!r}, want AutoModelForCausalLM")

    _rec(
        guard_a_table_size=len(CONDITIONAL_GENERATION_ARCHS) + len(CAUSAL_LM_ARCHS),
        guard_a_conditional_generation_count=len(CONDITIONAL_GENERATION_ARCHS),
        guard_a_causal_lm_count=len(CAUSAL_LM_ARCHS),
        guard_a_resolved=resolved,
        guard_a_mismatch_count=len(mismatches),
    )

    assert not mismatches, "guard A: " + "; ".join(mismatches)

    # The predicate is a property of the string, so state it as one: every arch
    # naming ForConditionalGeneration is on the first side and no arch on the
    # second side names it. This keeps the table honest if a row is ever added.
    for arch in CONDITIONAL_GENERATION_ARCHS:
        assert "ForConditionalGeneration" in arch, (
            f"guard A: table row {arch!r} is on the conditional-generation side "
            f"but does not contain 'ForConditionalGeneration'"
        )
    for arch in CAUSAL_LM_ARCHS:
        assert "ForConditionalGeneration" not in arch, (
            f"guard A: table row {arch!r} is on the causal-LM side but contains "
            f"'ForConditionalGeneration'"
        )


def test_init_hf_model_is_wired_to_the_extracted_selector() -> None:
    """The extraction is additive only if the caller actually uses it.

    ``init_hf_model`` cannot be called here -- it loads a checkpoint on every
    path -- so this is a presence predicate on the compiled function: the
    extracted name appears among the globals its body references. Without this,
    a selector could be authored, pass item 1 and guard A, and leave a stale
    inline copy of the branch running in production.
    """
    referenced = init_hf_model.__code__.co_names
    _rec(wiring_co_names_has_selector="select_hf_model_cls" in referenced)

    assert "select_hf_model_cls" in referenced, (
        "init_hf_model does not reference select_hf_model_cls; the extraction is "
        f"not wired in. Referenced globals: {sorted(referenced)}"
    )
