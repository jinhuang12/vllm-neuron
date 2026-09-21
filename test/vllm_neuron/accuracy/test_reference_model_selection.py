"""Tests for the reference model's Hugging Face class selection.

``select_hf_model_cls`` maps an architecture string to the ``AutoModel`` class
the reference model is built from: an architecture naming
``ForConditionalGeneration`` selects ``AutoModelForImageTextToText``, anything
else selects ``AutoModelForCausalLM``. The selector's whole input is a string,
so the mapping is exercised without loading a checkpoint or reading a config.
"""

from __future__ import annotations

from vllm_neuron.accuracy.goldens.reference_model import (
    init_hf_model,
    select_hf_model_cls,
)

GLM5NEXT_ARCH = "Glm5NextForConditionalGeneration"

#: Architectures that must select the image-text-to-text class.
CONDITIONAL_GENERATION_ARCHS = (
    GLM5NEXT_ARCH,
    "Qwen3VLForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
)

#: Architectures that must select the causal-LM class. ``""`` is the case
#: ``init_hf_model`` reaches when a config declares no architectures at all: it
#: leaves ``arch`` empty and must still select a class.
CAUSAL_LM_ARCHS = (
    "LlamaForCausalLM",
    "GptOssForCausalLM",
    "Qwen3ForCausalLM",
    "MixtralForCausalLM",
    "",
)


def test_glm5next_arch_selects_image_text_to_text() -> None:
    """The GLM-5 Next architecture selects ``AutoModelForImageTextToText``."""
    from transformers import AutoModelForImageTextToText

    selected = select_hf_model_cls(GLM5NEXT_ARCH)

    assert selected is AutoModelForImageTextToText, (
        f"select_hf_model_cls({GLM5NEXT_ARCH!r}) returned {selected!r}, not "
        "AutoModelForImageTextToText"
    )


def test_selector_maps_both_sides_of_the_arch_table() -> None:
    """Every architecture in the table selects the class its side expects."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    mismatches: list[str] = []

    for arch in CONDITIONAL_GENERATION_ARCHS:
        got = select_hf_model_cls(arch)
        if got is not AutoModelForImageTextToText:
            mismatches.append(
                f"{arch!r} -> {got!r}, expected AutoModelForImageTextToText"
            )

    for arch in CAUSAL_LM_ARCHS:
        got = select_hf_model_cls(arch)
        if got is not AutoModelForCausalLM:
            mismatches.append(f"{arch!r} -> {got!r}, expected AutoModelForCausalLM")

    assert not mismatches, "; ".join(mismatches)


def test_init_hf_model_uses_the_selector() -> None:
    """``init_hf_model`` picks its model class through ``select_hf_model_cls``."""
    # init_hf_model cannot be called here: it loads a checkpoint on every path,
    # so the wiring is read from the compiled function's global references.
    referenced = init_hf_model.__code__.co_names

    assert "select_hf_model_cls" in referenced, (
        "init_hf_model does not reference select_hf_model_cls, so an inline copy "
        f"of the branch may still be live. Referenced globals: {sorted(referenced)}"
    )
