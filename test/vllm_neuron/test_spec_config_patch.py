# SPDX-License-Identifier: Apache-2.0
"""The speculative-config patch: GLM-5.3-Flash accepted as an MTP target.

``vllm_neuron/vllm/patches/spec_config_patch.py`` teaches vLLM's
``SpeculativeConfig`` that a ``Glm5NextForConditionalGeneration`` checkpoint
carries its own multi-token-prediction draft layer, so
``--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'`` is
accepted instead of refused with upstream's ``Unsupported speculative method``.
Covered here: the engine config is built for k in {1, 3} and names the draft layer
count the checkpoint declares; the target's own config is not rewritten; a
checkpoint of another architecture is left to upstream untouched; a GLM checkpoint
without a draft layer is refused by name; applying the patch twice leaves one
wrapper layer and one added model type; importing the plugin wires the patch; the
plugin still loads in the production import order (``vllm`` first).

Every expectation is read off the fixture's ``config.json`` or the module's own
named constants; nothing here pins a model geometry.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap
import typing

import pytest

FIXTURE = pathlib.Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures"
TARGET_ARCHITECTURE = "Glm5NextForConditionalGeneration"


def _patch():
    from vllm_neuron.vllm.patches import spec_config_patch

    return spec_config_patch


def _speculative():
    import vllm.config.speculative as speculative

    return speculative


def _fixture_text_config() -> dict:
    return json.loads((FIXTURE / "config.json").read_text())["text_config"]


def _engine_config(spec: dict | None):
    """The engine config the tiny first-request test builds, with a speculative config."""
    from vllm.engine.arg_utils import EngineArgs

    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

    neuron_config = {
        "num_batched_tokens_buckets": [fr.PREFILL_BUCKET, e2e.E2E_MAX_SEQ_LEN],
        "num_seqs_buckets": [fr.DECODE_BATCH],
    }
    return EngineArgs(
        model=str(FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=e2e.E2E_MAX_SEQ_LEN,
        max_num_seqs=e2e.E2E_MAX_NUM_SEQS,
        max_num_batched_tokens=e2e.E2E_MAX_SEQ_LEN,
        block_size=fr.tiny.MLA_PAGE_SIZE,
        enforce_eager=True,
        enable_prefix_caching=False,
        speculative_config=spec,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


@pytest.mark.forked
@pytest.mark.parametrize("k", [1, 3])
def test_mtp_on_the_glm5next_target_builds_an_engine_config(k):
    """``method=mtp`` on the GLM checkpoint is accepted and names the checkpoint's draft layer."""
    patch = _patch()
    config = _engine_config({"method": "mtp", "num_speculative_tokens": k})
    spec = config.speculative_config

    assert spec.method == "mtp"
    assert spec.num_speculative_tokens == k
    assert spec.use_eagle(), "the runner keys its spec-decode paths on use_eagle()"
    assert spec.model == str(FIXTURE), "an mtp draft is the target checkpoint itself"

    draft = spec.draft_model_config.hf_config
    assert draft.model_type == patch.GLM5_NEXT_MTP_MODEL_TYPE
    assert draft.n_predict == _fixture_text_config()["num_nextn_predict_layers"]
    # No draft model class exists or is loaded: the runner builds the proposer from the
    # root's own head. The draft config therefore keeps the one architecture the
    # plugin registers, which is what lets ModelConfig validate it.
    assert spec.draft_model_config.architecture == TARGET_ARCHITECTURE


@pytest.mark.forked
def test_the_target_config_is_not_rewritten():
    """Only the draft's copy of the config is retyped; the target keeps its own model_type."""
    config = _engine_config({"method": "mtp", "num_speculative_tokens": 1})
    target = config.model_config.hf_config
    assert target.model_type == json.loads((FIXTURE / "config.json").read_text())["model_type"]
    assert target.architectures == [TARGET_ARCHITECTURE]
    assert not hasattr(target, "n_predict")


@pytest.mark.forked
def test_a_target_of_another_architecture_is_left_to_upstream():
    """A non-GLM config comes back as upstream returns it: same object, type untouched."""
    from transformers import PretrainedConfig

    patch = _patch()
    speculative = _speculative()
    patch.apply_spec_config_patch()

    config = PretrainedConfig(architectures=["LlamaForCausalLM"], model_type="llama")
    out = speculative.SpeculativeConfig.hf_config_override(config)

    assert out is config
    assert config.model_type == "llama"
    # Still outside the MTP list, so upstream's own ``Unsupported speculative method``
    # refusal stands for such a target, unchanged by this patch.
    assert config.model_type not in typing.get_args(speculative.MTPModelTypes)


@pytest.mark.forked
def test_a_glm_checkpoint_without_a_draft_layer_is_refused_by_name():
    """A GLM config declaring no ``num_nextn_predict_layers`` cannot serve as its own draft."""
    from transformers import PretrainedConfig

    patch = _patch()
    speculative = _speculative()
    patch.apply_spec_config_patch()

    config = PretrainedConfig(architectures=[TARGET_ARCHITECTURE], model_type="glm5_next")
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        speculative.SpeculativeConfig.hf_config_override(config)

    zero = PretrainedConfig(
        architectures=[TARGET_ARCHITECTURE],
        model_type="glm5_next",
        num_nextn_predict_layers=0,
    )
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        speculative.SpeculativeConfig.hf_config_override(zero)


@pytest.mark.forked
def test_applying_twice_leaves_one_wrapper_layer_and_one_added_type():
    patch = _patch()
    speculative = _speculative()
    patch.apply_spec_config_patch()
    patch.apply_spec_config_patch()

    override = speculative.SpeculativeConfig.hf_config_override
    inner = getattr(override, "__wrapped__", None)
    assert inner is not None, "the wrapper must expose upstream's override"
    assert inner.__module__ == speculative.__name__
    assert not hasattr(inner, "__wrapped__"), "exactly one wrapper layer"

    types = typing.get_args(speculative.MTPModelTypes)
    assert types.count(patch.GLM5_NEXT_MTP_MODEL_TYPE) == 1
    assert "mtp" in types, "upstream's own entries are kept"


@pytest.mark.forked
def test_importing_the_plugin_wires_the_patch():
    import vllm_neuron  # noqa: F401

    speculative = _speculative()
    patch = _patch()
    assert hasattr(speculative.SpeculativeConfig.hf_config_override, "__wrapped__")
    assert patch.GLM5_NEXT_MTP_MODEL_TYPE in typing.get_args(speculative.MTPModelTypes)


def test_plugin_loads_when_vllm_is_imported_first():
    """Production order -- ``import vllm`` first -- loads the plugin and binds the patch.

    In that order the plugin is loaded from inside ``vllm.utils.torch_utils``'s own
    initialisation, before ``vllm.config`` exists, so an eager import of
    ``vllm.config.speculative`` would walk back into the partially initialised
    module and fail the whole plugin. Import order is process-global, hence a child.
    """
    child = textwrap.dedent(
        """
        import sys
        import typing
        import vllm  # noqa: F401  -- production order: vllm first
        import vllm.config.speculative as speculative
        import vllm_neuron.vllm.patches.spec_config_patch as p

        print("PLUGIN_LOADED=%s" % ("vllm_neuron" in sys.modules))
        override = speculative.SpeculativeConfig.hf_config_override
        print("WRAPPED=%s" % hasattr(override, "__wrapped__"))
        print("TYPE_ADDED=%s" % (p.GLM5_NEXT_MTP_MODEL_TYPE in typing.get_args(speculative.MTPModelTypes)))
        print("BOUND=%s" % p._bound)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=300
    )
    keys = ("PLUGIN_LOADED", "WRAPPED", "TYPE_ADDED", "BOUND")
    readings = dict(
        line.split("=", 1)
        for line in completed.stdout.splitlines()
        if line.startswith(tuple(f"{key}=" for key in keys))
    )
    assert completed.returncode == 0, (
        f"rc={completed.returncode}\nstdout:\n{completed.stdout[-3000:]}\n"
        f"stderr:\n{completed.stderr[-3000:]}"
    )
    assert "Failed to load plugin" not in completed.stderr + completed.stdout
    assert readings == {
        "PLUGIN_LOADED": "True",
        "WRAPPED": "True",
        "TYPE_ADDED": "True",
        "BOUND": "True",
    }, readings
