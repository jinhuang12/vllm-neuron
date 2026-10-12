# SPDX-License-Identifier: Apache-2.0
"""The per-image embedding limit: one number for the request check and the encoder budget.

The vision encoder is compiled for the ``num_vision_tokens_buckets`` sizes, counted in raw
(pre-merge) vision tokens. An image whose embeddings exceed the largest bucket divided by
the model's merge factor fits no compiled bucket. Two places enforce that limit:

* ``NeuronPlatform.validate_request`` refuses such an image with HTTP 400 before the request
  reaches the engine. The refusal is a ``VLLMValidationError`` whose parameter is the content
  part (``image_url``), and its message names the image's count, the limit, the setting the
  limit comes from, and a fix the client can apply.
* ``NeuronScheduler`` caps the per-step encoder budget at the same limit.

``vision_utils.get_vision_embed_limit`` computes the limit for both, so they cannot disagree.

The served line is built the way ``vllm serve`` builds it (``EngineArgs`` ->
``create_engine_config``) from the fixture checkpoint config. The merge factor of 4 that the
model surface will declare (``SupportsSpatialMerge``, ``spatial_merge_size ** 2``) is
reproduced by giving the model class that classmethod for the test.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/test_image_embed_limit.py
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
from types import SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import SamplingParams

from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration
from vllm_neuron.model.interfaces import SupportsSpatialMerge
from vllm_neuron.utils import vision_utils
from vllm_neuron.vllm import admission
from vllm_neuron.vllm.core.scheduler import NeuronScheduler
from vllm_neuron.vllm.platform import NeuronPlatform

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.vllm.test_admission import (
    ALL_GREEDY,
    KNOB,
    SERVED_GENERATION_CONFIG,
    STANDARD_LINE,
    _serve,
)

pytestmark = [pytest.mark.fast]

# The vision buckets the platform's auto-config gives the served line: one bucket of one
# 2048-token attention block.
SERVED_MAX_BUCKET = 2048
# spatial_merge_size 2 in the checkpoint's vision config: 2 x 2 raw tokens per embedding.
GLM_MERGE_FACTOR = 4
LIMIT = SERVED_MAX_BUCKET // GLM_MERGE_FACTOR  # 512


@pytest.fixture(scope="module")
def served_model_dir():
    """The fixture checkpoint config beside the served checkpoint's generation config."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="glm53f-image-limit-"))
    shutil.copy(fr.FIXTURE / "config.json", root / "config.json")
    (root / "generation_config.json").write_text(json.dumps(SERVED_GENERATION_CONFIG))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _restore_platform_policy():
    """Each config build replaces the class-level policy; give the next test the old one."""
    saved = NeuronPlatform._admission
    yield
    NeuronPlatform._admission = saved


@pytest.fixture
def merge_factor_4():
    """The model class declares the merge factor its vision config implies.

    ``SupportsSpatialMerge`` is a runtime-checkable protocol, and ``issubclass`` caches
    its answer per class, so the cache is cleared on both sides of the change.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            Glm5NextForConditionalGeneration,
            "get_vision_token_merge_factor",
            classmethod(lambda cls, hf_config: hf_config.vision_config.spatial_merge_size**2),
            raising=False,
        )
        SupportsSpatialMerge._abc_caches_clear()
        yield
    SupportsSpatialMerge._abc_caches_clear()


@pytest.fixture
def served(served_model_dir, monkeypatch, merge_factor_4):
    monkeypatch.setenv(KNOB, "1")
    return _serve(served_model_dir, STANDARD_LINE, sampler=ALL_GREEDY, async_scheduling=True)


def _image_request(num_embeds: int) -> dict:
    """A processed chat prompt with one image of ``num_embeds`` placeholder tokens."""
    return {
        "type": "multimodal",
        "prompt_token_ids": [11] * (num_embeds + 16),
        "mm_placeholders": {"image": [PlaceholderRange(offset=8, length=num_embeds)]},
    }


GREEDY = SamplingParams(temperature=0.0, max_tokens=8)


def test_the_limit_is_the_largest_bucket_over_the_merge_factor(served):
    limit = vision_utils.get_vision_embed_limit(served)
    assert limit == vision_utils.VisionEmbedLimit(
        max_bucket=SERVED_MAX_BUCKET, merge_factor=GLM_MERGE_FACTOR
    )
    assert limit.embeds == LIMIT


def test_an_image_above_the_limit_is_refused_naming_image_url_and_the_limit(served):
    with pytest.raises(VLLMValidationError) as refused:
        NeuronPlatform.validate_request(_image_request(LIMIT + 1), GREEDY)
    assert refused.value.parameter == "image_url"
    assert refused.value.value == LIMIT + 1
    message = str(refused.value)
    assert f"{LIMIT + 1} embedding tokens" in message
    assert f"at most {LIMIT}" in message
    assert "num_vision_tokens_buckets" in message
    assert f"merge factor {GLM_MERGE_FACTOR}" in message


def test_the_refusal_names_the_processor_key_this_model_reads(served):
    """transformers' GLM-5.3-Flash image processor caps an image with max_image_tokens,
    counted in embeddings, and does not read max_pixels."""
    with pytest.raises(VLLMValidationError) as refused:
        NeuronPlatform.validate_request(_image_request(LIMIT + 1), GREEDY)
    message = str(refused.value)
    assert f'"max_image_tokens": {LIMIT}' in message
    assert "max_pixels" not in message


def test_an_image_at_the_limit_is_admitted(served):
    NeuronPlatform.validate_request(_image_request(LIMIT), GREEDY)


def test_a_text_request_is_admitted(served):
    NeuronPlatform.validate_request({"type": "token", "prompt_token_ids": [11] * 64}, GREEDY)


def test_an_image_refusal_beside_another_refusal_names_both(served):
    """Two reasons, one 400: the message carries both, and no single parameter is named."""
    too_long = STANDARD_LINE["max_model_len"]
    request = _image_request(LIMIT + 1)
    request["prompt_token_ids"] = [11] * too_long
    with pytest.raises(ValueError) as refused:
        NeuronPlatform.validate_request(request, GREEDY)
    assert not isinstance(refused.value, VLLMValidationError)
    message = str(refused.value)
    assert "max_model_len" in message
    assert f"{LIMIT + 1} embedding tokens" in message


def test_the_encoder_budget_is_capped_at_the_same_limit(served):
    scheduler = NeuronScheduler.__new__(NeuronScheduler)
    scheduler.max_num_encoder_input_tokens = 8000
    scheduler._cap_encoder_budget_to_vision_bucket(served)
    assert scheduler.max_num_encoder_input_tokens == LIMIT
    # The request check holds the helper's value; it does not compute its own.
    assert NeuronPlatform._admission.image.embed_limit == vision_utils.get_vision_embed_limit(
        served
    )


def test_a_config_without_vision_buckets_has_no_limit():
    config = SimpleNamespace(additional_config={}, model_config=None)
    assert vision_utils.get_vision_embed_limit(config) is None


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("Glm5NextForConditionalGeneration", "max_image_tokens"),
        ("Qwen3VLForConditionalGeneration", "max_pixels"),
        ("LlamaForCausalLM", None),
    ],
)
def test_the_resize_hint_names_the_key_each_image_processor_reads(arch, expected):
    hint = admission.image_resize_kwarg(SimpleNamespace(architectures=[arch]))
    assert hint == expected
