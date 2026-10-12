# SPDX-License-Identifier: Apache-2.0
"""A server that admits images refuses to start when the merge factor is wrong.

The per-image limit and the encoder budget divide the largest vision bucket by
``get_vision_token_merge_factor``, which is 1 for a model class that does not implement
``SupportsSpatialMerge``. GLM-5.3-Flash's vision config merges 2 x 2 raw tokens into one
embedding (``spatial_merge_size`` 2), so a factor of 1 would make both 4 times too large.
``NeuronPlatform.check_and_update_config`` refuses that configuration when the server admits
image input. A text-only server, which is what this model serves until its class declares
``supports_multimodal``, starts as before.

The config is the stand-in of ``test_platform_hybrid_config.py`` (vLLM's own cache, parallel
and scheduler configs, a stand-in model config) with the fixture checkpoint's vision config
and vLLM's own ``MultiModalConfig``.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/test_vision_merge_factor_refusal.py
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from vllm.config.multimodal import MultiModalConfig

from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration
from vllm_neuron.model.interfaces import SupportsSpatialMerge
from vllm_neuron.vllm.platform import NeuronPlatform

from test.vllm_neuron.test_platform_hybrid_config import (
    FIXTURE_CONFIG,
    _build_config,
    _isolated_platform_class_state,
)

pytestmark = [pytest.mark.fast]

REFUSAL = "cannot serve image input"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Keep the class-level state and the OMP default a multimodal config sets."""
    monkeypatch.setenv("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    saved = NeuronPlatform._admission
    with _isolated_platform_class_state():
        yield
    NeuronPlatform._admission = saved


@pytest.fixture
def merge_factor_4():
    """The model class declares the merge factor its vision config implies.

    ``issubclass`` against the runtime-checkable protocol caches its answer per class, so
    the cache is cleared on both sides of the change.
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


def _config(multimodal: MultiModalConfig | None, *, vision: dict | None = None):
    """The stand-in config with the fixture checkpoint's vision config."""
    cfg = _build_config(neuron_config={"enable_hybrid_kv_cache": True})
    raw = json.loads(FIXTURE_CONFIG.read_text())["vision_config"] if vision is None else vision
    cfg.model_config.hf_config.vision_config = SimpleNamespace(**raw)
    cfg.model_config.multimodal_config = multimodal
    return cfg


def test_a_server_that_admits_images_with_merge_factor_1_does_not_start():
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 0}))
    with pytest.raises(ValueError, match=REFUSAL) as refused:
        NeuronPlatform.check_and_update_config(cfg)
    message = str(refused.value)
    assert "Glm5NextForConditionalGeneration" in message
    assert "spatial_merge_size=2" in message
    assert "merge factor of 1" in message
    assert "get_vision_token_merge_factor" in message
    assert "4 times too large" in message
    assert "return 4" in message


def test_the_default_image_limit_admits_images_too():
    """vLLM's limit for a modality the operator leaves out is 999."""
    with pytest.raises(ValueError, match=REFUSAL):
        NeuronPlatform.check_and_update_config(_config(MultiModalConfig()))


def test_a_server_that_admits_images_with_merge_factor_4_starts(merge_factor_4):
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 0}))
    NeuronPlatform.check_and_update_config(cfg)
    assert NeuronPlatform._admission.image.limit == 512


def test_a_text_only_server_starts():
    """The served form today: the class is not multimodal, so vLLM builds no
    multimodal config."""
    NeuronPlatform.check_and_update_config(_config(None))


@pytest.mark.parametrize(
    "multimodal",
    [
        MultiModalConfig(limit_per_prompt={"image": 0, "video": 0}),
        MultiModalConfig(language_model_only=True),
    ],
    ids=["image_limit_0", "language_model_only"],
)
def test_a_server_that_admits_no_image_starts(multimodal):
    NeuronPlatform.check_and_update_config(_config(multimodal))


def test_a_vision_config_that_does_not_merge_starts():
    raw = dict(json.loads(FIXTURE_CONFIG.read_text())["vision_config"], spatial_merge_size=1)
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1}), vision=raw)
    NeuronPlatform.check_and_update_config(cfg)
