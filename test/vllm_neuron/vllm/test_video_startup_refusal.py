# SPDX-License-Identifier: Apache-2.0
"""A GLM-5.3-Flash server does not start with a video limit above 0.

The GLM-5.3-Flash bridge offers images only. vLLM reads ``--limit-mm-per-prompt`` only for
the modalities a bridge offers, so a video limit the operator sets would be ignored without
a word, and every video request would still be refused.
``NeuronPlatform.check_and_update_config`` refuses that limit at startup instead. A server
that leaves video out of the limit, or sets it to 0, starts.

The config is the stand-in of ``test_vision_merge_factor_refusal.py``, with the model class
declaring its merge factor so that refusal stays out of the way.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/test_video_startup_refusal.py
"""

from __future__ import annotations

import pytest
from vllm.config.multimodal import MultiModalConfig

from vllm_neuron.vllm.platform import NeuronPlatform

from test.vllm_neuron.vllm.test_vision_merge_factor_refusal import (  # noqa: F401 (fixtures)
    _config,
    _isolated,
    merge_factor_4,
)

pytestmark = [pytest.mark.fast]

REFUSAL = "serves image input only"


def test_a_video_limit_above_0_does_not_start(merge_factor_4):
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 2}))
    with pytest.raises(ValueError, match=REFUSAL) as refused:
        NeuronPlatform.check_and_update_config(cfg)
    message = str(refused.value)
    assert "--limit-mm-per-prompt" in message
    assert "2 video(s)" in message
    assert "Glm5NextForConditionalGeneration" in message
    assert '"video": 0' in message


def test_the_video_refusal_comes_before_the_merge_factor_one():
    """With both wrong, the operator first hears about the setting they gave."""
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 1}))
    with pytest.raises(ValueError, match=REFUSAL):
        NeuronPlatform.check_and_update_config(cfg)


@pytest.mark.parametrize(
    "limits",
    [{"image": 1, "video": 0}, {"image": 1}],
    ids=["video_0", "video_left_out"],
)
def test_a_server_without_a_video_limit_starts(merge_factor_4, limits):
    NeuronPlatform.check_and_update_config(_config(MultiModalConfig(limit_per_prompt=limits)))


def test_language_model_only_starts(merge_factor_4):
    multimodal = MultiModalConfig(limit_per_prompt={"video": 1}, language_model_only=True)
    NeuronPlatform.check_and_update_config(_config(multimodal))


def test_a_text_only_server_starts():
    NeuronPlatform.check_and_update_config(_config(None))


def test_a_model_whose_bridge_offers_video_is_not_refused():
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 1}))
    cfg.model_config.hf_config.architectures = ["Qwen3VLForConditionalGeneration"]
    NeuronPlatform._refuse_video_input(cfg)


def test_the_image_only_list_matches_the_bridge():
    """Every architecture in the list has a bridge, and that bridge offers no video."""
    from vllm_neuron.model.glm5_next.utils.vision_preprocessing import Glm5NextProcessingInfo

    bridges = {"Glm5NextForConditionalGeneration": Glm5NextProcessingInfo}
    assert NeuronPlatform._IMAGE_ONLY_ARCHS
    for arch in NeuronPlatform._IMAGE_ONLY_ARCHS:
        assert arch in bridges, f"{arch} is image-only, but this test names no bridge for it"
        assert "video" not in bridges[arch].get_supported_mm_limits(None)
