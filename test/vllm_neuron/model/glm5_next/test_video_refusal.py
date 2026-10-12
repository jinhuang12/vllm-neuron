# SPDX-License-Identifier: Apache-2.0
"""A video content part is refused by name, before any media is fetched.

The GLM-5.3-Flash bridge offers images only (``get_supported_mm_limits``). vLLM's chat
parser asks the processing info to validate each content part before it fetches the media
(``validate_num_items``), and vLLM's own answer for a modality the bridge does not offer is
"At most 0 video(s) may be provided in one prompt.", with parameter ``video``: it names
neither the reason nor the fix. ``Glm5NextProcessingInfo.validate_num_items`` refuses a
video with a message that does, and names the content part (``video_url``) as the
parameter, so vLLM's 400 carries it as ``param``. Every other modality keeps vLLM's check.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_video_refusal.py
"""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace

import pytest
from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMValidationError

from vllm_neuron.model.glm5_next.utils.vision_preprocessing import Glm5NextProcessingInfo

pytestmark = [pytest.mark.fast]


def _info(**limits) -> Glm5NextProcessingInfo:
    """The processing info with an input context that carries only the multimodal config,
    which is all ``validate_num_items`` reads."""
    config = MultiModalConfig(limit_per_prompt=limits)
    return Glm5NextProcessingInfo(SimpleNamespace(get_mm_config=lambda: config))


def test_a_video_is_refused_naming_video_url():
    with pytest.raises(VLLMValidationError) as refused:
        _info(image=1).validate_num_items("video", 1)
    assert refused.value.parameter == "video_url"
    message = str(refused.value)
    assert "image input only" in message
    assert "video_url" in message
    assert "image_url" in message


def test_the_video_refusal_is_a_400_with_param_video_url():
    with pytest.raises(VLLMValidationError) as refused:
        _info(image=1).validate_num_items("video", 1)
    error = create_error_response(refused.value).error
    assert error.code == HTTPStatus.BAD_REQUEST
    assert error.type == "BadRequestError"
    assert error.param == "video_url"


def test_a_video_limit_the_operator_sets_does_not_admit_a_video():
    """The bridge offers no video, so a video limit above 0 changes nothing here."""
    with pytest.raises(VLLMValidationError) as refused:
        _info(image=1, video=4).validate_num_items("video", 1)
    assert refused.value.parameter == "video_url"


def test_an_image_within_the_limit_is_admitted():
    _info(image=1).validate_num_items("image", 1)


def test_images_beyond_the_limit_keep_vllms_refusal():
    with pytest.raises(VLLMValidationError) as refused:
        _info(image=1).validate_num_items("image", 2)
    assert refused.value.parameter == "image"
    assert "At most 1 image(s)" in str(refused.value)


def test_audio_keeps_vllms_refusal():
    with pytest.raises(VLLMValidationError) as refused:
        _info(image=1).validate_num_items("audio", 1)
    assert refused.value.parameter == "audio"
