# SPDX-License-Identifier: Apache-2.0
"""A server that admits image or video input does not start at decode_context_parallel_size > 1.

Image prefill under decode context parallelism is not implemented: the runner's multimodal
embedding gather has no DCP path, and a prefill chunk that the scheduler stops at an image
start (when the encoder budget or cache cannot take the image that step) would begin the next
DCP prefill chunk at a cached length that is not a multiple of
``decode_context_parallel_size * block_size``, which the runner refuses in the step, after the
engine is serving. ``NeuronPlatform._validate_dcp_config`` refuses the configuration at
startup instead. A text-only server at DCP > 1, and a multimodal server at DCP 1, are not
refused here.

The config is the stand-in of ``test_vision_merge_factor_refusal.py`` with vLLM's own
``ParallelConfig`` at the DCP degree under test.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/test_multimodal_dcp_refusal.py
"""

from __future__ import annotations

import pytest
from vllm.config.multimodal import MultiModalConfig
from vllm.config.parallel import ParallelConfig

from vllm_neuron.vllm.platform import NeuronPlatform

from test.vllm_neuron.vllm.test_vision_merge_factor_refusal import (  # noqa: F401 (fixtures)
    _config,
    _isolated,
)

pytestmark = [pytest.mark.fast]

REFUSAL = "not supported with decode_context_parallel_size > 1"


def _dcp_config(multimodal: MultiModalConfig | None, *, dcp: int = 2, arch: str | None = None):
    cfg = _config(multimodal)
    cfg.parallel_config = ParallelConfig(
        tensor_parallel_size=64, decode_context_parallel_size=dcp
    )
    cfg.speculative_config = None
    # The DCP head rules read the text config off the model config, as vLLM's does.
    cfg.model_config.hf_text_config = cfg.model_config.hf_config.text_config
    if arch is not None:
        cfg.model_config.hf_config.architectures = [arch]
    return cfg


def test_a_server_that_admits_images_does_not_start_at_dcp_2():
    cfg = _dcp_config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 0}))
    with pytest.raises(ValueError, match=REFUSAL) as refused:
        NeuronPlatform._validate_dcp_config(cfg)
    message = str(refused.value)
    assert "dcp=2" in message
    assert "image" in message
    assert "--language-model-only" in message
    assert "decode_context_parallel_size=1" in message


def test_the_default_image_limit_is_refused_too():
    with pytest.raises(ValueError, match=REFUSAL):
        NeuronPlatform._validate_dcp_config(_dcp_config(MultiModalConfig()))


def test_a_video_limit_is_refused_for_a_model_that_offers_video():
    cfg = _dcp_config(
        MultiModalConfig(limit_per_prompt={"image": 0, "video": 1}),
        arch="Qwen3VLForConditionalGeneration",
    )
    with pytest.raises(ValueError, match=REFUSAL) as refused:
        NeuronPlatform._refuse_multimodal_input_under_dcp(cfg)
    assert "video" in str(refused.value)


@pytest.mark.parametrize(
    "multimodal",
    [
        None,
        MultiModalConfig(language_model_only=True),
        MultiModalConfig(limit_per_prompt={"image": 0, "video": 0}),
        # This bridge offers no video, so only the image limit admits input.
        MultiModalConfig(limit_per_prompt={"image": 0}),
    ],
    ids=["text_only", "language_model_only", "limits_0", "glm_image_0"],
)
def test_a_server_that_admits_no_image_or_video_is_not_refused(multimodal):
    NeuronPlatform._refuse_multimodal_input_under_dcp(_dcp_config(multimodal))


def test_a_multimodal_server_at_dcp_1_is_not_refused():
    cfg = _dcp_config(MultiModalConfig(limit_per_prompt={"image": 1}), dcp=1)
    NeuronPlatform._validate_dcp_config(cfg)
