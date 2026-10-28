# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 image input at config time: processor, encoder protocols, refusal.

The engine's config is built for the checkpoint in CPU mode with no workers.
A fresh process stands for a spawned worker: it registers the plugin's model
classes as the worker does, then reads the config the engine built. Set
``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``.

The platform serves Kimi K3 as text only, so its request check refuses an image
over the largest bucket with the text-only message, before it counts the image's
embedding tokens. The bucket check is read with Kimi K3 taken out of the
platform's text-only set, in this process only.
"""

import multiprocessing
import os
import pickle
import traceback
from pathlib import Path

import pytest

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

# The served encoder: one bucket per registered image size, block size 5,504.
SERVED_BLOCK = 5504
REGISTERED_BUCKETS = [5504, 33536]
ARCH = "KimiK3ForConditionalGeneration"


def _image(size: int):
    from PIL import Image

    return Image.new("RGB", (size, size), (128, 64, 32))


def _expansion(processor, side: int) -> dict:
    """Expand one placeholder with one square image; return the marker layout."""
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    info = processor.info
    tokenizer = info.get_tokenizer()
    prompt = f"{info.get_hf_config().image_placeholder}Describe this image."
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    items = info.parse_mm_data({"image": [_image(side)]})
    out = processor.apply(ProcessorInputs(list(ids), items), TimingContext())
    tokens = list(out["prompt_token_ids"])
    begin = tokenizer.convert_tokens_to_ids("<|media_begin|>")
    end = tokenizer.convert_tokens_to_ids("<|media_end|>")
    pad = info.media_token_id
    first, last = tokens.index(begin), tokens.index(end)
    return {
        "begins": tokens.count(begin),
        "ends": tokens.count(end),
        "pads": tokens.count(pad),
        "pads_between": tokens[first:last].count(pad),
    }


def _worker_view(config_bytes: bytes, queue) -> None:
    """Resolve the K3 class and its processor as a freshly spawned worker does."""
    try:
        from vllm import ModelRegistry
        from vllm.model_executor.model_loader import get_model_architecture
        from vllm.multimodal import MULTIMODAL_REGISTRY

        from vllm_neuron.model import registry

        model_config = pickle.loads(config_bytes)
        for arch, model_cls in registry.get_models():
            ModelRegistry.register_model(arch, model_cls)
        model_cls, _ = get_model_architecture(model_config)
        image_input = MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config)
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        queue.put(
            {
                "model_cls": model_cls,
                "image_input": image_input,
                "processor_cls": type(processor),
                "expansion": _expansion(processor, 1024),
            }
        )
    except Exception:
        queue.put({"error": traceback.format_exc()})


@pytest.fixture(scope="module")
def model_config():
    from vllm.config import ModelConfig

    return ModelConfig(
        model=str(CHECKPOINT),
        tokenizer=str(CHECKPOINT),
        trust_remote_code=True,
        max_model_len=131072,
        limit_mm_per_prompt={"image": 7},
    )


@pytest.fixture(scope="module")
def worker_view(model_config) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_worker_view, args=(pickle.dumps(model_config), queue))
    child.start()
    view = queue.get(timeout=900)
    child.join()
    return view


def test_worker_resolves_plugin_class_with_image_input(worker_view):
    from vllm.models.kimi_k3.common.mm_preprocess import KimiK3MultiModalProcessor

    from vllm_neuron.model.kimi_k3 import KimiK3ForConditionalGeneration
    from vllm_neuron.model.kimi_k3.vision_processor import KimiK3Uint8Processor

    assert "error" not in worker_view, worker_view.get("error")
    assert worker_view["model_cls"] is KimiK3ForConditionalGeneration
    assert worker_view["image_input"] is True
    assert worker_view["processor_cls"] is KimiK3Uint8Processor
    assert issubclass(KimiK3Uint8Processor, KimiK3MultiModalProcessor)


def test_protocols_at_the_served_bucket(model_config):
    from vllm_neuron.model.neuron_config import EncoderNeuronConfig
    from vllm_neuron.utils.mm_utils import (
        get_max_encoder_blocks_per_request,
        get_max_encoder_embed_budget,
        get_vision_token_merge_factor,
    )

    hf_config = model_config.hf_config
    served = EncoderNeuronConfig.from_dict(
        {
            "encoder_attention_block_size": SERVED_BLOCK,
            "num_encoder_input_buckets": [SERVED_BLOCK],
        }
    )
    assert (
        get_vision_token_merge_factor(hf_config),
        get_max_encoder_embed_budget(hf_config, served),
        get_max_encoder_blocks_per_request(hf_config, served),
    ) == (4, 1376, 1)


def test_one_1024_image_expands_to_1369_pads(worker_view):
    assert "error" not in worker_view, worker_view.get("error")
    assert worker_view["expansion"] == {
        "begins": 1,
        "ends": 1,
        "pads": 1369,
        "pads_between": 1369,
    }


def _oversize_image_outcome(model_config, monkeypatch, gate_lifted: bool):
    """Send one image over the largest bucket through the engine's input step.

    Return the refusal messages and the count of requests admitted. The config
    step sets the platform's request-check state on the class, so each item
    patches that state first and gets it back at its end.
    """
    from vllm.config import SchedulerConfig, VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    from vllm_neuron.vllm.platform import NeuronPlatform

    for name in ("_text_only_architecture", "_max_embeds_per_image"):
        monkeypatch.setattr(NeuronPlatform, name, getattr(NeuronPlatform, name))
    if gate_lifted:
        monkeypatch.setattr(
            NeuronPlatform,
            "_text_only_architectures",
            NeuronPlatform._text_only_architectures - {ARCH},
        )
    vllm_config = VllmConfig(
        model_config=model_config,
        # Sequence cap: the platform refuses more sequences than the KDA kernels take.
        scheduler_config=SchedulerConfig.default_factory(max_num_seqs=16),
        additional_config={
            "encoder_neuron_config": {
                "encoder_attention_block_size": SERVED_BLOCK,
                "num_encoder_input_buckets": REGISTERED_BUCKETS,
            }
        },
    )
    processor = InputProcessor(vllm_config)
    placeholder = model_config.hf_config.image_placeholder
    # 2,800 px a side is 200 x 200 patches: 40,000 patches, 10,000 image tokens.
    request = {
        "prompt": f"{placeholder}Describe this image.",
        "multi_modal_data": {"image": [_image(2800)]},
    }
    refused, admitted = [], 0
    try:
        processor.process_inputs(
            "oversize", request, SamplingParams(max_tokens=1), ("generate",)
        )
        admitted += 1
    except ValueError as error:
        refused.append(str(error))
    return refused, admitted


def test_image_over_largest_bucket_is_refused_by_the_text_only_gate(
    model_config, monkeypatch
):
    refused, admitted = _oversize_image_outcome(
        model_config, monkeypatch, gate_lifted=False
    )
    assert (len(refused), admitted) == (1, 0)
    assert f"serves {ARCH} as text only" in refused[0]
    assert "produces 10000 embedding tokens" not in refused[0]


def test_image_over_largest_bucket_is_refused_with_the_gate_lifted(
    model_config, monkeypatch
):
    refused, admitted = _oversize_image_outcome(
        model_config, monkeypatch, gate_lifted=True
    )
    assert (len(refused), admitted) == (1, 0)
    assert "produces 10000 embedding tokens" in refused[0]
    assert "the compiled encoder supports (8384)" in refused[0]
