# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 uint8 pixels, with the normalize folded into the patch embed, in CPU mode.

A spawned process stands for a worker: it registers the plugin's models, builds
the registered K3 processor from the parent's model config, and runs it beside
the checkpoint's own image processor. Items 1 and 2 read the encoder hook's
tower input at the served buckets, item 3 the processor, the dummy inputs and
the warmup, item 4 the processor's host time on one core, and item 5 the
processor the API process builds, with vLLM's general plugins and no worker
registration. Set
``KIMI_K3_CHECKPOINT`` as for the tower test. Run with ``VLLM_NEURON_CPU_MODE=1``
and ``NKI_SIMULATOR=1``.
"""

import copy
import io
import math
import multiprocessing
import os
import pickle
import statistics
import sys
import time
import traceback

import numpy as np
import pytest
import torch
import torch.nn as nn
from test_vision_tower import (  # noqa: F401  (pytestmark is used by pytest)
    CHECKPOINT,
    build_model,
    encoder_config,
    hf_config,
    new_cache,
    pytestmark,
    read,
)

SERVED_BUCKETS = [5504, 33536]
SERVED_BLOCK = 5504
PATCH_DIM = 588
#: The host cost of the uint8 processor's call on one 1,024 px JPEG, and the pass factor on it.
HOST_ROW_MS = 5.455
HOST_FACTOR = 1.25


def _images() -> dict:
    """Item 1's three images first, then four more for the 7-image prompt, the last past the first bucket."""
    from PIL import Image

    rng = np.random.default_rng(7)
    chart = np.zeros((330, 460, 3), dtype=np.uint8)
    colours = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, colour in enumerate(colours):
        chart[:, i * 460 // 6 : (i + 1) * 460 // 6] = colour
    rgba = np.zeros((200, 300, 4), dtype=np.uint8)
    rgba[..., :3] = (40, 90, 200)
    # Transparent, half transparent and opaque thirds.
    rgba[:, 100:200, 3] = 128
    rgba[:, 200:, 3] = 255
    images = {
        "field": Image.fromarray(rng.integers(0, 256, (690, 1010, 3), dtype=np.uint8)),
        "chart": Image.fromarray(chart),
        "rgba": Image.fromarray(rgba),
    }
    for name, (h, w) in {"small": (224, 224), "wide": (360, 640), "served": (1036, 1036),
                         "large": (1092, 1092)}.items():
        images[name] = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    return images


def _jpeg() -> bytes:
    """A 1,024 px JPEG as the serving benchmark makes its images: a seeded field, bicubic up, noise, q90 4:2:0."""
    from PIL import Image

    rng = np.random.default_rng([20261024, 9000])
    field = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    up = np.asarray(Image.fromarray(field).resize((1024, 1024), Image.BICUBIC)).astype(np.int16)
    pixels = np.clip(up + rng.integers(-8, 9, (1024, 1024, 3)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=90, subsampling=2)
    return buffer.getvalue()


def _worker_processor(config_bytes: bytes):
    """The registered K3 processor as a worker builds it."""
    from vllm import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY

    from vllm_neuron.model import registry

    for arch, model_cls in registry.get_models():
        ModelRegistry.register_model(arch, model_cls)
    return MULTIMODAL_REGISTRY.create_processor(pickle.loads(config_bytes))


def _reference(info, image) -> dict:
    """The checkpoint processor's pixels for one image, and the same pixels before its normalize, uint8."""
    checkpoint = info.image_processor
    media = sys.modules[type(checkpoint).__module__.rsplit(".", 1)[0] + ".media_utils"]
    config = checkpoint.media_proc_cfg
    out = checkpoint.preprocess([{"type": "image", "image": image.copy()}])
    pixels = out["pixel_values"]
    raw = ((pixels.double() * 0.5 + 0.5) * 255).round().to(torch.uint8)
    # The inverse holds only if the checkpoint's normalize gives its pixels back bit for bit.
    again = media.normalize(raw.numpy().transpose(0, 2, 3, 1), np.array(config["image_mean"]),
                            1.0 / np.array(config["image_std"])).transpose(0, 3, 1, 2)
    return {"pixel_values": pixels, "grid_thws": out["grid_thws"].reshape(-1), "raw": raw,
            "inverse_exact": bool(np.array_equal(again, pixels.numpy()))}


def _put(queue, job) -> None:
    """Put ``job()``'s result on the queue as bytes, or the traceback."""
    try:
        buffer = io.BytesIO()
        torch.save(job(), buffer)
        queue.put(buffer.getvalue())
    except Exception:
        queue.put({"error": traceback.format_exc()})


def _process(config_bytes: bytes, queue) -> None:
    """Run the registered processor and the checkpoint's on the images, and the dummy inputs."""

    def job():
        from vllm.config.multimodal import ImageDummyOptions
        from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
        from vllm.multimodal.processing.context import TimingContext
        from vllm.multimodal.processing.inputs import ProcessorInputs

        processor = _worker_processor(config_bytes)
        info = processor.info
        placeholder = info.get_hf_config().image_placeholder

        def apply(inputs):
            result = processor.apply(inputs, TimingContext())
            return [{k: item[k].data.clone() for k in ("pixel_values", "grid_thws")}
                    for item in result["mm_kwargs"]["image"]]

        def plugin(images, parse=True):
            ids = info.get_tokenizer().encode(placeholder * len(images) + "Describe.",
                                              add_special_tokens=False)
            images = [i.copy() for i in images]
            # vLLM's parser turns a transparent image RGB on white, so the processor's own
            # fill reads an image given past the parser.
            items = (info.parse_mm_data({"image": images}) if parse
                     else MultiModalDataItems({"image": ImageProcessorItems(images)}))
            return apply(ProcessorInputs(list(ids), items))

        images = list(_images().values())
        dummy = []
        for side in (1036, 1092):
            options = {"image": ImageDummyOptions(width=side, height=side)}
            dummy += apply(processor.dummy_inputs.get_dummy_processor_inputs(8192, {"image": 1}, options))
        return {
            "class": type(processor).__qualname__,
            "items": [(plugin([image], parse=False)[0], _reference(info, image)) for image in images[:3]],
            "prompt": plugin(images),
            "prompt_reference": [_reference(info, image) for image in images],
            "dummy": dummy,
        }

    _put(queue, job)


def _children() -> list[str]:
    """This process's live child processes."""
    return [c for t in os.listdir("/proc/self/task")
            for c in open(f"/proc/self/task/{t}/children").read().split()]


def _time(config_bytes: bytes, jpeg: bytes, queue) -> None:
    """Time the registered processor's call on the decoded JPEG on one core: 1 cold call, then 5."""

    def job():
        from PIL import Image

        processor = _worker_processor(config_bytes)
        # Upstream's model import starts one-time build helpers; they would share the timed core.
        deadline = time.monotonic() + 600
        while _children() and time.monotonic() < deadline:
            time.sleep(0.5)
        helpers = _children()
        core = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {core})
        torch.set_num_threads(1)
        image = Image.open(io.BytesIO(jpeg))
        image.load()
        items = processor.info.parse_mm_data({"image": [image]})
        processor._apply_hf_processor_main(items, {})
        times = []
        for _ in range(5):
            start = time.perf_counter()
            out = processor._apply_hf_processor_main(items, {})
            times.append((time.perf_counter() - start) * 1e3)
        return {"core": core, "helpers left": len(helpers), "times_ms": times,
                "pixels": (out["pixel_values"].dtype, tuple(out["pixel_values"].shape))}

    _put(queue, job)


def _api_process(queue) -> None:
    """Build the registered processor as the API process does and run it on item 1's RGB field."""

    def job():
        from vllm.config import ModelConfig
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.processing.context import TimingContext
        from vllm.multimodal.processing.inputs import ProcessorInputs
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        model_config = ModelConfig(model=str(CHECKPOINT), tokenizer=str(CHECKPOINT), trust_remote_code=True,
                                   max_model_len=131072, limit_mm_per_prompt={"image": 7})
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        info = processor.info
        ids = info.get_tokenizer().encode(info.get_hf_config().image_placeholder + "Describe.",
                                          add_special_tokens=False)
        image = _images()["field"]
        result = processor.apply(ProcessorInputs(list(ids), info.parse_mm_data({"image": [image]})), TimingContext())
        pixels = result["mm_kwargs"]["image"][0]["pixel_values"].data
        raw = _reference(info, image)["raw"]
        return {"class": f"{type(processor).__module__}.{type(processor).__qualname__}",
                "dtype": pixels.dtype, "shape is the checkpoint's": pixels.shape == raw.shape,
                "differing values": int((pixels.double() != raw.double()).sum()) if pixels.shape == raw.shape
                else None}

    _put(queue, job)


def _spawn(target, *args):
    """Run ``target(*args, queue)`` in a spawned process; return its result."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=target, args=(*args, queue))
    child.start()
    out = queue.get(timeout=900)
    child.join()
    assert not isinstance(out, dict), out.get("error")
    return torch.load(io.BytesIO(out), weights_only=False)


@pytest.fixture(scope="module")
def model_config() -> bytes:
    """The model config the API process builds, pickled as the engine sends it to a worker."""
    from vllm.config import ModelConfig

    return pickle.dumps(ModelConfig(model=str(CHECKPOINT), tokenizer=str(CHECKPOINT), trust_remote_code=True,
                                    max_model_len=131072, limit_mm_per_prompt={"image": 7}))


@pytest.fixture(scope="module")
def processed(model_config) -> dict:
    return _spawn(_process, model_config)


class _Recorder(nn.Module):
    """Stands in for the compiled tower: keeps each call's tensor inputs."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({k: v.clone() for k, v in kwargs.items()
                           if isinstance(v, torch.Tensor) and k != "encoder_cache_buffer"})
        return kwargs["encoder_cache_buffer"]


@pytest.fixture(scope="module")
def tower_inputs(processed):
    """Each item image's tower call at the served buckets on one rank, and the loaded patch embed in fp64."""
    model = build_model(hf_config(), encoder_config(1, 1, SERVED_BUCKETS, SERVED_BLOCK))
    embed = copy.deepcopy(model.visual.vision_tower.patch_embed).double()
    recorder = _Recorder()
    model.visual = recorder
    cache = new_cache(2, SERVED_BLOCK // 4)
    calls = []
    for plugin, _ in processed["items"]:
        recorder.calls.clear()
        model.embed_multimodal(pixel_values=plugin["pixel_values"], grid_thws=plugin["grid_thws"].reshape(1, 3),
                               encoder_cache=cache, mm_hashes=["image"])
        cache.free("image")
        calls += recorder.calls
    return calls, embed


def _window(h: int, w: int) -> torch.Tensor:
    """Raster index of each patch of an h x w grid in 2 x 2 window order."""
    return torch.arange(h * w).view(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).reshape(-1)


def test_tower_input_is_the_checkpoint_pixels_before_normalize(processed, tower_inputs):
    """Item 1: the tower's input equals the checkpoint's pixels before normalize, window order, zero pad."""
    calls, _ = tower_inputs
    found = []
    for (_, ref), call in zip(processed["items"], calls):
        _, h, w = ref["grid_thws"].tolist()
        expected = torch.zeros(SERVED_BUCKETS[0], PATCH_DIM, dtype=torch.uint8)
        expected[: h * w] = ref["raw"].reshape(h * w, -1)[_window(h, w)]
        got = call["pixel_values"]
        found.append((ref["inverse_exact"], got.dtype, tuple(got.shape), int((got != expected).sum())))
        print(f"grid {h} x {w}, {SERVED_BUCKETS[0] - h * w} pad rows: inverse exact, dtype, shape, "
              f"differing values {found[-1]}")
    assert found == [(True, torch.uint8, (SERVED_BUCKETS[0], PATCH_DIM), 0)] * 3


def test_folded_patch_embed_matches_the_original_in_fp64(processed, tower_inputs):
    """Item 2: the folded patch embed on uint8 rows matches the original on normalized rows, fp64, within 1e-6."""
    calls, embed = tower_inputs
    keys = ("vision_tower.patch_embed.proj.weight", "vision_tower.patch_embed.pos_emb.weight")
    stored = read(keys)
    weight = stored[keys[0]].double().reshape(stored[keys[0]].shape[0], -1).T
    table = stored[keys[1]].double().reshape(-1, weight.shape[1])
    errors = []
    for (_, ref), call in zip(processed["items"], calls):
        _, h, w = ref["grid_thws"].tolist()
        n = h * w
        folded = embed(call["pixel_values"], call["pos_index"], call["pos_weight"])[:n]
        normalized = ref["pixel_values"].double().reshape(n, -1)[_window(h, w)]
        corners = table[call["pos_index"][:, :n].long()]
        original = normalized @ weight + (corners * call["pos_weight"][:, :n].double().unsqueeze(-1)).sum(0)
        # The largest difference over the original's largest value, on the image's real rows.
        errors.append(((folded - original).abs().max() / original.abs().max()).item())
        print(f"grid {h} x {w}: {folded.dtype}, max relative error {errors[-1]:.3e}")
    assert max(errors) <= 1e-6, errors


def test_every_pixel_tensor_is_uint8(processed):
    """Item 3: a 7-image prompt with one past the first bucket, the dummy inputs and warmup: 0 non-uint8 pixels."""
    prompt, refs = processed["prompt"], processed["prompt_reference"]
    shapes = [tuple(p["pixel_values"].shape) == tuple(r["pixel_values"].shape)
              and p["grid_thws"].tolist() == r["grid_thws"].tolist() for p, r in zip(prompt, refs)]

    def bucket(item):
        return min(b for b in SERVED_BUCKETS if b >= item["grid_thws"].prod())

    served = encoder_config(4, 8, SERVED_BUCKETS, SERVED_BLOCK)
    model = build_model(hf_config(27), served, load=False)
    recorder = _Recorder()
    model.visual = recorder
    cache = new_cache(16, SERVED_BLOCK // 4)
    synthetic = []
    for rows in SERVED_BUCKETS:
        inputs = model.build_encoder_synthetic_inputs(rows, served, torch.device("cpu"))
        synthetic.append(inputs["pixel_values"])
        # The worker adds the cache and the write ids, one per encoder block, padded to DP.
        blocks = math.ceil(math.ceil(rows / SERVED_BLOCK) / served.dp_size) * served.dp_size
        inputs["encoder_cache_buffer"] = cache.buffer
        inputs["write_block_ids"] = torch.zeros(blocks, dtype=torch.int64)
        model.get_encoder()(**inputs)
    warmed = [call["pixel_values"] for call in recorder.calls]
    pixels = [p["pixel_values"] for p in prompt + processed["dummy"]] + synthetic + warmed
    report = {
        "processor": processed["class"],
        "prompt images, shapes match": (len(prompt), shapes),
        "prompt buckets": [bucket(p) for p in prompt],
        "dummy buckets": [bucket(d) for d in processed["dummy"]],
        "synthetic rows": [t.shape[0] for t in synthetic],
        "warmup rows": sorted({t.shape[0] for t in warmed}),
        "non-uint8 pixel tensors": sum(t.dtype != torch.uint8 for t in pixels),
    }
    for key, value in report.items():
        print(f"{key}: {value}")
    assert report["prompt images, shapes match"] == (7, [True] * 7)
    assert report["prompt buckets"].count(SERVED_BUCKETS[1]) == 1
    assert report["dummy buckets"] == report["synthetic rows"] == SERVED_BUCKETS
    assert set(SERVED_BUCKETS) <= set(report["warmup rows"])
    assert report["non-uint8 pixel tensors"] == 0


def test_processor_host_time_on_one_core(model_config):
    """Item 4: the processor's median of 5 calls on the decoded 1,024 px JPEG is within 1.25 x its host row."""
    out = _spawn(_time, model_config, _jpeg())
    median = statistics.median(out["times_ms"])
    limit = HOST_FACTOR * HOST_ROW_MS
    calls = [round(t, 3) for t in out["times_ms"]]
    print(f"core {out['core']}, build helpers left {out['helpers left']}; calls {calls} ms; "
          f"median {median:.3f} ms; limit {limit:.3f} ms; pixels {out['pixels']}")
    assert median <= limit


def test_api_process_builds_the_uint8_processor():
    """Item 5: with no worker registration, the API process gets the plugin's processor and its uint8 pixels."""
    out = _spawn(_api_process)
    print(f"API process: {out}")
    assert out == {"class": "vllm_neuron.model.kimi_k3.vision_processor.KimiK3Uint8Processor",
                   "dtype": torch.uint8, "shape is the checkpoint's": True, "differing values": 0}
