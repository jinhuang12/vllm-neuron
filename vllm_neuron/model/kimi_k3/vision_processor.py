# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 image processor that gives uint8 patches.

It keeps the checkpoint processor's resize, transparent fill and pad, and skips
its fp32 normalize. One pass writes each padded image as patches in the
checkpoint's order, so the encoder takes the raw pixel values and its patch
embed applies the normalize. The encoder orders the patches into windows and
pads them to its bucket on each worker.
"""

import sys

import numpy as np
import torch
from transformers import BatchFeature
from vllm.models.kimi_k3.common.mm_preprocess import KimiK3MultiModalProcessor
from vllm.multimodal.parse import ImageProcessorItems

#: The checkpoint's mean and std in each channel, which the patch embed's fold assumes.
FOLDED_MEAN = FOLDED_STD = (0.5, 0.5, 0.5)


def padded_pixels(image_processor, image) -> np.ndarray:
    """One image resized, filled and padded as the checkpoint does it, uint8 [H, W, 3]."""
    # The checkpoint's media step, the function its own preprocess calls.
    ensure_media_type = sys.modules[type(image_processor).__module__].ensure_media_type
    media = ensure_media_type(
        {"type": "image", "image": image},
        transparent_bg_config=image_processor._transparent_bg_config,
        transparent_bg_fill_stage=image_processor._transparent_bg_fill_stage,
    )
    size = image_processor.get_resize_config(media)
    return image_processor.resize_image(
        media["image"],
        size["new_width"],
        size["new_height"],
        size["pad_width"],
        size["pad_height"],
    )


class KimiK3Uint8Processor(KimiK3MultiModalProcessor):
    """Upstream's K3 multimodal processor with uint8 patches and no normalize."""

    def __init__(self, info, dummy_inputs, *, cache=None):
        super().__init__(info, dummy_inputs, cache=cache)
        config = info.image_processor.media_proc_cfg
        stats = (tuple(config["image_mean"]), tuple(config["image_std"]))
        if stats != (FOLDED_MEAN, FOLDED_STD):
            raise ValueError(
                f"the patch embed folds mean {FOLDED_MEAN} and std {FOLDED_STD}; "
                f"the checkpoint normalizes with mean {stats[0]} and std {stats[1]}"
            )

    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs) -> BatchFeature:
        """Give the images' uint8 patches [patches, 3, p, p] and grids; other inputs take upstream's path."""
        images = mm_items.get("image")
        if not isinstance(images, ImageProcessorItems) or not len(images):
            return super()._apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)
        image_processor = self.info.image_processor
        p = image_processor.media_proc_cfg["patch_size"]
        padded = [padded_pixels(image_processor, images.get(i)) for i in range(len(images))]
        grids = [(1, a.shape[0] // p, a.shape[1] // p) for a in padded]
        patches = np.empty((sum(h * w for _, h, w in grids), 3, p, p), dtype=np.uint8)
        start = 0
        for pixels, (_, h, w) in zip(padded, grids):
            # The one pass: patches in raster order, each channel-major as the patch weight reads it.
            rows = patches[start : start + h * w].reshape(h, w, 3, p, p)
            rows[...] = pixels.reshape(h, p, w, p, 3).transpose(0, 2, 4, 1, 3)
            start += h * w
        return BatchFeature(
            {
                "pixel_values": torch.from_numpy(patches),
                "grid_thws": torch.tensor(grids, dtype=torch.int64),
            }
        )
