# SPDX-License-Identifier: Apache-2.0
"""The glm-5.3-Flash implementation for a server that admits images.

:class:`Glm5NextImageForConditionalGeneration` is the text root of ``model_fp8.py`` with the vision
tower added, and the two members the runner and the worker call on an image-serving model:
``embed_multimodal``, which encodes a batch of images into the encoder cache, and
``build_vision_synthetic_inputs``, which gives the tower's warm-up inputs for one bucket.

It is a separate class, not a branch inside the text root, because the worker decides whether to
warm up and compile a vision graph by whether the model has ``build_vision_synthetic_inputs``
(``SupportsVisionWarmup``, ``NeuronWorker._unwrap_vision_model``). So a text-only server must build
a class without that method. The factory selects the class (``factory.py``, ``_serves_images``).

Layout: one image per vision attention block, at the block's first rows, and one encoder-cache
block per image. A cache block holds the merged rows of one attention block
(``vision_attention_block_size // spatial_merge_size ** 2``, ``NeuronModelRunner._init_encoder_cache``),
so the tower writes each attention block's output into one cache block.
"""

from __future__ import annotations

import torch

from vllm_neuron.model.neuron_config import VisionNeuronConfig

from .config import Glm5NextConfig
from .model_fp8 import Glm5NextForConditionalGeneration
from .vision_encoder import (
    Glm5NextVisionEncoder,
    Glm5NextVisionEncoderError,
    pack_images_into_blocks,
)

#: The tower's dtype. The checkpoint stores every tower tensor in bf16, and the encoder cache
#: the tower writes into is bf16 (``NeuronModelRunner._init_encoder_cache``).
VISION_DTYPE = torch.bfloat16

#: The columns of ``image_grid_thw``: the temporal, height and width patch counts of an image.
_GRID_THW_COLUMNS = 3


class Glm5NextImageForConditionalGeneration(Glm5NextForConditionalGeneration):
    """glm-5.3-Flash with its vision tower: the implementation of a server that admits images."""

    def __init__(self, config: Glm5NextConfig) -> None:
        """Build the text root, then the vision tower in :data:`VISION_DTYPE`.

        Args:
            config: the model config, with the resolved vision neuron config.

        Raises:
            Glm5NextVisionEncoderError: no vision neuron config; a vision ``tp_size`` other
                than 1; or a vision bucket above the attention block size. The checks read
                the config only, so they run before the decoder stack is built.
        """
        vision_neuron_config = config.vision_config.neuron_config
        if vision_neuron_config is None:
            raise Glm5NextVisionEncoderError(
                "the image root needs the vision neuron config for its attention block size "
                "and its buckets; the factory selects this class only when one is given"
            )
        if vision_neuron_config.tp_size != 1:
            raise Glm5NextVisionEncoderError(
                f"vision tp_size={vision_neuron_config.tp_size}: the vision tower has no "
                f"tensor-parallel split and every rank holds it whole, so vision ranks split "
                f"the blocks by data parallelism only; set the vision tp_size to 1"
            )
        buckets = vision_neuron_config.num_vision_tokens_buckets
        block_size = vision_neuron_config.vision_attention_block_size
        if buckets and max(buckets) > block_size:
            raise Glm5NextVisionEncoderError(
                f"the largest vision bucket {max(buckets)} is above the attention block size "
                f"{block_size}. The per-image limit is the largest bucket over the merge "
                f"factor, so an admitted image could need more than one block, and this "
                f"model places one image in one block"
            )
        super().__init__(config)
        self.visual = Glm5NextVisionEncoder(config.vision_config, dtype=VISION_DTYPE)

    def embed_multimodal(
        self,
        *,
        encoder_cache,
        mm_hashes: list[str],
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **other_fields: object,
    ) -> None:
        """Encode a batch of images and write their merged rows into the encoder cache.

        The runner calls this once per modality batch (``NeuronModelRunner._execute_mm_encoder``)
        and marks each hash written after the call. Each image gets one cache block, and the
        bucket is the smallest whose attention blocks hold every image.

        Args:
            encoder_cache: the runner's ``EncoderCacheBlocks``.
            mm_hashes: one per image, in batch order.
            pixel_values: ``[total_patches, patch_dim]``, every image's patch rows in turn.
            image_grid_thw: ``[num_images, 3]`` (temporal, height, width) patch counts.
            other_fields: the fields of any other modality; refused.

        Raises:
            Glm5NextVisionEncoderError: a field of another modality, a hash count that differs
                from the image count, a cache whose block is not one attention block's merged
                rows, more images than the largest bucket holds, or an image the layout
                refuses (:func:`~.vision_encoder.pack_images_into_blocks`). No cache block is
                allocated before every check passes.
        """
        present = sorted(
            name for name, value in other_fields.items() if value is not None
        )
        if present:
            raise Glm5NextVisionEncoderError(
                f"this model encodes images only, and the batch carries {present}"
            )
        num_images = int(image_grid_thw.shape[0])
        if len(mm_hashes) != num_images:
            raise Glm5NextVisionEncoderError(
                f"{len(mm_hashes)} hashes for {num_images} images"
            )
        vnc = self.vision_config.neuron_config
        block_size = vnc.vision_attention_block_size
        merge_factor = self.vision_config.spatial_merge_size**2
        if encoder_cache.block_size * merge_factor != block_size:
            raise Glm5NextVisionEncoderError(
                f"the encoder cache block holds {encoder_cache.block_size} rows, and one "
                f"attention block of {block_size} rows merges to {block_size // merge_factor}"
            )
        fitting = [
            blocks
            for blocks in vnc.num_total_vision_attention_blocks
            if blocks >= num_images
        ]
        if not fitting:
            raise Glm5NextVisionEncoderError(
                f"{num_images} images need {num_images} attention blocks, and the largest "
                f"bucket has {max(vnc.num_total_vision_attention_blocks)}"
            )
        num_blocks = min(fitting)
        packed = pack_images_into_blocks(
            pixel_values,
            image_grid_thw,
            block_size=block_size,
            num_blocks=num_blocks,
            config=self.vision_config,
            dtype=VISION_DTYPE,
        )

        write_block_ids = []
        for mm_hash, (t, h, w) in zip(mm_hashes, image_grid_thw.tolist()):
            write_block_ids += encoder_cache.allocate(
                mm_hash, [int(t * h * w) // merge_factor]
            )
        # The blocks past the last image are padding, written to the cache's scratch
        # block, which is never allocated.
        write_block_ids += [encoder_cache.scratch_block_id] * (
            num_blocks - len(write_block_ids)
        )
        device = encoder_cache.buffer.device
        self.visual(
            **{name: tensor.to(device) for name, tensor in packed.items()},
            encoder_cache_buffer=encoder_cache.buffer,
            write_block_ids=torch.tensor(
                write_block_ids, dtype=torch.int64, device=device
            ),
        )

    def build_vision_synthetic_inputs(
        self,
        bucket: int,
        vision_neuron_config: VisionNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """The tower's inputs for one bucket, for the worker's warm-up and graph capture.

        Every block is a padding block, so the shapes are the bucket's and the values are the
        ones padding carries. The worker adds ``encoder_cache_buffer`` and ``write_block_ids``.

        Args:
            bucket: one entry of ``num_vision_tokens_buckets``.
            vision_neuron_config: the resolved vision config, for the bucket's block count.
            device: where the tensors go; ``meta`` for shape-only capture.

        Returns:
            ``pixel_values``, ``cos``, ``sin``, ``bound_min`` and ``bound_max`` of
            :meth:`~.vision_encoder.Glm5NextVisionEncoder.forward`.
        """
        index = vision_neuron_config.num_vision_tokens_buckets.index(bucket)
        packed = pack_images_into_blocks(
            torch.empty(0, self.vision_config.patch_dim),
            torch.empty(0, _GRID_THW_COLUMNS, dtype=torch.int64),
            block_size=vision_neuron_config.vision_attention_block_size,
            num_blocks=vision_neuron_config.num_total_vision_attention_blocks[index],
            config=self.vision_config,
            dtype=VISION_DTYPE,
        )
        return {name: tensor.to(device) for name, tensor in packed.items()}
