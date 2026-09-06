# SPDX-License-Identifier: Apache-2.0
"""Vision preprocessing bridge for GLM-5.3-Flash (``glm5_next``).

WHAT THIS FILE IS FOR. vLLM 0.24.0 has no ``glm5_next`` multimodal processor, so an image handed to this
architecture never reaches the vision tower. Transformers 5.16.1 already has the patch-grid arithmetic and
this file does not re-derive it: a second implementation of the same rule is a place for the two to drift
apart. What is missing, and what this file supplies, is the vLLM-side bridge -- the processing-info,
dummy-inputs and multimodal-processor triple that vLLM asks a model for -- plus a closed form for the grid
that a test can compare against the transformers path.

THE GRID, IN WORDS. The processor pads an image out to a canvas whose sides are multiples of
``patch_size * merge_size`` (28 for this checkpoint), then cuts the canvas into 14-pixel patches and merges
them 2x2 for the language model. Three cases arise, and every one of them is named below as a regime:

  A  the aligned canvas already fits the token budget, so the content is padded with zeros, never resampled;
  B  the canvas holds fewer than ``min_image_tokens`` tokens, so the content is scaled UP first;
  C  the canvas holds more than ``max_image_tokens``, so a bounded search cuts it down.

Two facts are easy to get wrong and are therefore written down rather than left to the reader. First, the
image path tells the resizer there are ``temporal_patch_size`` frames, not one, so the floor in regime B
lands at 112 pixels for a 29x29 input and not at the 168 a one-frame reading gives. Second, the returned
grid's temporal element is a literal 1 for an image -- the temporal patch is folded into the patch vector,
not into the grid.

WHAT THIS FILE DOES NOT DO. It declares no protocol on the model class and it does not enable video at the
vLLM boundary: the video placeholder is a per-frame timestamp structure needing video metadata that this
increment does not own. The video grid arithmetic and field mapping ARE here, because they are the same
arithmetic and the later WP10 blocks build on them.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from transformers import BatchFeature

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

#: The transformers release that carries ``transformers.models.glm5_next``. Named here only so the refusal
#: below can say what to install; ``requirements/core.txt`` is the single place the floor is enforced.
GLM5_NEXT_TRANSFORMERS_FLOOR = "5.16.1"

#: The three regimes, spelled once. A caller comparing against these gets a name, not a magic string.
REGIME_PLAIN = "A"
REGIME_BELOW_FLOOR = "B"
REGIME_ABOVE_CEILING = "C"

#: The six attributes the grid arithmetic reads off an image or video processor. Listed so a missing one
#: produces a message naming it instead of an ``AttributeError`` from the middle of a computation.
_REQUIRED_PROCESSOR_ATTRS = (
    "patch_size",
    "merge_size",
    "temporal_patch_size",
    "patch_expand_factor",
    "min_image_tokens",
    "max_image_tokens",
)


def require_transformers_glm5_next():
    """Import the transformers ``glm5_next`` package, refusing by name when it is absent.

    ``requirements/core.txt`` pins the floor, but a fork can be run against whatever is installed. An
    absent model package then surfaces as a bare ``ModuleNotFoundError`` from deep inside an auto-class
    lookup, which tells the reader nothing. This says what is missing and what to install.
    """
    try:
        import transformers.models.glm5_next as glm5_next_pkg
    except ImportError as exc:  # pragma: no cover - exercised only on an under-floor install
        import transformers

        raise RuntimeError(
            "GLM-5.3-Flash vision preprocessing needs transformers.models.glm5_next, which the installed "
            f"transformers {transformers.__version__} does not provide. Install "
            f"transformers>={GLM5_NEXT_TRANSFORMERS_FLOOR},<6.0.0 (the floor requirements/core.txt names)."
        ) from exc

    return glm5_next_pkg


@dataclass(frozen=True)
class GridConstants:
    """The six numbers the patch grid depends on, read off a processor rather than typed.

    Every derived quantity is a property so that a checkpoint with a different patch or merge size gets the
    right answer without a second place to edit.
    """

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    patch_expand_factor: int
    min_image_tokens: int
    max_image_tokens: int

    @classmethod
    def from_processor(cls, processor) -> GridConstants:
        """Read the six values off an image or video processor instance."""
        missing = [name for name in _REQUIRED_PROCESSOR_ATTRS if not hasattr(processor, name)]
        if missing:
            raise AttributeError(
                f"{type(processor).__name__} does not carry {missing}, which the GLM-5.3-Flash patch grid "
                "needs. This bridge reads the grid constants off the processor and never assumes them."
            )
        return cls(*(getattr(processor, name) for name in _REQUIRED_PROCESSOR_ATTRS))

    @property
    def factor(self) -> int:
        """The canvas alignment, 28 for this checkpoint."""
        return self.patch_size * self.merge_size * self.patch_expand_factor

    @property
    def pixels_per_token(self) -> int:
        """Pixels one merged token stands for, 1568 for this checkpoint."""
        return self.temporal_patch_size * self.factor**2

    @property
    def floor_pixels(self) -> int:
        """Below this the content is scaled up (regime B)."""
        return self.min_image_tokens * self.pixels_per_token

    @property
    def ceiling_pixels(self) -> int:
        """Above this the canvas is cut down (regime C)."""
        return self.max_image_tokens * self.pixels_per_token

    @property
    def merge_length(self) -> int:
        """Patch rows per merged token, 4 for this checkpoint."""
        return self.merge_size**2


@dataclass(frozen=True)
class VisionGridSpec:
    """One item's patch grid, with the regime that produced it."""

    grid_t: int
    grid_h: int
    grid_w: int
    regime: str
    canvas_height: int
    canvas_width: int
    merge_length: int

    @property
    def thw(self) -> tuple[int, int, int]:
        """The triple the transformers path returns as ``image_grid_thw`` / ``video_grid_thw``."""
        return (self.grid_t, self.grid_h, self.grid_w)

    @property
    def num_patch_rows(self) -> int:
        """Rows in the flat pixel tensor for this item."""
        return self.grid_t * self.grid_h * self.grid_w

    @property
    def num_merged_tokens(self) -> int:
        """Placeholder tokens the language model sees for this item."""
        return self.num_patch_rows // self.merge_length


@dataclass(frozen=True)
class ResampleRecord:
    """What happened to one item's pixels, so that no resample is silent.

    ``resampled`` is the processor's own condition -- the content size differing from the input size is
    exactly what guards its resize call -- rather than a second rule that could disagree with it.
    """

    modality: str
    num_frames: int
    height: int
    width: int
    canvas_height: int
    canvas_width: int
    content_height: int
    content_width: int
    regime: str

    @property
    def resampled(self) -> bool:
        """True when the pixels were scaled, either up (regime B) or down (regime C)."""
        return (self.content_height, self.content_width) != (self.height, self.width)

    @property
    def pad_bottom(self) -> int:
        """Zero rows added below the content."""
        return self.canvas_height - self.content_height

    @property
    def pad_right(self) -> int:
        """Zero columns added right of the content."""
        return self.canvas_width - self.content_width

    def describe(self) -> str:
        """One plain line a person can read in a log."""
        action = "resampled" if self.resampled else "padded only"
        return (
            f"{self.modality} item {self.height}x{self.width}"
            f"{'' if self.num_frames == 1 else f' x{self.num_frames} frames'} "
            f"-> content {self.content_height}x{self.content_width} on a "
            f"{self.canvas_height}x{self.canvas_width} canvas, regime {self.regime}, {action} "
            f"(pad {self.pad_bottom} bottom, {self.pad_right} right)"
        )


def _align(value: int, factor: int) -> int:
    """Round up to the next multiple of ``factor``."""
    return math.ceil(value / factor) * factor


def _largest_canvas_within_budget(
    aligned_frames: int, height: int, width: int, factor: int, ceiling_pixels: int
) -> tuple[int, int]:
    """The regime C search: the largest aligned canvas whose budget fits under the ceiling.

    This walks the same bounded search the transformers path walks, on the content height, keeping the
    aspect ratio. It is a search rather than one division because the alignment makes the budget a step
    function of the content height, so the largest admissible height cannot be divided out directly.
    """
    low, high = 1, height
    best_height, best_width = factor, factor
    while low <= high:
        content_height = (low + high) // 2
        content_width = max(1, math.floor(width * content_height / height))
        candidate_height = _align(content_height, factor)
        candidate_width = _align(content_width, factor)
        if aligned_frames * candidate_height * candidate_width <= ceiling_pixels:
            best_height, best_width = candidate_height, candidate_width
            low = content_height + 1
        else:
            high = content_height - 1
    return best_height, best_width


def resolve_canvas(
    consts: GridConstants, *, num_frames: int, height: int, width: int, ceiling_pixels: int | None = None
) -> tuple[int, int, str]:
    """The canvas an input is padded or scaled onto, and which regime decided it.

    ``num_frames`` is what the caller hands the resizer: ``temporal_patch_size`` for a still image, the real
    frame count for a video. Getting that wrong moves regime B's answer, which is why it is a parameter here
    and not an assumption.
    """
    factor = consts.factor
    ceiling = consts.ceiling_pixels if ceiling_pixels is None else ceiling_pixels
    aligned_frames = max(
        consts.temporal_patch_size,
        round(num_frames / consts.temporal_patch_size) * consts.temporal_patch_size,
    )

    canvas_height, canvas_width = _align(height, factor), _align(width, factor)
    budget = aligned_frames * canvas_height * canvas_width
    regime = REGIME_PLAIN

    if budget < consts.floor_pixels:
        regime = REGIME_BELOW_FLOOR
        scale = math.sqrt(consts.floor_pixels / (num_frames * height * width))
        canvas_height = _align(max(1, math.ceil(height * scale)), factor)
        canvas_width = _align(max(1, math.ceil(width * scale)), factor)
        budget = aligned_frames * canvas_height * canvas_width

    if budget > ceiling:
        regime = REGIME_ABOVE_CEILING
        canvas_height, canvas_width = _largest_canvas_within_budget(
            aligned_frames, height, width, factor, ceiling
        )

    return canvas_height, canvas_width, regime


def image_grid_spec(consts: GridConstants, height: int, width: int) -> VisionGridSpec:
    """The patch grid for one still image.

    The temporal element is 1 by construction: the image path folds ``temporal_patch_size`` into the patch
    vector, so the grid stays two-dimensional even though the resizer was told there were two frames.
    """
    canvas_height, canvas_width, regime = resolve_canvas(
        consts, num_frames=consts.temporal_patch_size, height=height, width=width
    )
    return VisionGridSpec(
        grid_t=1,
        grid_h=canvas_height // consts.patch_size,
        grid_w=canvas_width // consts.patch_size,
        regime=regime,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        merge_length=consts.merge_length,
    )


def video_grid_spec(
    consts: GridConstants, num_frames: int, height: int, width: int
) -> VisionGridSpec:
    """The patch grid for one video, whose temporal element counts padded frame pairs."""
    canvas_height, canvas_width, regime = resolve_canvas(
        consts, num_frames=num_frames, height=height, width=width
    )
    padded_frames = num_frames + (-num_frames % consts.temporal_patch_size)
    return VisionGridSpec(
        grid_t=padded_frames // consts.temporal_patch_size,
        grid_h=canvas_height // consts.patch_size,
        grid_w=canvas_width // consts.patch_size,
        regime=regime,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        merge_length=consts.merge_length,
    )


def describe_resample(
    consts: GridConstants, *, modality: str, num_frames: int, height: int, width: int
) -> ResampleRecord:
    """One record saying whether this item's pixels were scaled, and onto what canvas.

    The scale, the clamp and the two content sides are written as the transformers path writes them, in the
    same order, so a floating-point boundary lands the same way in both. Only the guard is worth a comment:
    an input already big enough for the floor may never be scaled UP, so the scale is clamped at 1.0 and a
    regime A item is padded with zeros alone.
    """
    canvas_height, canvas_width, regime = resolve_canvas(
        consts, num_frames=num_frames, height=height, width=width
    )
    scale = min(canvas_height / height, canvas_width / width)
    if num_frames * height * width >= consts.floor_pixels:
        scale = min(1.0, scale)
    content_height = max(1, min(canvas_height, math.floor(height * scale)))
    content_width = max(1, min(canvas_width, math.floor(width * scale)))
    return ResampleRecord(
        modality=modality,
        num_frames=num_frames,
        height=height,
        width=width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        content_height=content_height,
        content_width=content_width,
        regime=regime,
    )


def build_mm_fields_config(
    hf_inputs: Mapping[str, torch.Tensor], spatial_merge_size: int
) -> Mapping[str, MultiModalFieldConfig]:
    """How vLLM slices a batch of processor outputs back into per-item pieces.

    The pixel tensors are flat and row-major, so each item's slice is its grid product; the embed tensors
    are one row per merged token, so each item's slice is that product divided by the merge area. The grids
    themselves are one row per item and stay on the CPU, which is where the placeholder arithmetic reads
    them.

    Declared here rather than taken from vLLM's shared qwen2-vl factory: that helper is private, and this
    fork now pins the transformers floor it depends on, so a private import would be a place for the two
    versions to drift apart silently. ``timestamps`` is absent because this checkpoint's video processor
    does not return one.
    """
    empty_grid = torch.empty((0, 3), dtype=torch.long)

    image_grid_thw = hf_inputs.get("image_grid_thw", empty_grid)
    image_patch_rows = image_grid_thw.prod(-1)
    image_embed_rows = image_patch_rows // spatial_merge_size // spatial_merge_size

    video_grid_thw = hf_inputs.get("video_grid_thw", empty_grid)
    video_patch_rows = video_grid_thw.prod(-1)
    video_embed_rows = video_patch_rows // spatial_merge_size // spatial_merge_size

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_patch_rows),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_embed_rows),
        image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_patch_rows),
        video_embeds=MultiModalFieldConfig.flat_from_sizes("video", video_embed_rows),
        video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
    )


class Glm5NextProcessingInfo(BaseProcessingInfo):
    """What vLLM asks about this checkpoint's multimodal inputs before it processes any."""

    def get_hf_processor(self, **kwargs: object):
        """Resolve the transformers processor, refusing by name when the model package is absent."""
        require_transformers_glm5_next()
        return super().get_hf_processor(**kwargs)

    def get_image_processor(self, **kwargs: object):
        """The sub-processor that owns the image patch grid."""
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object):
        """The sub-processor that owns the video patch grid, whose token ceiling differs."""
        return self.get_hf_processor(**kwargs).video_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """Any number of images; video is not offered at this boundary yet.

        Omitting video is the same shape 31 of vLLM 0.24.0's own image-only models use. The video grid
        arithmetic in this module is correct and tested, but the video PLACEHOLDER is a per-frame timestamp
        structure needing video metadata, and this increment does not own that plumbing. Offering video with
        a placeholder this bridge cannot build would drop frames silently, which is the failure mode this
        block exists to avoid.
        """
        return {"image": None}

    def get_grid_constants(self, **kwargs: object) -> GridConstants:
        """The image grid constants, read off the resolved image processor."""
        return GridConstants.from_processor(self.get_image_processor(**kwargs))

    def get_num_image_tokens(self, *, image_width: int, image_height: int, **kwargs: object) -> int:
        """Placeholder tokens one image of this size expands to."""
        consts = self.get_grid_constants(**kwargs)
        return image_grid_spec(consts, image_height, image_width).num_merged_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        """The largest square input that still fits under the token ceiling.

        Derived by stepping up in canvas-alignment units until the next step would exceed the budget, so it
        follows the checkpoint's own constants rather than a remembered number.
        """
        consts = self.get_grid_constants()
        aligned_frames = consts.temporal_patch_size
        side = consts.factor
        while aligned_frames * (side + consts.factor) ** 2 <= consts.ceiling_pixels:
            side += consts.factor
        return ImageSize(width=side, height=side)


class Glm5NextDummyInputsBuilder(BaseDummyInputsBuilder[Glm5NextProcessingInfo]):
    """The worst-case input vLLM profiles memory with."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """One image placeholder per profiled image."""
        num_images = mm_counts.get("image", 0)
        return self.info.get_hf_processor().image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Images at the largest size the token ceiling admits."""
        num_images = mm_counts.get("image", 0)
        target = self.info.get_image_size_with_most_features()
        overrides = (mm_options or {}).get("image")
        return {
            "image": self._get_dummy_images(
                width=target.width, height=target.height, num_images=num_images, overrides=overrides
            )
        }


class Glm5NextMultiModalProcessor(BaseMultiModalProcessor[Glm5NextProcessingInfo]):
    """The processor vLLM runs, and the field and placeholder rules it needs from this checkpoint."""

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the processor's flat outputs per item, using this checkpoint's merge size."""
        return build_mm_fields_config(hf_inputs, self.info.get_image_processor().merge_size)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each image placeholder to as many tokens as its own grid produced."""
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        merge_length = self.info.get_image_processor(**hf_processor_mm_kwargs).merge_size**2

        def get_image_replacement(item_idx: int) -> list[int]:
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement,
            )
        ]


def register_glm5_next_multimodal(model_cls: type) -> type:
    """Attach this bridge to the model class vLLM loads, and return that class.

    The class is passed in rather than imported here so that the package's ``__init__`` stays the single
    place the two halves meet and no import cycle is created. vLLM resolves this architecture through the
    lazy name ``vllm_neuron.model.glm5_next:Glm5NextForConditionalGeneration``, whose module part is the
    package, so importing the package is what the resolution itself does -- the registration below has
    always run by the time vLLM takes the class attribute.

    Nothing here is defensive on purpose. A registration that quietly failed would leave the server running
    with images dropped, because vLLM catches the unregistered-processor error and falls back to text.
    """
    return MULTIMODAL_REGISTRY.register_processor(
        Glm5NextMultiModalProcessor,
        info=Glm5NextProcessingInfo,
        dummy_inputs=Glm5NextDummyInputsBuilder,
    )(model_cls)


__all__ = [
    "GLM5_NEXT_TRANSFORMERS_FLOOR",
    "REGIME_ABOVE_CEILING",
    "REGIME_BELOW_FLOOR",
    "REGIME_PLAIN",
    "Glm5NextDummyInputsBuilder",
    "Glm5NextMultiModalProcessor",
    "Glm5NextProcessingInfo",
    "GridConstants",
    "ResampleRecord",
    "VisionGridSpec",
    "build_mm_fields_config",
    "describe_resample",
    "image_grid_spec",
    "register_glm5_next_multimodal",
    "require_transformers_glm5_next",
    "resolve_canvas",
    "video_grid_spec",
]
