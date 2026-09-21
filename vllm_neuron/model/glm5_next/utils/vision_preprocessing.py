# SPDX-License-Identifier: Apache-2.0
"""Vision preprocessing bridge for GLM-5.3-Flash (``glm5_next``).

vLLM 0.24.0 has no ``glm5_next`` multimodal processor, so an image handed to this architecture never reaches
the vision tower. This module supplies the missing vLLM-side bridge: the processing-info, dummy-inputs and
multimodal-processor triple vLLM asks a model for.

Every canvas is transformers' own answer -- :func:`resolve_canvas` calls ``smart_resize`` through the one
helper below that names its private submodule. Three things transformers does not offer are derived here
instead:

* The **regime label**, because ``smart_resize`` returns a canvas and never says which of its three branches
  produced it.
* The **video grid**, because the video processor carries no ``get_number_of_video_patches`` and the method
  that would call it raises first.
* The **resample record**, because the same arithmetic lives inside the image processor's ``resize``, which
  needs real pixels and hands back pixels rather than numbers.

The patch count is this module's own division over the consumed canvas rather than transformers'
``get_number_of_image_patches``, because that method drops ``patch_expand_factor`` where the real
preprocessing path applies it.

The grid, in words: the processor pads an image out to a canvas whose sides are multiples of
``patch_size * merge_size`` (28 for this checkpoint), cuts the canvas into 14-pixel patches, and merges them
2x2 for the language model. Three cases arise, each named below as a regime:

  A  the aligned canvas already fits the token budget, so the content is padded with zeros, never resampled;
  B  the canvas holds fewer than ``min_image_tokens`` tokens, so the content is scaled up first;
  C  the canvas holds more than ``max_image_tokens``, so a bounded search cuts it down.

Two details are easy to get wrong. The image path tells the resizer there are ``temporal_patch_size`` frames
rather than one, so regime B's floor lands at 112 pixels for a 29x29 input and not the 168 a one-frame reading
gives. And the returned grid's temporal element is a literal 1 for an image, because the temporal patch is
folded into the patch vector rather than into the grid.

Video is not offered at the vLLM boundary: its placeholder is a per-frame timestamp structure that needs video
metadata this bridge does not receive. The video grid arithmetic and field mapping are here, because they are
the same arithmetic.
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
#: below can say what to install; ``requirements/core.txt`` is where the floor is enforced.
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

    ``requirements/core.txt`` sets the floor, but the fork can be run against whatever is installed. An absent
    model package otherwise surfaces as a bare ``ModuleNotFoundError`` from inside an auto-class lookup, which
    tells the reader nothing. This names what is missing and what to install.
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


#: Where transformers keeps the canvas arithmetic, and what it is called there. Named once so the refusal
#: below and any caller checking for the symbol read the same two strings.
SMART_RESIZE_MODULE = "transformers.models.glm5_next.image_processing_glm5_next"
SMART_RESIZE_NAME = "smart_resize"


def require_transformers_smart_resize():
    """Return transformers' own ``smart_resize``, refusing by name when it is not reachable.

    The one place in this fork that names a private transformers path, because the function is not on the
    package's public surface: ``transformers.models.glm5_next`` is a lazy module built from each submodule's
    ``__all__``, and those name the classes only, so importing ``smart_resize`` from the package raises
    ``AttributeError``.

    A private name can move within the supported version range, and that trade is deliberate: a second
    implementation of the arithmetic would drift silently, while a moved import fails loudly. This refusal is
    what makes it loud, and it names the remedy. Do not answer it by reinstating the arithmetic here.
    """
    require_transformers_glm5_next()

    import importlib

    import transformers

    try:
        module = importlib.import_module(SMART_RESIZE_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            f"{SMART_RESIZE_MODULE} is not importable in the installed transformers "
            f"{transformers.__version__}. GLM-5.3-Flash consumes its canvas arithmetic from that module and "
            "does not re-derive it. Port this one helper to wherever the function now lives."
        ) from exc

    smart_resize = getattr(module, SMART_RESIZE_NAME, None)
    if smart_resize is None:
        raise RuntimeError(
            f"{SMART_RESIZE_MODULE} carries no {SMART_RESIZE_NAME} in the installed transformers "
            f"{transformers.__version__}. It is a private name, so a release inside the declared range "
            f"(>={GLM5_NEXT_TRANSFORMERS_FLOOR},<6.0.0) can move or rename it. Port this one helper to the "
            "new name rather than reinstating the arithmetic here."
        )
    return smart_resize


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

    def aligned_frames(self, num_frames: int) -> int:
        """Frames the PIXEL BUDGET is measured over, aligned to whole temporal patches.

        Not the grid's frame count, and the difference is deliberate. The budget rounds to the nearest
        multiple, as transformers' ``smart_resize`` does before comparing against the floor, while the grid
        pads up by ``-num_frames % temporal_patch_size``, as its ``patchify`` does. Two roundings of the same
        quantity, so two methods rather than one helper that would have to be right for both.
        """
        return max(
            self.temporal_patch_size,
            round(num_frames / self.temporal_patch_size) * self.temporal_patch_size,
        )


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

    ``resampled`` is the processor's own condition -- the content size differing from the input size is what
    guards its resize call -- rather than a second rule that could disagree with it.
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


def resolve_canvas(
    consts: GridConstants, *, num_frames: int, height: int, width: int, ceiling_pixels: int | None = None
) -> tuple[int, int, str]:
    """The canvas an input is padded or scaled onto, and which regime decided it.

    The canvas is transformers' answer, not a second implementation of it. Only the regime label is derived
    here, because ``smart_resize`` hands back two numbers and never says which branch produced them.

    The label comes from transformers' own condition evaluated on a canvas transformers produced: calling the
    same function with the floor switched off returns the plainly aligned canvas, which is what transformers
    tests its floor against, and above the ceiling is then the remaining case on the same budget. Reading the
    branch this way means the label cannot disagree with the canvas beside it.

    ``num_frames`` is what the caller hands the resizer: ``temporal_patch_size`` for a still image, the real
    frame count for a video. Getting it wrong moves regime B's answer, which is why it is a parameter rather
    than an assumption.

    The ceiling is expressed to transformers in tokens, so an override that is not a whole number of tokens is
    truncated down. The default is ``max_image_tokens`` exactly.

    One configuration is out of scope by construction: a processor whose ``min_image_tokens`` exceeds its
    ``max_image_tokens`` could take the floor branch and still land above the ceiling, and the label would read
    the floor.
    """
    smart_resize = require_transformers_smart_resize()
    factor = consts.factor
    ceiling = consts.ceiling_pixels if ceiling_pixels is None else ceiling_pixels

    def canvas(min_tokens: int, max_tokens: int) -> tuple[int, int]:
        return smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=consts.temporal_patch_size,
            factor=factor,
            min_pixels=min_tokens,
            max_pixels=max_tokens,
        )

    # A token count the plainly aligned canvas cannot reach, so passing it switches the ceiling off without a
    # magic number: alignment adds under one factor to each side, and the budget's frame count exceeds the
    # input's by at most one temporal patch.
    unreachable_tokens = (
        (num_frames + consts.temporal_patch_size) * (height + factor) * (width + factor)
    ) // consts.pixels_per_token + 1

    plain_height, plain_width = canvas(0, unreachable_tokens)
    canvas_height, canvas_width = canvas(consts.min_image_tokens, ceiling // consts.pixels_per_token)

    plain_budget = consts.aligned_frames(num_frames) * plain_height * plain_width
    regime = REGIME_PLAIN
    if plain_budget < consts.floor_pixels:
        regime = REGIME_BELOW_FLOOR
    elif plain_budget > ceiling:
        regime = REGIME_ABOVE_CEILING

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
    same order, so a floating-point boundary lands the same way in both. The guard is the part worth naming:
    an input already big enough for the floor may never be scaled up, so the scale is clamped at 1.0 and a
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

    Written out here rather than taken from vLLM's shared qwen2-vl factory, because that helper is private and
    a private import would be a place for the two versions to drift apart silently. ``timestamps`` is absent
    because this checkpoint's video processor does not return one.
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


def closest_factor_pair(count: int) -> tuple[int, int]:
    """Split ``count`` into the two whole factors nearest each other, shorter side first.

    Used to turn a token budget into a token grid. Walking down from the integer square root returns the
    first divisor at or below it, which is the closest-to-square pair by construction, so the caller gets the
    least elongated rectangle of that exact area without searching for it.

    A prime ``count`` has only ``(1, count)``, and that is returned rather than smoothed over. The caller is
    what decides whether such a shape is acceptable.
    """
    for divisor in range(math.isqrt(count), 0, -1):
        if count % divisor == 0:
            return divisor, count // divisor
    return 1, count


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

        Omitting video is the shape most of vLLM's own image-only models use. The video grid arithmetic in
        this module works, but the video placeholder is a per-frame timestamp structure needing video metadata
        this bridge does not receive. Offering video with a placeholder it cannot build would drop frames
        silently.
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
        """The input reaching the most placeholder tokens the ceiling admits. It is not a square.

        vLLM profiles memory with this size and then refuses any request needing more tokens than the profiled
        run reserved. A size that falls short therefore does not merely waste memory: it makes the server
        reject requests it has the capacity to serve.

        A square cannot be the answer, because the token ceiling limits the area of the token grid and a square
        grid's area must be a perfect square. This checkpoint's ceiling is 8000 tokens, which is not one: the
        best square is 89x89 = 7921, leaving 79 tokens unreachable by any square. Stepping the side more finely
        does not help, since the shape is the limit rather than the step. So the ceiling is factorised into a
        height and a width directly, as vLLM's own ``qwen2_vl`` does.

        Every number is derived from the processor. The token unit is ``patch_size * merge_size``, the pixel
        side of one merged token, which is not ``factor``: factor is the canvas alignment and equals the token
        unit times ``patch_expand_factor``. The two coincide on this checkpoint only because that expand factor
        is 1. Each token factor must be a whole number of expand units, or the pixel size is not
        factor-aligned and the processor hands back a different canvas than the one asked for.

        Suitability is two predicates, of different kinds:

        * **Required.** The token counter, driven through the real processor, reads exactly the target for the
          candidate. That one call also settles the round trip, since a canvas the processor would shrink or
          pad comes back with a different count. This is what makes the answer true.
        * **Chosen.** Neither side is thinner than ``isqrt(min_image_tokens)`` token units. The config does not
          require this -- ``min_image_tokens`` constrains the token total, not either side, and the processor's
          real per-side minimum is one token unit. It is kept because the dummy inputs builder synthesises a
          real image at this size for the profiling run, and without the bound a ceiling whose closest pair is
          degenerate (any prime) would profile on a strip one token tall and thousands wide -- a shape no
          request resembles. The bound costs at most a token or two of the ceiling, because the search steps
          down one count at a time and a short run of consecutive counts contains one with a divisor at or
          above the floor. On this checkpoint it is inert: the first candidate's shorter side is 80 token
          units. Where the exact ceiling matters more than the profiling image's shape, lower it to one and
          take the degenerate pair knowingly.

        The aspect-ratio guard ``qwen2_vl`` uses is deliberately not ported: that model's helper rejects ratios
        above 200, while this checkpoint's ``smart_resize`` has no ratio constraint at all, so porting the
        number would import a constraint the processor does not have.

        If no pair at the ceiling passes, the target steps down and the search repeats.

        Raises:
            RuntimeError: no aligned size reaches any count at or below ``max_image_tokens``, which means the
                grid constants are inconsistent.
        """
        consts = self.get_grid_constants()
        token_unit = consts.patch_size * consts.merge_size
        expand_step = consts.patch_expand_factor
        # A bound chosen here, computed from the config but not demanded by it. See the
        # docstring for what it costs and when a future port should drop it.
        chosen_min_side_tokens = math.isqrt(consts.min_image_tokens)

        for target_tokens in range(consts.max_image_tokens, 0, -1):
            height_factor, width_factor = closest_factor_pair(target_tokens)
            if height_factor % expand_step or width_factor % expand_step:
                continue
            if height_factor < chosen_min_side_tokens:
                continue
            height = token_unit * height_factor
            width = token_unit * width_factor
            if self.get_num_image_tokens(image_width=width, image_height=height) != target_tokens:
                continue
            return ImageSize(width=width, height=height)

        raise RuntimeError(
            "no aligned image size reaches any token count at or below "
            f"max_image_tokens={consts.max_image_tokens} for this processor. The search covers every count "
            "down to one, so reaching this line means the grid constants are inconsistent -- report them "
            "rather than widening the search."
        )


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

    The class is passed in rather than imported here, so the package's ``__init__`` stays the single place the
    two halves meet and no import cycle is created. vLLM resolves this architecture through the lazy name
    ``vllm_neuron.model.glm5_next:Glm5NextForConditionalGeneration``, whose module part is the package, so
    importing the package is what the resolution itself does and the registration has run by the time vLLM
    reads the class attribute.

    Nothing here is defensive, on purpose: a registration that failed quietly would leave the server running
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
    "SMART_RESIZE_MODULE",
    "SMART_RESIZE_NAME",
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
    "require_transformers_smart_resize",
    "resolve_canvas",
    "video_grid_spec",
]
