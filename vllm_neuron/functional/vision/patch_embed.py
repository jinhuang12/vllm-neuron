# SPDX-License-Identifier: Apache-2.0
"""Vision patch embedding: a thin wrap of ``nkilib``'s NKI ``conv3d``.

The per-patch convolution the vision tower applies to cut an image into patch
vectors before its attention blocks run. ``nkilib`` already ships the kernel, so
no numerics are authored here::

    nkilib.experimental.conv.conv3d.conv3d

The torch reference this module exposes likewise delegates to the substrate's own
``conv3d_torch_ref``, which ships beside the kernel.

Patch embedding as a convolution
--------------------------------
Cutting an image into non-overlapping ``P x P`` patches and projecting each patch
to a hidden vector is a convolution whose kernel window equals its stride. The
substrate ships that convolution in three dimensions, so this module reaches it
with the depth axis degenerate:

* ``K_d = 1``, so the filter spans one depth slice and the depth axis contracts
  nothing;
* ``stride = (1, P, P)``, unit stride on depth and patch stride on height and
  width, which is what makes the windows non-overlapping in the two axes that
  carry the image.

The kernel's documented Intended Usage Range admits both (``K_d: 1-64``, stride
``1-64`` per dimension), so this is a supported degeneration.

What the wrap adds is argument handling the kernel makes easy to get wrong: it
derives the stride triple from one ``patch_size`` argument and refuses a filter
whose depth extent is not 1 (the substrate default ``stride = (1, 1, 1)`` would
compute overlapping windows and return a different extent), it checks every
extent in one place, and it unwraps the reference's ``{"out": tensor}`` return so
both paths hand back a plain tensor. That key is ``"out"`` here and ``"output"``
in the conv1d reference; assuming one from the other is a defect.

This module computes no patch grid. The rule that turns an image size into a
patch grid belongs to the model side, and nothing under
``vllm_neuron/functional/`` imports ``vllm_neuron.model``, so reaching for it
here would invert the tree's dependency direction. The only arithmetic in this
file is the substrate's convolution output formula, stated once in
:func:`output_extents`.

The refusals below are the kernel's own documented ranges, each named beside the
extent it bounds. The kernel carries no ``assert`` statements, and it takes LNC
sharding as an opt-in argument (``lnc_shard``, default ``False``) rather than
imposing a shard divisibility rule, so the channel-divisibility refusal the
depthwise conv1d needs does not transfer here; copying it would refuse geometry
the substrate serves. A refused geometry raises; it never routes to the torch
reference.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.experimental.conv.conv3d import conv3d
from nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The depth extent this wrap degenerates. Both the input's depth axis and the
#: filter's depth extent are 1, because a patch embedding is a 2-D operation
#: expressed in the substrate's 3-D ``[B, C_in, D, H, W]`` argument layout.
SINGLETON_D = 1

#: Zero padding on all three axes, as the substrate spells it: a flat
#: ``(pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left,
#: pad_w_right)``. A patch embedding over an already-aligned canvas needs none, and
#: aligning the canvas is the preprocessing module's job.
NO_PADDING = (0, 0, 0, 0, 0, 0)

#: Unit dilation on all three axes. A dilated patch window would sample gaps
#: between pixels, which is not what a patch embedding is.
UNIT_DILATION = (1, 1, 1)

#: LNC sharding stays off. This kernel takes sharding as an opt-in argument, so
#: nothing here has to be divisible by a shard count. Named so the value this wrap
#: passes is visible rather than defaulted silently.
LNC_SHARD = False

# The kernel's own documented Intended Usage Range, transcribed once. Each bound is
# (low, high), inclusive, and each is quoted in the refusal that uses it so a reader
# can check the refusal against the substrate's docstring.
#: "B: 1-128"
BATCH_RANGE = (1, 128)
#: "C_in: 3-1280"
C_IN_RANGE = (3, 1280)
#: "C_out: 3-2048"
C_OUT_RANGE = (3, 2048)
#: "D: 1-1024, H: 1-1024, W: 1-1024"
SPATIAL_RANGE = (1, 1024)
#: "K_d: 1-64, K_h: 1-64, K_w: 1-64"
FILTER_RANGE = (1, 64)
#: "Stride: 1-64 per dimension"
STRIDE_RANGE = (1, 64)
#: "Dilation: 1-64 per dimension"
DILATION_RANGE = (1, 64)

__all__ = [
    "BATCH_RANGE",
    "C_IN_RANGE",
    "C_OUT_RANGE",
    "DILATION_RANGE",
    "FILTER_RANGE",
    "LNC_SHARD",
    "NO_PADDING",
    "SINGLETON_D",
    "SPATIAL_RANGE",
    "STRIDE_RANGE",
    "VisionPatchEmbedError",
    "can_run_patch_embed",
    "dispatch_counters",
    "kernel_identity",
    "output_extents",
    "patch_embed",
    "patch_embed_torch_reference",
    "patch_stride",
    "reset_dispatch_counters",
]


class VisionPatchEmbedError(ValueError):
    """A geometry, dtype or option this wrap refuses, named rather than coerced.

    Raised in preference to letting the kernel trap at trace time, because a
    refusal that names the offending extent is what a caller can act on.
    """


def patch_stride(patch_size: int) -> tuple[int, int, int]:
    """The stride triple a patch embedding needs: ``(1, P, P)``.

    Unit stride on depth is what makes the ``K_d = 1`` degeneration a no-op on that
    axis; ``P`` on height and width is what makes the patch windows
    non-overlapping.

    Raises:
        VisionPatchEmbedError: if ``patch_size`` is outside the kernel's own
            documented stride range.
    """
    low, high = STRIDE_RANGE
    if not isinstance(patch_size, int) or isinstance(patch_size, bool):
        raise VisionPatchEmbedError(
            f"patch_size must be an int, got {type(patch_size).__name__}"
        )
    if not low <= patch_size <= high:
        raise VisionPatchEmbedError(
            f"patch_size={patch_size} is outside the substrate kernel's "
            f'documented stride range "Stride: {low}-{high} per dimension"'
        )
    return (1, patch_size, patch_size)


def output_extents(
    spatial: tuple[int, int, int],
    filter_extents: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int] = NO_PADDING,
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = UNIT_DILATION,
) -> tuple[int, int, int]:
    """``(D_out, H_out, W_out)``, from the substrate's own formula.

    The kernel's docstring states it three times, once per axis::

        D_out = (D + pad_d_left + pad_d_right
                 - dilation_d * (K_d - 1) - 1) // stride_d + 1

    and identically for H and W, written once here so callers do not restate the
    arithmetic.

    This is the substrate's convolution formula and not a vision patch grid: this
    function knows nothing about images, canvases, merge sizes or token budgets.

    Args:
        spatial: ``(D, H, W)`` input extents.
        filter_extents: ``(K_d, K_h, K_w)``.
        padding: the substrate's flat six-tuple.
        stride: ``(stride_d, stride_h, stride_w)``.
        dilation: ``(dilation_d, dilation_h, dilation_w)``.

    Raises:
        VisionPatchEmbedError: if any axis has no positive output extent, in
            which case there is nothing valid to return.
    """
    pads = ((padding[0], padding[1]), (padding[2], padding[3]), (padding[4], padding[5]))
    names = ("D", "H", "W")
    out: list[int] = []
    for axis, (extent, k, (pad_lo, pad_hi), s, d) in enumerate(
        zip(spatial, filter_extents, pads, stride, dilation)
    ):
        if s <= 0:
            raise VisionPatchEmbedError(
                f"stride_{names[axis].lower()}={s} must be positive"
            )
        effective = extent + pad_lo + pad_hi - d * (k - 1) - 1
        if effective < 0:
            raise VisionPatchEmbedError(
                f"{names[axis]}_out would be non-positive: {names[axis]}="
                f"{extent} padded by ({pad_lo}, {pad_hi}) cannot hold a "
                f"dilated filter extent of {d * (k - 1) + 1}"
            )
        out.append(effective // s + 1)
    return (out[0], out[1], out[2])


def _in_range(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _require_admissible(
    x_in: Tensor,
    filters: Tensor,
    patch_size: int,
    bias: Optional[Tensor],
    padding: tuple[int, int, int, int, int, int],
    dilation: tuple[int, int, int],
) -> None:
    """Every condition this wrap and the substrate kernel impose, in one place.

    Each message names what needs the condition, either this wrap's ``K_d = 1`` /
    ``(1, P, P)`` degeneration or a bound from the kernel's Intended Usage Range,
    so a refusal can be checked against the substrate.
    """
    problems: list[str] = []

    if x_in.dim() != 5:
        problems.append(
            f"x_in must be 5-D [B, C_in, D, H, W], got {x_in.dim()}-D "
            f"{tuple(x_in.shape)}"
        )
    if filters.dim() != 5:
        problems.append(
            f"filters must be 5-D [K_d, K_h, K_w, C_in, C_out], got "
            f"{filters.dim()}-D {tuple(filters.shape)}"
        )
    if problems:
        # Every check below indexes those axes, so stop here.
        raise VisionPatchEmbedError(
            "vision patch embed refuses this call: " + "; ".join(problems)
        )

    batch, c_in, depth, height, width = (int(v) for v in x_in.shape)
    k_d, k_h, k_w, f_c_in, c_out = (int(v) for v in filters.shape)

    # This wrap's own degeneration.
    if k_d != SINGLETON_D:
        problems.append(
            f"filter depth extent K_d={k_d}, must be {SINGLETON_D}: a patch "
            f"embedding is a 2-D operation expressed in the substrate's 3-D "
            f"layout, and this wrap exists to fix that degeneration. A real "
            f"K_d > 1 convolution is a different call to the same kernel and "
            f"is not this seam's"
        )
    if depth != SINGLETON_D:
        problems.append(
            f"x_in depth extent D={depth}, must be {SINGLETON_D} for the "
            f"K_d = {SINGLETON_D} degeneration this seam wires"
        )
    if k_h != patch_size or k_w != patch_size:
        problems.append(
            f"filter spatial extents (K_h, K_w)=({k_h}, {k_w}) must both equal "
            f"patch_size={patch_size}; a patch embedding's window is its "
            f"stride, and a window that differs from the stride either overlaps "
            f"patches or skips pixels"
        )
    if height % patch_size or width % patch_size:
        problems.append(
            f"input (H, W)=({height}, {width}) is not a whole number of "
            f"patch_size={patch_size} patches; the canvas alignment that "
            f"guarantees it is the preprocessing module's job, and padding it "
            f"here would invent pixels the tower never saw"
        )

    # Dtype.
    if x_in.dtype != filters.dtype:
        problems.append(
            f"x_in dtype {x_in.dtype} and filters dtype {filters.dtype} "
            f"differ; the kernel contracts the two tensors against each other "
            f"in one matmul"
        )
    if bias is not None:
        if bias.dim() != 1:
            problems.append(
                f"bias must be 1-D [C_out], got {bias.dim()}-D "
                f"{tuple(bias.shape)}"
            )
        elif int(bias.shape[0]) != c_out:
            problems.append(
                f"bias extent {int(bias.shape[0])} does not match C_out="
                f"{c_out}"
            )

    # Channel agreement.
    if f_c_in != c_in:
        problems.append(
            f"filters C_in={f_c_in} does not match x_in C_in={c_in}"
        )

    # The kernel's own documented Intended Usage Range.
    for label, value, bounds in (
        ("B", batch, BATCH_RANGE),
        ("C_in", c_in, C_IN_RANGE),
        ("C_out", c_out, C_OUT_RANGE),
        ("D", depth, SPATIAL_RANGE),
        ("H", height, SPATIAL_RANGE),
        ("W", width, SPATIAL_RANGE),
        ("K_d", k_d, FILTER_RANGE),
        ("K_h", k_h, FILTER_RANGE),
        ("K_w", k_w, FILTER_RANGE),
    ):
        if not _in_range(value, bounds):
            problems.append(
                f'{label}={value} is outside the substrate kernel\'s own '
                f'documented Intended Usage Range "{label}: {bounds[0]}-'
                f'{bounds[1]}"'
            )

    # Options this wrap does not pass through.
    if tuple(padding) != NO_PADDING:
        problems.append(
            f"padding={tuple(padding)} must be {NO_PADDING}: a patch embedding "
            f"runs on an already-aligned canvas, and padding it here would add "
            f"patches of invented pixels to the grid the preprocessing module "
            f"already computed"
        )
    if tuple(dilation) != UNIT_DILATION:
        problems.append(
            f"dilation={tuple(dilation)} must be {UNIT_DILATION}; a dilated "
            f"patch window would sample gaps between pixels"
        )

    if problems:
        raise VisionPatchEmbedError(
            "vision patch embed refuses this call: " + "; ".join(problems)
        )

    # Raises on its own account if any axis has no positive output extent.
    output_extents(
        (depth, height, width),
        (k_d, k_h, k_w),
        padding,
        patch_stride(patch_size),
        dilation,
    )


class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` call, ``torch_fallback``
    entries into the reference path. Two counters rather than one flag, so "the
    kernel ran" and "the fallback did not" are independent readings.
    """

    def __init__(self) -> None:
        self.nki_dispatch = 0
        self.torch_fallback = 0


#: Module level so a caller outside this file can zero and read the counters.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def can_run_patch_embed(
    x_in: Tensor,
    filters: Tensor,
    patch_size: int,
    bias: Optional[Tensor] = None,
    padding: tuple[int, int, int, int, int, int] = NO_PADDING,
    dilation: tuple[int, int, int] = UNIT_DILATION,
) -> bool:
    """Is the NKI path available *and* admissible for this call?

    Two independent conditions:
    :func:`~vllm_neuron.utils.neuron_utils.can_run_kernel` answers whether a device
    or simulator exists, :func:`_require_admissible` whether the substrate kernel
    accepts these extents and options.

    Raises:
        VisionPatchEmbedError: if the call is inadmissible. Inadmissible is not
            the same as unavailable, and it does not fall back.
    """
    _require_admissible(x_in, filters, patch_size, bias, padding, dilation)
    return can_run_kernel(x_in)


def patch_embed(
    x_in: Tensor,
    filters: Tensor,
    patch_size: int,
    bias: Optional[Tensor] = None,
    padding: tuple[int, int, int, int, int, int] = NO_PADDING,
    dilation: tuple[int, int, int] = UNIT_DILATION,
) -> Tensor:
    """Patch-embed an aligned image.

    Args:
        x_in: ``[B, C_in, 1, H, W]`` input, with ``H`` and ``W`` whole multiples
            of ``patch_size``.
        filters: ``[1, P, P, C_in, C_out]`` patch projection, in the substrate's
            own filter layout (depth, height, width, in-channels, out-channels).
        patch_size: ``P``. ``stride = (1, P, P)`` is derived from it.
        bias: optional ``[C_out]``.
        padding: must be :data:`NO_PADDING`.
        dilation: must be :data:`UNIT_DILATION`.

    Returns:
        ``[B, C_out, 1, H // P, W // P]`` at the accumulation dtype the kernel
        returns, with the extents :func:`output_extents` states.

    Raises:
        VisionPatchEmbedError: on an inadmissible geometry, dtype or option.

    ``stride`` is deliberately not a parameter here. The substrate kernel defaults
    it to ``(1, 1, 1)``, which on these same arguments computes overlapping windows
    at every pixel offset and returns a different extent: a silently wrong patch
    embedding rather than an error. It is derived from ``patch_size`` instead.
    """
    if not can_run_patch_embed(x_in, filters, patch_size, bias, padding, dilation):
        _count_torch_fallback()
        logger.debug(
            "patch_embed: NKI route unavailable, using the substrate's torch "
            "reference (reference only, not the shipped path)"
        )
        return patch_embed_torch_reference(
            x_in,
            filters,
            patch_size,
            bias=bias,
            padding=padding,
            dilation=dilation,
        )

    _count_nki_dispatch()
    return wrap_nki(conv3d)(
        x_in=x_in,
        filters=filters,
        bias=bias,
        stride=patch_stride(patch_size),
        padding=padding,
        dilation=dilation,
        activation_fn=None,
        lnc_shard=LNC_SHARD,
    )


def patch_embed_torch_reference(
    x_in: Tensor,
    filters: Tensor,
    patch_size: int,
    bias: Optional[Tensor] = None,
    padding: tuple[int, int, int, int, int, int] = NO_PADDING,
    dilation: tuple[int, int, int] = UNIT_DILATION,
) -> Tensor:
    """``nkilib``'s own torch reference for this kernel. Never the shipped path.

    Delegates to ``conv3d_torch_ref``, deriving the stride triple as
    :func:`patch_embed` does and unwrapping the reference's ``{"out": tensor}``
    return so both paths hand back a plain tensor.

    The dictionary key is ``"out"``, read off this kernel's reference. The conv1d
    reference returns ``"output"``, so the key is read rather than carried over.

    Returns:
        ``[B, C_out, 1, H // P, W // P]``.
    """
    result = conv3d_torch_ref(
        x_in,
        filters,
        bias=bias,
        stride=patch_stride(patch_size),
        padding=padding,
        dilation=dilation,
        activation_fn=None,
        lnc_shard=LNC_SHARD,
    )
    return result["out"]


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the wrapped kernel, read off the object.

    Lets a caller check that this module dispatches to ``nkilib``'s member rather
    than to anything authored here, so a substitution shows up as a changed
    reading. It certifies what this module imported and not what ran;
    :func:`dispatch_counters` is what says a kernel dispatched.
    """
    func = getattr(conv3d, "func", None)
    target = func if func is not None else conv3d
    return target.__module__, target.__qualname__
