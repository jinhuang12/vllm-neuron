# SPDX-License-Identifier: Apache-2.0
"""Vision patch embedding: a thin WRAP of the substrate's NKI ``conv3d``.

`inc-glm53f-057`. This is WP10's first kernel-class increment -- the per-patch
convolution the GLM-5.3-Flash vision tower applies to cut an image into patch
vectors before the tower's attention blocks run.

It is **kernel-class** under P13, and it is a **WRAP, not SCRATCH**: ``nkilib``
already ships the kernel this increment needs, so nothing here authors kernel
numerics::

    nkilib.experimental.conv.conv3d.conv3d

That member is an ``@nki.jit`` tensor-engine 3-D convolution whose own docstring
records it as "3D Convolution using tensor engine with K-replication strategy
and W-contiguous tiling". Wrapping it is what P13 asks for -- the existing NKI
member is the substrate, and a torch-level convolution written here would be
exactly the fallback D6 forbids.

**No torch numerics are authored in this module at all**, not even for the
oracle. The reference this module exposes delegates to the substrate's own
reference, ``conv3d_torch_ref``, which ships beside the kernel for precisely
this purpose. So the acceptance compares the simulated NKI kernel against the
vendor's reference for that kernel, and neither side of the comparison is code
this increment wrote.

What "patch embedding" is, in this kernel's terms
------------------------------------------------
Cutting an image into non-overlapping ``P x P`` patches and projecting each
patch to a hidden vector IS a convolution whose kernel window equals its stride.
The substrate ships that convolution in three dimensions, so this module reaches
it with the depth axis DEGENERATE:

* ``K_d = 1`` -- the filter spans one depth slice, so the depth axis contracts
  nothing;
* ``stride = (1, P, P)`` -- unit stride on depth, patch stride on height and
  width, which is what makes the windows non-overlapping in the two axes that
  carry the image.

The kernel's own Intended Usage Range admits both (``K_d: 1-64``, stride
``1-64`` per dimension), so this is a supported degeneration and not a trick.
:func:`patch_embed` derives the stride triple from one ``patch_size`` argument
and REFUSES a filter whose depth extent is not 1, which is the whole of what
this wrap adds to the substrate on the argument side.

What this module adds to the substrate
--------------------------------------
Four things, and nothing else:

1. **A seam** -- :func:`patch_embed` -- that the route predicate counts.
2. **A gate** -- :func:`can_run_patch_embed` -- combining "is there a device or
   a simulator" with "does this kernel accept these extents".
3. **Derivation of the two things a caller must not get wrong**: the stride
   triple ``(1, P, P)`` and the ``K_d = 1`` degeneration. With the substrate
   default ``stride = (1, 1, 1)`` the same call computes OVERLAPPING windows and
   returns a different extent, which the acceptance records as a control, so the
   derivation is measured rather than assumed.
4. **A return-type normalisation.** The kernel returns a tensor; the substrate's
   torch reference returns ``{"out": tensor}``. Both paths through this module
   return a plain tensor, so a caller and a test compare like with like. (The
   key is ``"out"``, read off this kernel's own reference -- the conv1d
   reference `inc-glm53f-034` wraps uses ``"output"``, and assuming one from the
   other would be a defect.)

What this module does NOT do, and why that matters
--------------------------------------------------
It computes **no vision patch grid**. The rule that turns an image size into a
patch grid is transformers' and the fork re-derives it nowhere (design ruling
``design-20260905-aq`` (iii)); the fork's single consumer of that rule is
`inc-glm53f-056`'s ``vision_preprocessing`` module. This module is not that
consumer and does not import it: no module under ``vllm_neuron/functional/``
imports ``vllm_neuron.model`` (measured, 0 of them, while 11 model modules
import ``vllm_neuron.functional``), so reaching across that boundary here would
invert the tree's dependency direction and pull vLLM's multimodal registry into
a ``functional/`` import. The arithmetic in this file is the SUBSTRATE's
convolution output formula and nothing else, stated once in
:func:`output_extents` and cited to the kernel's own docstring. Joining a patch
grid to this seam is the model-side consumer's job (`inc-glm53f-060`).

Route
-----
Acceptance is Tier N: the NKI simulator, reached through this module's own
:func:`patch_embed` seam (``wrap_nki -> NKIHOPCaller -> HOP -> DispatchKey.CPU
-> nki.simulator.simulate_kernel``), on the harness form `inc-glm53f-025`
declared and `inc-glm53f-034` landed. The seam counts its dispatches, and the
counters are module-level state with module-level reset and read functions,
mirroring `inc-glm53f-026`'s landed placement
(``functional/blockwise_fp8_mm.py``) and `inc-glm53f-034`'s
(``functional/kda/depthwise_conv1d.py``) so that every seam a later route
predicate reads presents one shape.

Under F1 a numeric comparison alone cannot prove a kernel ran -- a torch
fallback would put torch on both sides of the comparison and pass green -- so
the counters below are acceptance criteria, not diagnostics.

How this kernel states its constraints, and why the gate is written from it
--------------------------------------------------------------------------
**Read, not assumed:** this kernel carries **zero** ``assert`` statements. It
documents an *Intended Usage Range* instead, and it takes LNC sharding as an
opt-in argument (``lnc_shard``, default ``False``) rather than imposing a shard
divisibility rule. So the channel-divisibility refusal `inc-glm53f-034` needs
for the substrate's depthwise conv1d **does not transfer to this kernel**, and
copying it here would refuse geometry the substrate serves. The refusals below
are this kernel's documented ranges, each named beside the extent it bounds.

A refused geometry **raises**; it never routes to the torch reference. Falling
back would ship a torch path for kernel-class work (P13, D6).
"""

from __future__ import annotations

import logging
from typing import Optional

from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.experimental.conv.conv3d import conv3d
from nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The depth extent this wrap degenerates. Both the input's depth axis and the
#: filter's depth extent are 1, because a patch embedding is a 2-D operation
#: expressed in the substrate's 3-D ``[B, C_in, D, H, W]`` argument layout.
#: ``K_d = 1`` is the plan block's own words for this increment.
SINGLETON_D = 1

#: Zero padding on all three axes, as the substrate spells it: a flat
#: ``(pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left,
#: pad_w_right)``. A patch embedding over an already-aligned canvas needs none,
#: and the canvas alignment is the preprocessing module's job, not this seam's.
NO_PADDING = (0, 0, 0, 0, 0, 0)

#: Unit dilation on all three axes. A dilated patch window would sample gaps
#: between pixels, which is not what a patch embedding is.
UNIT_DILATION = (1, 1, 1)

#: LNC sharding stays OFF. This kernel takes sharding as an opt-in argument, so
#: nothing here has to be divisible by a shard count -- named as a constant so
#: the value this wrap passes is visible rather than defaulted silently.
LNC_SHARD = False

# --------------------------------------------------------------------------- #
# The kernel's OWN documented Intended Usage Range, transcribed once.          #
# Each bound is (low, high), inclusive, and each is quoted in the refusal that #
# uses it so a reader can check the refusal against the substrate's docstring  #
# rather than against prose here.                                             #
# --------------------------------------------------------------------------- #
#: "B: 1-128"
BATCH_RANGE = (1, 128)
#: "C_in: 3-1280" -- the range the plan block cites as covering a 3-channel
#: image tower.
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
    refusal that names the offending extent is what a caller can act on. Raising
    is also what P13 requires here: a geometry this kernel cannot serve must NOT
    quietly route to the torch reference, since that would ship a torch path for
    kernel-class work (D6).
    """


def patch_stride(patch_size: int) -> tuple[int, int, int]:
    """The stride triple a patch embedding needs: ``(1, P, P)``.

    One function so the seam, a consumer and a test read the same triple instead
    of three copies of the same tuple literal. Unit stride on depth is what
    makes the ``K_d = 1`` degeneration a no-op on that axis; ``P`` on height and
    width is what makes the patch windows non-overlapping.

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

    and identically for H and W. It is written once here so the seam, a consumer
    and a test all read the same number instead of restating the arithmetic.

    **This is the SUBSTRATE's convolution formula, not a vision patch grid.** The
    rule that turns an image size into a patch grid belongs to transformers and
    is consumed in the fork in exactly one place (`inc-glm53f-056`); this
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


# --------------------------------------------------------------------------- #
# Geometry admission.                                                          #
# --------------------------------------------------------------------------- #
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

    Each condition names what needs it -- either this wrap's ``K_d = 1`` /
    ``(1, P, P)`` degeneration, or a bound the kernel's own Intended Usage Range
    documents -- so a reader can check the refusal against the substrate rather
    than against prose.
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

    # --- this wrap's own degeneration ------------------------------------- #
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
            f"patch_size={patch_size}; a patch embedding's window IS its "
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

    # --- dtype ------------------------------------------------------------- #
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

    # --- channel agreement ------------------------------------------------- #
    if f_c_in != c_in:
        problems.append(
            f"filters C_in={f_c_in} does not match x_in C_in={c_in}"
        )

    # --- the kernel's OWN documented Intended Usage Range ------------------ #
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

    # --- options this wrap does not pass through --------------------------- #
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


# --------------------------------------------------------------------------- #
# The route seam and its counters.                                             #
# --------------------------------------------------------------------------- #
class _DispatchCounters:
    """What route actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` seam;
    ``torch_fallback`` counts entries into the reference path. Two counters
    rather than one flag, so "the kernel ran" and "the fallback did not run" are
    independent readings and a test can require both.
    """

    def __init__(self) -> None:
        self.nki_dispatch = 0
        self.torch_fallback = 0


#: MODULE-LEVEL, on `inc-glm53f-026`'s and `inc-glm53f-034`'s landed placement:
#: a route predicate taken over this seam from another increment's test module
#: must be able to zero and read these counters from outside this file. A
#: test-local counter would satisfy this increment and break that one.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


def can_run_patch_embed(
    x_in: Tensor,
    filters: Tensor,
    patch_size: int,
    bias: Optional[Tensor] = None,
    padding: tuple[int, int, int, int, int, int] = NO_PADDING,
    dilation: tuple[int, int, int] = UNIT_DILATION,
) -> bool:
    """Is the NKI route available *and* admissible for this call?

    Two independent conditions, deliberately not merged:
    :func:`~vllm_neuron.utils.neuron_utils.can_run_kernel` answers "is there a
    device or a simulator", :func:`_require_admissible` answers "does the
    substrate kernel accept these extents and options".

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
    """Patch-embed an aligned image. The seam the route predicate counts.

    Args:
        x_in: ``[B, C_in, 1, H, W]`` input, with ``H`` and ``W`` whole multiples
            of ``patch_size``.
        filters: ``[1, P, P, C_in, C_out]`` patch projection, in the substrate's
            own filter layout (depth, height, width, in-channels, out-channels).
        patch_size: ``P``. The seam derives ``stride = (1, P, P)`` from it.
        bias: optional ``[C_out]``.
        padding: must be :data:`NO_PADDING`.
        dilation: must be :data:`UNIT_DILATION`.

    Returns:
        ``[B, C_out, 1, H // P, W // P]`` at the accumulation dtype the kernel
        returns, with the extents :func:`output_extents` states.

    Raises:
        VisionPatchEmbedError: on an inadmissible geometry, dtype or option.

    ``stride`` is **not** a parameter of this seam. The substrate kernel defaults
    it to ``(1, 1, 1)``, which on these same arguments computes OVERLAPPING
    windows at every pixel offset and returns a different extent -- a silently
    wrong patch embedding rather than an error. This seam derives it from
    ``patch_size`` instead, which is the one piece of argument handling a thin
    wrap is for.
    """
    if not can_run_patch_embed(x_in, filters, patch_size, bias, padding, dilation):
        _COUNTERS.torch_fallback += 1
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

    _COUNTERS.nki_dispatch += 1
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
    """The substrate's OWN torch reference for this kernel. Never shipped.

    Delegates to ``conv3d_torch_ref``, which ships in ``nkilib`` beside the
    kernel. **No torch numerics are authored here**: this function derives the
    stride triple exactly as the seam does and unwraps the reference's
    ``{"out": tensor}`` return so both paths through this module hand back a
    plain tensor.

    The dictionary key is ``"out"``, read off THIS kernel's reference. The
    conv1d reference `inc-glm53f-034` wraps returns ``"output"``; the two
    vendor references disagree, so the key is read rather than carried over.

    This is the acceptance's comparison target and the constraint-violation
    return for a route the gate reports unavailable. It is **never** the shipped
    kernel-class path (P13, D6).

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

    Exposed so a test can assert the seam dispatches to the SUBSTRATE's member
    rather than to anything authored here -- which is how a WRAP is checkable
    rather than merely claimed -- and so a substitution shows up as a changed
    reading instead of as silence.

    **This reading is NOT part of any route predicate** and may not be cited as
    one (D13.1): it certifies what this module imported, not what ran. The route
    reading is :func:`dispatch_counters`, taken through the seam.
    """
    func = getattr(conv3d, "func", None)
    target = func if func is not None else conv3d
    return target.__module__, target.__qualname__
