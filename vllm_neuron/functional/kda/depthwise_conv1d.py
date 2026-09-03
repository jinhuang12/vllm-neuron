# SPDX-License-Identifier: Apache-2.0
"""KDA prefill depthwise conv1d: a thin WRAP of the substrate's NKI kernel.

`inc-glm53f-034`. This is WP3's first increment -- the per-token depthwise
convolution the KDA prefill path applies along the sequence axis before the
delta rule runs.

It is **kernel-class** under P13, and it is a **WRAP, not SCRATCH**: ``nkilib``
already ships the kernel this increment needs, so nothing here authors kernel
numerics::

    nkilib.experimental.conv.depthwise_conv1d.depthwise_conv1d_implicit_gemm

That member is an ``@nki.jit`` implicit-GEMM depthwise conv1d whose own
docstring records it as "Optimized for TRN2 platform with LNC2 sharding on
channel dimension". Wrapping it is what P13 asks for -- the existing NKI member
is the substrate, and a torch-level convolution written here would be exactly
the fallback the rule forbids.

**No torch numerics are authored in this module at all**, not even for the
oracle. The reference this module exposes delegates to the substrate's own
reference, ``depthwise_conv1d_implicit_gemm_torch_ref``, which ships beside the
kernel for precisely this purpose. So the acceptance compares the simulated NKI
kernel against the vendor's reference for that kernel, and neither side of the
comparison is code this increment wrote.

What this module adds to the substrate
--------------------------------------
Four things, and nothing else:

1. **A seam** -- :func:`depthwise_conv1d` -- that the route predicate counts.
2. **A gate** -- :func:`can_run_depthwise_conv1d` -- combining "is there a
   device or a simulator" with "does this kernel accept these extents".
3. **Derivation of the one argument a caller must not get wrong.** The kernel
   asserts ``feature_group_count == C``; this seam derives ``C`` from the input
   and passes it, so the argument is never the caller's to supply. With the
   substrate default of ``1`` the kernel refuses outright, which the acceptance
   records as a control.
4. **A return-type normalisation.** The kernel returns a tensor; the substrate's
   torch reference returns ``{"output": tensor}``. Both paths through this module
   return a plain tensor, so a caller and a test compare like with like.

Route
-----
Acceptance is Tier N: the NKI simulator, reached through this module's own
:func:`depthwise_conv1d` seam (``wrap_nki -> NKIHOPCaller -> HOP ->
DispatchKey.CPU -> nki.simulator.simulate_kernel``), on the harness form
`inc-glm53f-025` landed. The seam counts its dispatches, and the counters are
module-level state with module-level reset and read functions, mirroring
`inc-glm53f-026`'s landed placement
(``functional/blockwise_fp8_mm.py:366-369``) and `inc-glm53f-028`'s
(``functional/mhc/sinkhorn.py``) so that every seam a later route predicate
reads presents one shape.

Under F1 a numeric comparison alone cannot prove a kernel ran -- a torch
fallback would put torch on both sides of the comparison and pass green -- so
the counters below are acceptance criteria, not diagnostics.

The channel-count refusal, and its honest ground
------------------------------------------------
The substrate's own Notes say "Requires C to be divisible by NUM_SHARDS (2)",
and its body divides the channel extent by ``nl.num_programs()`` and indexes
each shard from ``shard_id * C_per_shard``. So an odd channel count on a
two-shard device would silently drop channels rather than fail. This module
therefore refuses ``C % LNC_SHARDS != 0`` up front.

**Stated plainly because it matters for reading the acceptance:** the NKI
simulator does **not** enforce that constraint -- it computes the whole output
correctly for an odd ``C``, measured. The refusal is this module's deliberate
conservatism on the substrate's documented device requirement, not a
reproduction of a simulator failure, and the acceptance records both readings
side by side so the refusal cannot be mistaken for something the kernel did.

A refused geometry **raises**; it never routes to the torch reference. Falling
back would ship a torch path for kernel-class work (P13, D6).
"""

from __future__ import annotations

import logging

from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.experimental.conv.depthwise_conv1d import (
    depthwise_conv1d_implicit_gemm,
)
from nkilib.experimental.conv.depthwise_conv1d_torch import (
    depthwise_conv1d_implicit_gemm_torch_ref,
)

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Logical Neuron Core shard count the substrate kernel shards its channel
#: extent over. From the kernel's own Notes -- "Requires C to be divisible by
#: NUM_SHARDS (2)" -- and from its ``C_per_shard = C // nl.num_programs()``.
#: Named here so the refusal below and any consumer read one number.
LNC_SHARDS = 2

#: The one spatial extent the kernel's layout fixes: both the image and the
#: filter carry a singleton height axis, because this is a 1-D convolution
#: expressed in the substrate's 2-D ``[N, C, H, W]`` argument layout.
SINGLETON_H = 1

#: Zero padding on both sides of both axes, and the only padding the substrate
#: kernel supports ("Only supports zero padding"). Written as the module default
#: so a caller that wants no padding does not have to spell the nesting.
NO_PADDING = ((0, 0), (0, 0))

#: Unit stride on both axes. ``stride_h`` must be 1 -- the kernel asserts it --
#: while ``stride_w`` may be any positive integer.
UNIT_STRIDE = (1, 1)

#: The only dilation the kernel supports, on both axes, asserted in its body.
UNIT_DILATION = (1, 1)

__all__ = [
    "LNC_SHARDS",
    "NO_PADDING",
    "SINGLETON_H",
    "UNIT_DILATION",
    "UNIT_STRIDE",
    "KdaDepthwiseConv1dError",
    "can_run_depthwise_conv1d",
    "depthwise_conv1d",
    "depthwise_conv1d_torch_reference",
    "dispatch_counters",
    "kernel_identity",
    "output_width",
    "reset_dispatch_counters",
]


class KdaDepthwiseConv1dError(ValueError):
    """A geometry, dtype or option this wrap refuses, named rather than coerced.

    Raised in preference to letting the kernel trap at trace time, because a
    refusal that names the offending extent is what a caller can act on. Raising
    is also what P13 requires here: a geometry this kernel cannot serve must NOT
    quietly route to the torch reference, since that would ship a torch path for
    kernel-class work (D6).
    """


def output_width(
    width: int,
    kernel_size: int,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
) -> int:
    """The output extent ``Q``, from the substrate's own formula.

    ``Q = (W + W_pad_l + W_pad_r - S) // stride_w + 1``, stated once here so the
    seam, a consumer and a test all read the same number instead of restating
    the arithmetic. The substrate documents this formula on both the kernel and
    its reference.

    Raises:
        KdaDepthwiseConv1dError: if the padded width cannot hold the kernel, in
            which case there is no valid output extent to return.
    """
    pad_left, pad_right = padding[1]
    stride_w = stride[1]
    if stride_w <= 0:
        raise KdaDepthwiseConv1dError(f"stride_w={stride_w} must be positive")
    padded = width + pad_left + pad_right
    if padded < kernel_size:
        raise KdaDepthwiseConv1dError(
            f"padded width {padded} (W={width} + {pad_left} + {pad_right}) is "
            f"smaller than the kernel size S={kernel_size}, so the output "
            f"extent Q would be non-positive"
        )
    return (padded - kernel_size) // stride_w + 1


# --------------------------------------------------------------------------- #
# Geometry admission.                                                          #
# --------------------------------------------------------------------------- #
def _require_admissible(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]],
    stride: tuple[int, int],
    rhs_dilation: tuple[int, int],
    lhs_dilation: tuple[int, int],
    batch_group_count: int,
) -> None:
    """Every condition the substrate kernel imposes, checked in one place.

    Each condition names what in the kernel needs it, so a reader can check the
    refusal against the substrate rather than against prose.
    """
    problems: list[str] = []

    if img.dim() != 4:
        problems.append(
            f"img must be 4-D [N, C, 1, W], got {img.dim()}-D "
            f"{tuple(img.shape)}"
        )
    if filt.dim() != 4:
        problems.append(
            f"filter must be 4-D [C, 1, 1, S], got {filt.dim()}-D "
            f"{tuple(filt.shape)}"
        )
    if problems:
        # Every check below indexes those four axes, so stop here.
        raise KdaDepthwiseConv1dError(
            "kda depthwise conv1d refuses this call: " + "; ".join(problems)
        )

    channels = int(img.shape[1])

    if int(img.shape[2]) != SINGLETON_H:
        problems.append(
            f"img height extent is {int(img.shape[2])}, must be {SINGLETON_H}: "
            f"this is a 1-D convolution in the substrate's 2-D argument layout"
        )
    if int(filt.shape[0]) != channels:
        problems.append(
            f"filter channel extent {int(filt.shape[0])} does not match the "
            f"img channel extent {channels}; a depthwise convolution carries "
            f"one filter per channel"
        )
    if int(filt.shape[1]) != SINGLETON_H or int(filt.shape[2]) != SINGLETON_H:
        problems.append(
            f"filter must be [C, 1, 1, S], got {tuple(filt.shape)}"
        )
    if channels % LNC_SHARDS != 0:
        problems.append(
            f"C={channels} is not divisible by LNC_SHARDS={LNC_SHARDS}; the "
            f"substrate kernel shards its channel extent over the logical "
            f"cores and its own Notes require the division to be exact, so an "
            f"odd channel count would drop channels on a two-shard device "
            f"rather than fail. Refused here rather than routed to torch, "
            f"which would ship a torch path for kernel-class work"
        )
    if img.dtype != filt.dtype:
        problems.append(
            f"img dtype {img.dtype} and filter dtype {filt.dtype} differ; the "
            f"kernel allocates its output at the img dtype and contracts the "
            f"two tensors against each other in one matmul"
        )
    if padding[0] != (0, 0):
        problems.append(
            f"height padding {padding[0]} must be (0, 0): the kernel reads "
            f"padding[1] only, so height padding would be silently dropped "
            f"rather than applied"
        )
    if min(padding[1]) < 0:
        problems.append(f"width padding {padding[1]} must be non-negative")
    if stride[0] != 1:
        problems.append(
            f"stride_h={stride[0]} must be 1; the kernel asserts it"
        )
    if stride[1] <= 0:
        problems.append(f"stride_w={stride[1]} must be positive")
    if tuple(rhs_dilation) != UNIT_DILATION:
        problems.append(
            f"rhs_dilation={tuple(rhs_dilation)} must be {UNIT_DILATION}; the "
            f"kernel asserts it"
        )
    if tuple(lhs_dilation) != UNIT_DILATION:
        problems.append(
            f"lhs_dilation={tuple(lhs_dilation)} must be {UNIT_DILATION}; the "
            f"kernel asserts it"
        )
    if batch_group_count != 1:
        problems.append(
            f"batch_group_count={batch_group_count} must be 1; the kernel "
            f"asserts it"
        )

    if problems:
        raise KdaDepthwiseConv1dError(
            "kda depthwise conv1d refuses this call: " + "; ".join(problems)
        )

    # Raises on its own account if the padded width cannot hold the kernel.
    output_width(int(img.shape[3]), int(filt.shape[3]), padding, stride)


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


#: MODULE-LEVEL, on `inc-glm53f-026`'s and `inc-glm53f-028`'s landed placement:
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


def can_run_depthwise_conv1d(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
    rhs_dilation: tuple[int, int] = UNIT_DILATION,
    lhs_dilation: tuple[int, int] = UNIT_DILATION,
    batch_group_count: int = 1,
) -> bool:
    """Is the NKI route available *and* admissible for this call?

    Two independent conditions, deliberately not merged:
    :func:`~vllm_neuron.utils.neuron_utils.can_run_kernel` answers "is there a
    device or a simulator", :func:`_require_admissible` answers "does the
    substrate kernel accept these extents and options".

    Raises:
        KdaDepthwiseConv1dError: if the call is inadmissible. Inadmissible is
            not the same as unavailable, and it does not fall back.
    """
    _require_admissible(
        img, filt, padding, stride, rhs_dilation, lhs_dilation, batch_group_count
    )
    return can_run_kernel(img)


def depthwise_conv1d(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
    rhs_dilation: tuple[int, int] = UNIT_DILATION,
    lhs_dilation: tuple[int, int] = UNIT_DILATION,
    batch_group_count: int = 1,
) -> Tensor:
    """Depthwise conv1d over the sequence axis. The seam the predicate counts.

    Args:
        img: ``[N, C, 1, W]`` input.
        filt: ``[C, 1, 1, S]`` depthwise taps, one filter per channel.
        padding: ``((0, 0), (W_pad_l, W_pad_r))``. Height padding must be zero;
            width padding must be zero or positive. Defaults to
            :data:`NO_PADDING`.
        stride: ``(1, stride_w)``. Defaults to :data:`UNIT_STRIDE`.
        rhs_dilation: must be :data:`UNIT_DILATION`.
        lhs_dilation: must be :data:`UNIT_DILATION`.
        batch_group_count: must be ``1``.

    Returns:
        ``[N, C, 1, Q]`` at the input dtype, with ``Q`` from
        :func:`output_width`.

    Raises:
        KdaDepthwiseConv1dError: on an inadmissible geometry, dtype or option.

    ``feature_group_count`` is **not** a parameter of this seam. The substrate
    kernel asserts ``feature_group_count == C`` and defaults it to ``1``, so
    every caller would have to restate the channel extent correctly or get a
    trace-time failure. This seam derives it from ``img`` instead, which is the
    one piece of argument handling a thin wrap is for.
    """
    if not can_run_depthwise_conv1d(
        img, filt, padding, stride, rhs_dilation, lhs_dilation, batch_group_count
    ):
        _COUNTERS.torch_fallback += 1
        logger.debug(
            "depthwise_conv1d: NKI route unavailable, using the substrate's "
            "torch reference (reference only, not the shipped path)"
        )
        return depthwise_conv1d_torch_reference(
            img,
            filt,
            padding=padding,
            stride=stride,
            rhs_dilation=rhs_dilation,
            lhs_dilation=lhs_dilation,
            batch_group_count=batch_group_count,
        )

    _COUNTERS.nki_dispatch += 1
    return wrap_nki(depthwise_conv1d_implicit_gemm)(
        img_ref=img,
        filter_ref=filt,
        padding=padding,
        stride=stride,
        rhs_dilation=rhs_dilation,
        lhs_dilation=lhs_dilation,
        feature_group_count=int(img.shape[1]),
        batch_group_count=batch_group_count,
    )


def depthwise_conv1d_torch_reference(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
    rhs_dilation: tuple[int, int] = UNIT_DILATION,
    lhs_dilation: tuple[int, int] = UNIT_DILATION,
    batch_group_count: int = 1,
) -> Tensor:
    """The substrate's OWN torch reference for this kernel. Never shipped.

    Delegates to ``depthwise_conv1d_implicit_gemm_torch_ref``, which ships in
    ``nkilib`` beside the kernel. **No torch numerics are authored here**: this
    function derives ``feature_group_count`` exactly as the seam does and
    unwraps the reference's ``{"output": tensor}`` return so both paths through
    this module hand back a plain tensor.

    This is the acceptance's comparison target and the constraint-violation
    return for a route the gate reports unavailable. It is **never** the shipped
    kernel-class path (P13, D6).

    One substrate asymmetry a caller must know, because this wrap does not hide
    it: the reference applies width padding through ``F.conv2d``, which pads
    both sides by the SAME amount, while the kernel pads left and right
    independently. The two therefore agree only for symmetric width padding,
    :data:`NO_PADDING` included. Asymmetric padding is a real kernel capability
    that this reference cannot express, so a comparison at an asymmetric pad
    measures the reference's limitation and not the kernel.

    Returns:
        ``[N, C, 1, Q]``.
    """
    result = depthwise_conv1d_implicit_gemm_torch_ref(
        img,
        filt,
        padding=padding,
        stride=stride,
        rhs_dilation=rhs_dilation,
        lhs_dilation=lhs_dilation,
        feature_group_count=int(img.shape[1]),
        batch_group_count=batch_group_count,
    )
    return result["output"]


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the wrapped kernel, read off the object.

    Exposed so a test can assert the seam dispatches to the SUBSTRATE's member
    rather than to anything authored here -- which is how a WRAP is checkable
    rather than merely claimed -- and so a substitution shows up as a changed
    reading instead of as silence.
    """
    func = getattr(depthwise_conv1d_implicit_gemm, "func", None)
    target = func if func is not None else depthwise_conv1d_implicit_gemm
    return target.__module__, target.__qualname__
