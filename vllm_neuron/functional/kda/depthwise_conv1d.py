# SPDX-License-Identifier: Apache-2.0
"""KDA prefill depthwise conv1d: a thin wrap of ``nkilib``'s NKI kernel.

The KDA prefill path applies a per-channel convolution along the sequence axis
before the delta rule runs. ``nkilib`` already ships that kernel, so no numerics
are authored here::

    nkilib.experimental.conv.depthwise_conv1d.depthwise_conv1d_implicit_gemm

The torch reference this module exposes likewise delegates to the substrate's own
``depthwise_conv1d_implicit_gemm_torch_ref``, which ships beside the kernel.

What the wrap adds is argument handling the kernel makes easy to get wrong:
it derives ``feature_group_count`` from the input channel extent (the kernel
asserts ``feature_group_count == C`` but defaults it to ``1``), it checks every
geometry and option condition in one place, and it unwraps the reference's
``{"output": tensor}`` return so both paths hand back a plain tensor.

The channel count must divide :data:`LNC_SHARDS`. The substrate kernel divides
the channel extent by ``nl.num_programs()`` and indexes each shard from
``shard_id * C_per_shard``, so an odd ``C`` on a two-shard device drops channels
rather than failing. The NKI simulator does not enforce this and computes an odd
``C`` correctly, so the refusal here is conservatism about the documented device
requirement, not a reproduction of a simulator failure.

A refused geometry raises; it never routes to the torch reference.
"""

from __future__ import annotations

import logging

import torch
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

#: Logical Neuron Core shard count the substrate kernel shards its channel extent
#: over, from its own Notes ("Requires C to be divisible by NUM_SHARDS (2)") and
#: its ``C_per_shard = C // nl.num_programs()``.
LNC_SHARDS = 2

#: The one spatial extent the kernel's layout fixes: both the image and the
#: filter carry a singleton height axis, because this is a 1-D convolution
#: expressed in the substrate's 2-D ``[N, C, H, W]`` argument layout.
SINGLETON_H = 1

#: Zero padding on both sides of both axes, the only padding the substrate kernel
#: supports ("Only supports zero padding").
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
    refusal that names the offending extent is what a caller can act on.
    """


def output_width(
    width: int,
    kernel_size: int,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
) -> int:
    """The output extent ``Q``, from the substrate's own formula.

    ``Q = (W + W_pad_l + W_pad_r - S) // stride_w + 1``, stated once here so
    callers do not restate the arithmetic.

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

    Each message names what in the kernel needs the condition, so a refusal can
    be checked against the substrate.
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
            f"rather than fail. Refused here rather than routed to the torch "
            f"reference."
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


def can_run_depthwise_conv1d(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
    rhs_dilation: tuple[int, int] = UNIT_DILATION,
    lhs_dilation: tuple[int, int] = UNIT_DILATION,
    batch_group_count: int = 1,
) -> bool:
    """Is the NKI path available *and* admissible for this call?

    Two independent conditions:
    :func:`~vllm_neuron.utils.neuron_utils.can_run_kernel` answers whether a
    device or simulator exists, :func:`_require_admissible` whether the substrate
    kernel accepts these extents and options.

    Raises:
        KdaDepthwiseConv1dError: if the call is inadmissible. Inadmissible is not
            the same as unavailable, and it does not fall back.
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
    """Depthwise conv1d over the sequence axis.

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

    ``feature_group_count`` is deliberately not a parameter here. The substrate
    kernel asserts ``feature_group_count == C`` yet defaults it to ``1``, so every
    caller would otherwise have to restate the channel extent correctly or hit a
    trace-time failure. It is derived from ``img`` instead.
    """
    if not can_run_depthwise_conv1d(
        img, filt, padding, stride, rhs_dilation, lhs_dilation, batch_group_count
    ):
        _count_torch_fallback()
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

    _count_nki_dispatch()
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
    """``nkilib``'s own torch reference for this kernel. Never the shipped path.

    Delegates to ``depthwise_conv1d_implicit_gemm_torch_ref``, deriving
    ``feature_group_count`` as :func:`depthwise_conv1d` does and unwrapping the
    reference's ``{"output": tensor}`` return so both paths hand back a plain
    tensor.

    One asymmetry this wrap does not hide: the reference applies width padding
    through ``F.conv2d``, which pads both sides by the same amount, while the
    kernel pads left and right independently. The two therefore agree only for
    symmetric width padding, :data:`NO_PADDING` included; at an asymmetric pad a
    comparison measures the reference's limitation, not the kernel.

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

    Lets a caller check that this module dispatches to ``nkilib``'s member rather
    than to anything authored here, so a substitution shows up as a changed
    reading.
    """
    func = getattr(depthwise_conv1d_implicit_gemm, "func", None)
    target = func if func is not None else depthwise_conv1d_implicit_gemm
    return target.__module__, target.__qualname__
