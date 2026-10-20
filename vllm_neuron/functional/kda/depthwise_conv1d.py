# SPDX-License-Identifier: Apache-2.0
"""KDA prefill depthwise conv1d: the wrap around this package's NKI kernel.

The KDA prefill path applies a per-channel convolution along the sequence axis
before the delta rule runs. The kernel is authored here, in
:mod:`vllm_neuron.functional.kda.depthwise_conv1d_kernel`; this module checks
every geometry and option condition in one place, picks the launch grid, counts
which path ran, and carries the torch reference the kernel is tested against.

Under ``NEURON_LOGICAL_NC_CONFIG=2`` the kernel launches on :data:`LNC_SHARDS`
programs that split the output columns, at every admissible geometry; otherwise
it launches on one.

:func:`depthwise_conv1d_rows` is the same convolution, stride 1 and unpadded, of an
input given token-major in pieces: the left-context rows and :data:`ROW_BLOCKS`
channel blocks, as the KDA layer holds them, so the caller builds no image.

A refused geometry raises; it never routes to the torch reference.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from torch import Tensor

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.functional.kda.depthwise_conv1d_kernel import (
    PARTITION_MAX,
    TAP_SLOTS_MAX,
    depthwise_conv1d_kernel,
    depthwise_conv1d_rows_kernel,
    tap_slots,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Programs the kernel launches on under ``NEURON_LOGICAL_NC_CONFIG=2``, one per
#: physical core of the logical core; each takes a contiguous share of the output
#: columns.
LNC_SHARDS = 2

#: The one spatial extent the layout fixes: both the image and the filter carry a
#: singleton height axis, because this is a 1-D convolution expressed in a 2-D
#: ``[N, C, H, W]`` argument layout.
SINGLETON_H = 1

#: No padding on either axis, the default. Width padding may be any non-negative
#: pair, left and right independently; height padding must stay zero.
NO_PADDING = ((0, 0), (0, 0))

#: Unit stride on both axes. ``stride_h`` must be 1, while ``stride_w`` may be any
#: positive integer.
UNIT_STRIDE = (1, 1)

#: The only dilation the kernel computes, on both axes.
UNIT_DILATION = (1, 1)

#: Dtypes the kernel loads and stores; it accumulates in float32 whichever it is.
_KERNEL_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

#: Channel blocks :func:`depthwise_conv1d_rows` takes, in channel order: the KDA
#: layer's q, k and v streams, which its input projections write as three tensors.
ROW_BLOCKS = 3

__all__ = [
    "LNC_SHARDS",
    "NO_PADDING",
    "ROW_BLOCKS",
    "SINGLETON_H",
    "UNIT_DILATION",
    "UNIT_STRIDE",
    "KdaDepthwiseConv1dError",
    "can_run_depthwise_conv1d",
    "carried_rows",
    "depthwise_conv1d",
    "depthwise_conv1d_rows",
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
    """The output extent ``Q`` of a padded, strided convolution.

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


def _filter_problems(filt: Tensor, channels: int, dtype: torch.dtype) -> list[str]:
    """What is wrong with a 4-D ``filt`` for ``channels`` inputs of ``dtype``."""
    problems: list[str] = []
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
    if dtype != filt.dtype:
        problems.append(
            f"img dtype {dtype} and filter dtype {filt.dtype} differ; a call "
            f"carries one precision, the output's"
        )
    if dtype not in _KERNEL_DTYPES:
        problems.append(
            f"dtype {dtype} is not one of {_KERNEL_DTYPES}, the dtypes the "
            f"kernel loads and stores"
        )
    return problems


def _require_admissible(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]],
    stride: tuple[int, int],
    rhs_dilation: tuple[int, int],
    lhs_dilation: tuple[int, int],
    batch_group_count: int,
) -> None:
    """Every condition of the convolution's contract, checked in one place.

    The contract is what the kernel and the torch reference both compute; the
    kernel's SBUF capacity is checked separately, by :func:`_require_capacity`.
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
            f"this is a 1-D convolution in a 2-D argument layout"
        )
    problems += _filter_problems(filt, channels, img.dtype)
    if tuple(padding[0]) != (0, 0):
        problems.append(
            f"height padding {padding[0]} must be (0, 0): the height axis is a "
            f"singleton, so height padding has no column to pad"
        )
    if min(padding[1]) < 0:
        problems.append(f"width padding {padding[1]} must be non-negative")
    if stride[0] != 1:
        problems.append(f"stride_h={stride[0]} must be 1")
    if stride[1] <= 0:
        problems.append(f"stride_w={stride[1]} must be positive")
    if tuple(rhs_dilation) != UNIT_DILATION:
        problems.append(
            f"rhs_dilation={tuple(rhs_dilation)} must be {UNIT_DILATION}"
        )
    if tuple(lhs_dilation) != UNIT_DILATION:
        problems.append(
            f"lhs_dilation={tuple(lhs_dilation)} must be {UNIT_DILATION}"
        )
    if batch_group_count != 1:
        problems.append(f"batch_group_count={batch_group_count} must be 1")

    if problems:
        raise KdaDepthwiseConv1dError(
            "kda depthwise conv1d refuses this call: " + "; ".join(problems)
        )

    # Raises on its own account if the padded width cannot hold the kernel.
    output_width(int(img.shape[3]), int(filt.shape[3]), padding, stride)


def _require_capacity(channels: int, taps: int) -> None:
    """The kernel holds every channel tile's taps in one SBUF tile; refuse more."""
    slots = tap_slots(channels, taps)
    if slots > TAP_SLOTS_MAX:
        raise KdaDepthwiseConv1dError(
            f"kda depthwise conv1d refuses this call: C={channels} channels "
            f"of S={taps} taps need {slots} float32 tap slots per "
            f"partition, more than the kernel's weight tile holds "
            f"(TAP_SLOTS_MAX={TAP_SLOTS_MAX})"
        )


class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the kernel launch, ``torch_fallback``
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
    device or simulator exists; :func:`_require_admissible` and
    :func:`_require_capacity` whether the kernel accepts these extents and
    options.

    Raises:
        KdaDepthwiseConv1dError: if the call is inadmissible. Inadmissible is not
            the same as unavailable, and it does not fall back.
    """
    _require_admissible(
        img, filt, padding, stride, rhs_dilation, lhs_dilation, batch_group_count
    )
    _require_capacity(int(img.shape[1]), int(filt.shape[3]))
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
            width padding must be zero or positive on each side. Defaults to
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

    There is no ``feature_group_count`` parameter: a depthwise convolution has one
    group per channel, read off ``img``.
    """
    if not can_run_depthwise_conv1d(
        img, filt, padding, stride, rhs_dilation, lhs_dilation, batch_group_count
    ):
        _count_torch_fallback()
        logger.debug(
            "depthwise_conv1d: NKI route unavailable, using the torch reference "
            "(reference only, not the shipped path)"
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
    call = wrap_nki(depthwise_conv1d_kernel)
    if lnc_pair():
        call = call[LNC_SHARDS]
    pad_left, pad_right = padding[1]
    return call(img, filt, int(pad_left), int(pad_right), int(stride[1]))


def _rows_image(history: Tensor, blocks: tuple[Tensor, ...]) -> Tensor:
    """The ``[1, C, 1, S - 1 + T]`` image the token-major pieces stand for."""
    padded = torch.cat((history, torch.cat(blocks, dim=-1)), dim=0)
    return padded.t().contiguous().reshape(1, padded.shape[1], 1, padded.shape[0])


def _require_rows_admissible(
    history: Tensor, blocks: tuple[Tensor, ...], filt: Tensor
) -> None:
    """Every condition of the token-major form, checked in one place.

    The convolution is :func:`depthwise_conv1d`'s at stride 1 with no padding, so
    the filter's conditions are that form's. The kernel turns ``PARTITION_MAX``
    channels of a block onto the partitions at a time, so every block is a whole
    number of channel tiles.
    """
    problems: list[str] = []
    if len(blocks) != ROW_BLOCKS:
        raise KdaDepthwiseConv1dError(
            f"kda depthwise conv1d refuses this call: {len(blocks)} channel blocks, "
            f"the token-major form takes {ROW_BLOCKS}"
        )
    if history.dim() != 2 or any(block.dim() != 2 for block in blocks):
        raise KdaDepthwiseConv1dError(
            "kda depthwise conv1d refuses this call: history and blocks must be 2-D "
            f"[rows, channels], got {tuple(history.shape)} and "
            f"{[tuple(block.shape) for block in blocks]}"
        )
    if filt.dim() != 4:
        raise KdaDepthwiseConv1dError(
            f"kda depthwise conv1d refuses this call: filter must be 4-D "
            f"[C, 1, 1, S], got {filt.dim()}-D {tuple(filt.shape)}"
        )
    tokens = int(blocks[0].shape[0])
    if tokens < 1:
        problems.append("the blocks hold no token")
    for index, block in enumerate(blocks):
        if int(block.shape[0]) != tokens:
            problems.append(
                f"block {index} has {int(block.shape[0])} tokens, block 0 {tokens}"
            )
        if int(block.shape[1]) % PARTITION_MAX:
            problems.append(
                f"block {index} has {int(block.shape[1])} channels, not a multiple "
                f"of {PARTITION_MAX}"
            )
        if block.dtype != history.dtype:
            problems.append(
                f"block {index} dtype {block.dtype} differs from the history's "
                f"{history.dtype}"
            )
    channels = sum(int(block.shape[1]) for block in blocks)
    if int(history.shape[1]) != channels:
        problems.append(
            f"history has {int(history.shape[1])} channels, the blocks {channels}"
        )
    if int(history.shape[0]) != int(filt.shape[3]) - 1:
        problems.append(
            f"history has {int(history.shape[0])} rows, the S={int(filt.shape[3])} "
            f"taps need S - 1"
        )
    problems += _filter_problems(filt, channels, history.dtype)
    if problems:
        raise KdaDepthwiseConv1dError(
            "kda depthwise conv1d refuses this call: " + "; ".join(problems)
        )
    _require_capacity(channels, int(filt.shape[3]))


def depthwise_conv1d_rows(
    history: Tensor, blocks: tuple[Tensor, ...], filt: Tensor
) -> Tensor:
    """Stride-1 unpadded depthwise conv1d of a token-major input given in pieces.

    The result is :func:`depthwise_conv1d`'s on the image
    ``cat(history, cat(blocks, -1)).t()``, bit for bit: the kernel puts the same
    values in the same window and runs the same passes on them. The image is never
    built.

    Args:
        history: ``[S - 1, C]``, the input rows before the first output's.
        blocks: :data:`ROW_BLOCKS` tensors ``[T, C_i]`` in channel order, each
            ``C_i`` a multiple of ``PARTITION_MAX``, at the history's dtype.
        filt: ``[C, 1, 1, S]`` depthwise taps, one filter per channel.

    Returns:
        ``[1, C, 1, T]`` at the input dtype.

    Raises:
        KdaDepthwiseConv1dError: on an inadmissible geometry or dtype.
    """
    blocks = tuple(blocks)
    _require_rows_admissible(history, blocks, filt)
    if not can_run_kernel(history):
        _count_torch_fallback()
        logger.debug(
            "depthwise_conv1d_rows: NKI route unavailable, using the torch "
            "reference (reference only, not the shipped path)"
        )
        return depthwise_conv1d_torch_reference(_rows_image(history, blocks), filt)

    _count_nki_dispatch()
    call = wrap_nki(depthwise_conv1d_rows_kernel)
    if lnc_pair():
        call = call[LNC_SHARDS]
    return call(history, *blocks, filt)


def carried_rows(
    history: Tensor, blocks: tuple[Tensor, ...], real_length: Tensor
) -> Tensor:
    """Rows ``L .. L + S - 2`` of the input ``cat(history, cat(blocks, -1))``, unbuilt.

    The left context the next step convolves with: the last ``S - 1`` rows of a
    sequence whose real length in this step is ``L``. Row ``r`` of the input is
    ``history[r]`` below ``S - 1`` and token ``r - S + 1`` of the blocks above it,
    so a step of fewer than ``S - 1`` real rows keeps the older rows it still needs.

    Args:
        history: ``[S - 1, C]``, the input rows before the first token's.
        blocks: ``[T, C_i]`` tensors in channel order, ``sum C_i = C``.
        real_length: int64 scalar ``L``, ``0 <= L <= T``.

    Returns:
        ``[S - 1, C]``, selected rows only: the values are the input's bits.
    """
    lead = int(history.shape[0])
    row = torch.arange(lead, device=history.device) + real_length
    from_tokens = torch.cat(
        [block.index_select(0, (row - lead).clamp(min=0)) for block in blocks], dim=-1
    )
    from_history = history.index_select(0, row.clamp(max=lead - 1))
    return torch.where((row < lead).unsqueeze(-1), from_history, from_tokens)


def depthwise_conv1d_torch_reference(
    img: Tensor,
    filt: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]] = NO_PADDING,
    stride: tuple[int, int] = UNIT_STRIDE,
    rhs_dilation: tuple[int, int] = UNIT_DILATION,
    lhs_dilation: tuple[int, int] = UNIT_DILATION,
    batch_group_count: int = 1,
) -> Tensor:
    """The convolution in torch, the comparator for the kernel. Never the shipped path.

    Pads the width left and right independently, then convolves with one group per
    channel at ``stride``. It computes in float32, as the kernel accumulates, and
    rounds once to the input dtype.

    Raises:
        KdaDepthwiseConv1dError: on a call outside the contract
            :func:`depthwise_conv1d` serves.

    Returns:
        ``[N, C, 1, Q]``.
    """
    _require_admissible(
        img, filt, padding, stride, rhs_dilation, lhs_dilation, batch_group_count
    )
    padded = F.pad(img.to(torch.float32), tuple(padding[1]))
    out = F.conv2d(
        padded, filt.to(torch.float32), stride=tuple(stride), groups=int(img.shape[1])
    )
    return out.to(img.dtype)


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel :func:`depthwise_conv1d` launches.

    Read off the object, so a substitution shows up as a changed reading.
    """
    func = getattr(depthwise_conv1d_kernel, "func", None)
    target = func if func is not None else depthwise_conv1d_kernel
    return target.__module__, target.__qualname__
