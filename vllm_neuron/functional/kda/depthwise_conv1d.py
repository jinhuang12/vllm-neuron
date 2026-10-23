# SPDX-License-Identifier: Apache-2.0
"""KDA prefill short convolution: the torch-callable wrap of its NKI kernel.

The KDA layer runs q, k and v through a causal depthwise conv1d of a few taps and
then SiLU. The rows before a chunk's first token come from the carried conv state,
so a chunk after the first convolves across the boundary. The call takes the three
streams token-major as the projections write them and returns the result
channel-major. A call the kernel cannot compute raises; there is no torch route.
"""

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.kda.depthwise_conv1d_kernel import (
    PARTITION_MAX,
    depthwise_conv1d_silu_kernel,
)
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


class KdaShortConvError(ValueError):
    """A call the short conv kernel does not compute."""


def _problems(history, blocks, weight):
    """What keeps the kernel from computing this call, one entry per defect."""
    if history.dim() != 2 or weight.dim() != 2 or any(b.dim() != 2 for b in blocks):
        shapes = [tuple(b.shape) for b in blocks]
        return [
            f"history {tuple(history.shape)}, blocks {shapes} and weight "
            f"{tuple(weight.shape)} must all be 2-D"
        ]
    problems = []
    tokens = blocks[0].shape[0]
    channels = sum(b.shape[1] for b in blocks)
    if tokens < 1:
        problems.append("the blocks hold no token")
    for name, block in zip("qkv", blocks):
        if block.shape[0] != tokens:
            problems.append(f"{name} has {block.shape[0]} tokens, q has {tokens}")
        if block.shape[1] % PARTITION_MAX:
            problems.append(
                f"{name} has {block.shape[1]} channels, not a multiple of "
                f"{PARTITION_MAX}"
            )
        if block.dtype != torch.bfloat16:
            problems.append(f"{name} is {block.dtype}, not torch.bfloat16")
    if history.dtype != torch.bfloat16:
        problems.append(f"history is {history.dtype}, not torch.bfloat16")
    taps = history.shape[0] + 1
    if taps < 2:
        problems.append("history has no row; the kernel's two tap chains need 2 taps")
    if tuple(weight.shape) != (channels, taps):
        problems.append(
            f"weight is {tuple(weight.shape)}; {channels} channels with "
            f"{history.shape[0]} history rows need ({channels}, {taps})"
        )
    if weight.dtype != torch.float32:
        problems.append(f"weight is {weight.dtype}, not torch.float32")
    if history.shape[1] != channels:
        problems.append(
            f"history has {history.shape[1]} channels, the blocks {channels}"
        )
    return problems


def depthwise_conv1d_silu(
    history: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """SiLU of the causal depthwise conv1d of ``[history; q | k | v]``, as ``[C, T]``.

    Args:
        history: ``[S - 1, C]`` bfloat16, the input rows before the first token:
            the carried conv state, or zeros for a fresh sequence.
        q, k, v: ``[T, C_i]`` bfloat16, each ``C_i`` a multiple of 128.
        weight: ``[C, S]`` float32, the ``S`` taps of each channel in q, k, v order.

    Returns:
        ``[C, T]`` bfloat16: ``out[c, t] = silu(sum_s weight[c, s] * x[t + s, c])``,
        with ``x`` the history rows followed by the tokens.

    Raises:
        KdaShortConvError: on a geometry or dtype the kernel does not compute, or
            when neither a Neuron device nor the NKI simulator is available.
    """
    blocks = (q, k, v)
    problems = _problems(history, blocks, weight)
    if problems:
        raise KdaShortConvError(
            "kda short conv refuses this call: " + "; ".join(problems)
        )
    if not can_run_kernel(history):
        raise KdaShortConvError(
            "kda short conv runs only as its NKI kernel: no Neuron device, and the "
            "NKI simulator is off (set NKI_SIMULATOR=1 in CPU mode)"
        )
    call = wrap_nki(depthwise_conv1d_silu_kernel)[get_logical_nc_config()]
    return call(history, q, k, v, weight)
