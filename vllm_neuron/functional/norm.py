# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm for decode rows: one NKI call instead of a chain of fp32 glue.

``out = (x * rsqrt(mean(x**2) + eps) * gain).to(x.dtype)``, computed in fp32,
exactly the formula of every decoder norm site at 5938748. Torch lowers that
formula to a handful of compiler ops over one ``[1, 4096]`` row (upcast, square,
mean, rsqrt, two multiplies, downcast), each with a fixed cost and each running
on a single partition. The kernel spreads the row over all 128 partitions
(``h = p * (H // 128) + j``), so every op touches ``H // 128`` elements per
partition, and the cross-partition sum is one Tensor Engine matmul by ones.

Rows ``1 <= T <= 128`` take the kernel; longer (prefill) inputs keep the torch
formula. The kernel runs on one physical core: at decode sizes it is a few
instructions long and an LNC2 split would only add a cross-core handoff.

Not wired into the decoder (measured, ``test/hardware/benchmark_dense_decode.py``):
as a standalone call it does not beat the compiler's lowering of the same formula
at decode rows. A bare NKI copy kernel of the same ``[1, 4096]`` row already costs
~5.5 us per call in a chain, about what the compiler spends on the whole norm, so
a norm only pays as a stage inside a neighbouring kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.utils.neuron_utils import can_run_kernel

PARTITIONS = 128
#: Longest row count the kernel takes; one row per partition in the stats tile.
MAX_KERNEL_ROWS = 128


@nki.jit
def rms_norm_kernel(x, gain, eps):
    """``[T, H]`` RMSNorm with an fp32 interior and ``x.dtype`` output.

    Args:
        x: ``[T, H]``, ``1 <= T <= 128``, ``H`` a multiple of 128.
        gain: ``[H]`` or ``[1, H]``.
        eps: the epsilon added to the mean square.
    """
    tokens, hidden = x.shape
    kernel_assert(0 < tokens <= MAX_KERNEL_ROWS, "rms_norm_kernel needs 1 <= T <= 128")
    kernel_assert(hidden % PARTITIONS == 0, "H must be a multiple of 128")
    per = hidden // PARTITIONS
    out = nl.ndarray((tokens, hidden), dtype=x.dtype, buffer=nl.shared_hbm)

    # x_sb[p, t, j] = x[t, p * per + j]; gain_sb[p, j] = gain[p * per + j].
    x_sb = nl.ndarray((PARTITIONS, tokens, per), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=x_sb,
        src=x.ap(pattern=[[per, PARTITIONS], [hidden, tokens], [1, per]]),
    )
    gain_sb = nl.ndarray((PARTITIONS, per), dtype=gain.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gain_sb, src=gain.ap(pattern=[[per, PARTITIONS], [1, per]])
    )

    # Per-partition partial sums of squares, fp32.
    squares = nl.ndarray((PARTITIONS, tokens, per), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=squares, data1=x_sb, data2=x_sb, op=nl.multiply)
    partial = nl.ndarray((PARTITIONS, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=partial, op=nl.add, data=squares, axis=(2,))

    # Sum over partitions and broadcast back: ones[128, 128]^T @ partial.
    ones = nl.ndarray((PARTITIONS, PARTITIONS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    total = nl.ndarray((PARTITIONS, tokens), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=total, stationary=ones, moving=partial, accumulate=False)

    # rstd = (total / H + eps) ** -0.5, per (partition, token).
    rstd = nl.ndarray((PARTITIONS, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=rstd,
        data=total,
        op0=nl.multiply,
        operand0=1.0 / hidden,
        op1=nl.add,
        operand1=float(eps),
    )
    # rsqrt on GpSimd: the higher-precision of its two engines, and the only
    # tensor_scalar form of it the ISA accepts (``power`` is not a valid op).
    nisa.tensor_scalar(
        dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd
    )

    # out = (x * rstd) * gain, cast to x.dtype on the last write.
    scaled = nl.ndarray((PARTITIONS, tokens, per), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled,
        data1=x_sb,
        data2=rstd.ap(pattern=[[tokens, PARTITIONS], [1, tokens], [0, per]]),
        op=nl.multiply,
    )
    result = nl.ndarray((PARTITIONS, tokens, per), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=result,
        data1=scaled,
        data2=gain_sb.ap(pattern=[[per, PARTITIONS], [0, tokens], [1, per]]),
        op=nl.multiply,
    )
    nisa.dma_copy(
        dst=out.ap(pattern=[[per, PARTITIONS], [hidden, tokens], [1, per]]),
        src=result,
    )
    return out


@dataclass
class _NormCounters:
    kernel: int = 0
    torch_path: int = 0


_COUNTERS = _NormCounters()


def reset_norm_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.kernel = 0
    _COUNTERS.torch_path = 0


def norm_dispatch_counters() -> tuple[int, int]:
    """``(kernel, torch_path)`` since the last reset."""
    return _COUNTERS.kernel, _COUNTERS.torch_path


@torch._dynamo.assume_constant_result
def _count_kernel() -> None:
    _COUNTERS.kernel += 1


@torch._dynamo.assume_constant_result
def _count_torch() -> None:
    _COUNTERS.torch_path += 1


def rms_norm_torch(hidden_states: Tensor, gain: Tensor, eps: float) -> Tensor:
    """The 5938748 formula, unchanged: fp32 interior, cast back at the end."""
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * gain.to(torch.float32)
    return normed.to(hidden_states.dtype)


def rms_norm(hidden_states: Tensor, gain: Tensor, eps: float) -> Tensor:
    """RMSNorm over the last axis of ``[T, H]``: the NKI kernel at decode rows.

    Same contract as :func:`rms_norm_torch`; the kernel route is taken for
    ``1 <= T <= 128`` 2-D inputs with ``H % 128 == 0`` when NKI can run.
    """
    if (
        hidden_states.dim() == 2
        and 0 < int(hidden_states.shape[0]) <= MAX_KERNEL_ROWS
        and int(hidden_states.shape[1]) % PARTITIONS == 0
        and gain.numel() == int(hidden_states.shape[1])
        and can_run_kernel(hidden_states)
    ):
        _count_kernel()
        return wrap_nki(rms_norm_kernel)(
            x=hidden_states, gain=gain.reshape(-1), eps=float(eps)
        )
    _count_torch()
    return rms_norm_torch(hidden_states, gain, eps)
