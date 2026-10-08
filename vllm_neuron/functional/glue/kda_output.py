# SPDX-License-Identifier: Apache-2.0
"""KDA decode output: gated RMSNorm and ``o_proj`` in one NKI launch.

0a08ff4's ``Glm5NextKDAAttention._gated_output`` computes, all in fp32::

    shaped = core.view(T, heads, kdim)
    shaped = shaped * rsqrt(shaped.pow(2).mean(-1) + eps) * o_norm * sigmoid(out_gate)
    attn_out = shaped.view(T, W) @ o_proj.float().t()          # [T, H]

then reduces ``attn_out`` over the TP group and casts it to the activations' dtype.
The kernel returns the same fp32 ``attn_out`` (the reduction and the cast stay at the
call site, in that order), without the per-step fp32 copy and transpose of ``o_proj``.

Method. The elementwise steps run with tokens on the partitions, in the torch
expression's order, in fp32. The projection runs on the tensor engine as two bf16
passes accumulated in fp32, ``g = hi + lo`` with ``hi = bf16(g)``,
``lo = bf16(g - hi)``: every product is exact in fp32 and ``g - hi - lo`` is at most
2^-18 of ``|g|``. ``o_proj^T`` is loaded by one DMA transpose per program (or, with
``DMA_TRANSPOSE=False``, a plain DMA and tensor-engine transposes). An fp32 ``o_proj``
(the CPU fixtures' dtype) takes one fp32 pass, with tensor-engine transposes.

Both LNC2 cores: yes. Each program forms the (tiny) gated rows and its half of the
``H`` output columns, from its half of ``o_proj``; no exchange is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron import envs
from vllm_neuron.functional.glue import glue_selected
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Largest token count served.
KDA_OUTPUT_MAX_TOKENS = 128

#: The switch (registered in :mod:`vllm_neuron.envs`): ``1`` loads ``o_proj^T`` by DMA
#: transpose, ``0`` by a plain DMA and tensor-engine transposes.
DMA_TRANSPOSE_ENV = "VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE"

_P = 128
_BANK = 512
_DGE_ROWS = 16

__all__ = [
    "KDA_OUTPUT_MAX_TOKENS",
    "dispatch_counters",
    "kda_gated_projection",
    "kda_gated_projection_torch",
    "kda_output_kernel",
    "reset_dispatch_counters",
]


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


@nki.jit
def kda_output_kernel(core, out_gate, o_norm, o_proj, HEADS: int = 1, EPS: float = 1e-6,
                      DMA_TRANSPOSE: bool = True):
    """``attn_out [T, H]`` fp32, before the TP reduction.

    Args:
        core: ``[T, W]`` fp32 recurrence output, ``W = HEADS * kdim <= 128``.
        out_gate: ``[T, W]`` fp32.
        o_norm: ``[kdim]`` gain (bf16 or fp32).
        o_proj: ``[H, W]`` bf16 (served) or fp32, this rank's row-parallel slice.
    """
    tokens, width = core.shape
    hidden, w2 = o_proj.shape
    kdim = width // HEADS
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(1 <= tokens <= KDA_OUTPUT_MAX_TOKENS, "1 <= T <= 128")
    kernel_assert(w2 == width and width <= _P and kdim * HEADS == width, "W <= 128")
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(hidden % (_P * programs) == 0, "H splits into 128-blocks per program")
    hp = hidden // programs
    h0 = program * hp
    out = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Gated RMSNorm, tokens on the partitions, fp32. -------------------- #
    c = nl.ndarray((tokens, HEADS, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c.reshape((tokens, width)), src=core)
    gate = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gate, src=out_gate)
    gain = nl.ndarray((tokens, kdim), dtype=o_norm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gain, src=o_norm.ap(pattern=[[0, tokens], [1, kdim]], offset=0))
    sq = nl.ndarray((tokens, HEADS, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=c, data2=c, op=nl.multiply)
    rstd = _tile(tokens, HEADS)
    nisa.tensor_reduce(dst=rstd, op=nl.add, data=sq, axis=(2,))
    # mean = sum * (1/kdim): exact for a power-of-two kdim (128 served; admitted only).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.multiply, operand0=1.0 / float(kdim),
                       op1=nl.add, operand1=float(EPS))
    # rsqrt on GpSimd, the higher-precision engine for it (see functional/norm.py).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    shaped = nl.ndarray((tokens, HEADS, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=shaped, data1=c, data2=rstd.expand_dim(2).broadcast(2, kdim),
                       op=nl.multiply)
    nisa.tensor_tensor(dst=shaped, data1=shaped,
                       data2=gain.expand_dim(1).broadcast(1, HEADS), op=nl.multiply)
    sig = nl.ndarray((tokens, HEADS, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sig.reshape((tokens, width)), op=nl.sigmoid, data=gate)
    nisa.tensor_tensor(dst=shaped, data1=shaped, data2=sig, op=nl.multiply)
    flat = shaped.reshape((tokens, width))

    fp32_proj = o_proj.dtype == nl.float32
    if fp32_proj:
        # g^T [W, T] in fp32, for one fp32 pass.
        g_ps = nl.ndarray((width, tokens), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_ps, data=flat)
        gt = nl.ndarray((width, 1, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gt.reshape((width, tokens)), src=g_ps)
    else:
        # g = hi + lo in bf16, then g^T [W, 2, T].
        hi = nl.ndarray((tokens, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hi, src=flat)
        rem = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=rem, data1=flat, data2=hi, op=nl.subtract)
        lo = nl.ndarray((tokens, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lo, src=rem)
        # Each bf16 PSUM slot padded to an even width: a slot must start 4-byte aligned.
        gt_ps = nl.ndarray((width, 2, tokens + tokens % 2), dtype=nl.bfloat16,
                           buffer=nl.psum)
        nisa.nc_transpose(dst=gt_ps[:, 0, 0:tokens], data=hi)
        nisa.nc_transpose(dst=gt_ps[:, 1, 0:tokens], data=lo)
        gt = nl.ndarray((width, 2, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gt, src=gt_ps[:, :, 0:tokens])

    # ---- o_proj^T [W, hp] for this program's output columns. --------------- #
    ot = nl.ndarray((width, hp), dtype=o_proj.dtype, buffer=nl.sbuf)
    if DMA_TRANSPOSE and not fp32_proj:
        # src[a, b, j] = o_proj[h0 + 16 b + a, j] -> dst[j, b, a]: one DMA.
        nisa.dma_transpose(
            dst=ot.reshape((width, hp // _DGE_ROWS, _DGE_ROWS)),
            src=o_proj.ap(pattern=[[width, _DGE_ROWS], [_DGE_ROWS * width, hp // _DGE_ROWS],
                                   [1, width]], offset=h0 * width))
    else:
        nblk = hp // _P
        natural = nl.ndarray((_P, nblk, width), dtype=o_proj.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=natural, src=o_proj.ap(
            pattern=[[width, _P], [_P * width, nblk], [1, width]], offset=h0 * width))
        per = 4 if fp32_proj else 8  # 128-column transposes per 2 KiB PSUM bank
        for g0 in range(0, nblk, per):
            gn = min(per, nblk - g0)
            flipped = nl.ndarray((width, gn * _P), dtype=o_proj.dtype, buffer=nl.psum)
            for j in range(gn):
                nisa.nc_transpose(dst=flipped[:, j * _P:(j + 1) * _P],
                                  data=natural[:, g0 + j, :])
            nisa.tensor_copy(dst=ot[:, g0 * _P:(g0 + gn) * _P], src=flipped)

    # ---- attn_out = g @ o_proj^T: two bf16 passes (one fp32) per PSUM bank. -- #
    passes = 1 if fp32_proj else 2
    rows = nl.ndarray((tokens, hp), dtype=nl.float32, buffer=nl.sbuf)
    for c0 in range(0, hp, _BANK):
        cn = min(_BANK, hp - c0)
        acc = nl.ndarray((tokens, cn), dtype=nl.float32, buffer=nl.psum)
        for part in range(passes):
            nisa.nc_matmul(dst=acc, stationary=gt[:, part, :], moving=ot[:, c0:c0 + cn],
                           accumulate=(part > 0))
        nisa.tensor_copy(dst=rows[:, c0:c0 + cn], src=acc)
    nisa.dma_copy(dst=out.ap(pattern=[[hidden, tokens], [1, hp]], offset=h0), src=rows)
    return out


_KERNELS = {1: wrap_nki(kda_output_kernel), 2: wrap_nki(kda_output_kernel)[2]}


@dataclass
class _DispatchCounters:
    nki_dispatch: int = 0
    declined: int = 0


_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.declined = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, declined)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.declined


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_declined() -> None:
    _COUNTERS.declined += 1


def launch_programs() -> int:
    """2 programs (both LNC2 cores) under ``NEURON_LOGICAL_NC_CONFIG=2``, else 1."""
    return 2 if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" else 1


def kda_gated_projection_admits(core: Tensor, out_gate: Tensor, attn,
                                phase: str | None = None) -> bool:
    """True when :func:`kda_gated_projection` serves this call on the kernel.

    Args:
        core: ``[T, W]`` fp32 attention core, ``W = heads * K``.
        out_gate: ``[T, W]`` fp32 output gate (before its sigmoid).
        attn: the KDA attention module (``o_norm_weight [K]``, ``o_proj_weight [H, W]``).
        phase: the step's ``"prefill"`` or ``"decode"``; None when it is not known,
            and then only a rule without a phase selects the kernel.

    The kernel serves ``1 <= T <= KDA_OUTPUT_MAX_TOKENS``, ``W <= 128``, ``K`` a power of
    two, bf16 or fp32 ``o_proj`` and gain, ``H`` in 128-blocks per program, on an NKI
    device or the simulator, when ``VLLM_NEURON_GLUE_FUSED`` selects ``kda_output`` for
    this row count and phase. Anything else is counted as declined.
    """
    o_proj, gain = attn.o_proj_weight, attn.o_norm_weight
    heads, kdim = int(attn.num_kv_heads_per_rank), int(attn.head_dim)
    width = heads * kdim
    ok = (
        core.dim() == 2 and out_gate.dim() == 2 and o_proj.dim() == 2
        and glue_selected("kda_output", int(core.shape[0]), phase)
        and core.dtype == torch.float32 and out_gate.dtype == torch.float32
        and o_proj.dtype in (torch.bfloat16, torch.float32)
        and gain.dtype in (torch.bfloat16, torch.float32)
        and tuple(gain.shape) == (kdim,) and kdim & (kdim - 1) == 0
        and 1 <= int(core.shape[0]) <= KDA_OUTPUT_MAX_TOKENS
        and tuple(core.shape) == tuple(out_gate.shape) == (int(core.shape[0]), width)
        and width <= _P and int(o_proj.shape[1]) == width
        and int(o_proj.shape[0]) % (_P * launch_programs()) == 0
        and can_run_kernel(core)
    )
    if not ok:
        _count_declined()
    return ok


def kda_gated_projection_torch(core: Tensor, out_gate: Tensor, attn) -> Tensor:
    """0a08ff4's ``_gated_output`` up to the reduction, verbatim: ``[T, H]`` fp32."""
    tokens = int(core.shape[0])
    heads = int(attn.num_kv_heads_per_rank)
    kdim = int(attn.head_dim)
    width = heads * kdim
    shaped = core.reshape(tokens, heads, kdim)
    variance = shaped.pow(2).mean(dim=-1, keepdim=True)
    shaped = shaped * torch.rsqrt(variance + attn.rms_norm_eps)
    shaped = shaped * attn.o_norm_weight.to(torch.float32).reshape(1, 1, kdim)
    shaped = shaped * torch.sigmoid(out_gate.reshape(tokens, heads, kdim))
    return shaped.reshape(tokens, width) @ (attn.o_proj_weight.to(torch.float32).t())


def kda_gated_projection(core: Tensor, out_gate: Tensor, attn,
                         phase: str | None = None) -> Tensor:
    """The gated RMSNorm of ``core`` and its output projection, on the kernel or in torch.

    Args:
        core: ``[T, W]`` fp32 attention core.
        out_gate: ``[T, W]`` fp32 output gate.
        attn: the KDA attention module.
        phase: the step's ``"prefill"`` or ``"decode"``, for ``VLLM_NEURON_GLUE_FUSED``.

    Returns:
        ``attn_out [T, H]`` fp32, this rank's partial sum before the TP reduction. When
        :func:`kda_gated_projection_admits` says no, it is the torch expression
        (:func:`kda_gated_projection_torch`).
    """
    if not kda_gated_projection_admits(core, out_gate, attn, phase):
        return kda_gated_projection_torch(core, out_gate, attn)
    _count_nki_dispatch()
    return _KERNELS[launch_programs()](
        core=core.contiguous(), out_gate=out_gate.contiguous(),
        o_norm=attn.o_norm_weight.contiguous(), o_proj=attn.o_proj_weight.contiguous(),
        HEADS=int(attn.num_kv_heads_per_rank), EPS=float(attn.rms_norm_eps),
        DMA_TRANSPOSE=envs.VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE,
    )
