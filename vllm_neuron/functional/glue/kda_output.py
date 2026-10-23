# SPDX-License-Identifier: Apache-2.0
"""KDA output: gated RMSNorm and ``o_proj`` in one NKI launch, at decode and at prefill.

The torch route, ``Glm5NextKDAAttention._gated_output``, computes, all in fp32::

    shaped = core.view(T, heads, kdim)
    shaped = shaped * rsqrt(shaped.pow(2).mean(-1) + eps) * o_norm * sigmoid(out_gate)
    attn_out = shaped.view(T, W) @ o_proj.float().t()          # [T, H]

then reduces ``attn_out`` over the TP group and casts it to the activations' dtype.
The kernels return the same fp32 ``attn_out`` (the reduction and the cast stay at the
call site, in that order), without the per-step fp32 copy and transpose of ``o_proj``.
A caller whose reduction rounds the partial to bf16 first (its wire dtype) may ask for
bf16: the prefill kernel then rounds each fp32 sum once, as it leaves PSUM, and stores
half the bytes; every other route casts its fp32 result.

Method (both kernels). The elementwise steps run with tokens on the partitions, in the
torch expression's order, in fp32. The projection runs on the tensor engine as two bf16
passes accumulated in fp32, ``g = hi + lo`` with ``hi = bf16(g)``,
``lo = bf16(g - hi)``: every product is exact in fp32 and ``g - hi - lo`` is at most
2^-17 of ``|g|`` (``g - hi`` is below 2^-8 of ``g``'s leading power of two, and
rounding it to bf16 leaves at most 2^-9 of that). ``o_proj^T`` is loaded by DMA
transposes (or, with
``DMA_TRANSPOSE=False``, a plain DMA and tensor-engine transposes). An fp32 ``o_proj``
(the CPU fixtures' dtype) takes one fp32 pass, with tensor-engine transposes; the
prefill kernel serves bf16 ``o_proj`` only.

The two kernels split the work over the LNC2 pair differently:

* :func:`kda_output_kernel` (``T <= KDA_OUTPUT_MAX_TOKENS``): each program forms the
  (tiny) gated rows and its half of the ``H`` output columns, from its half of
  ``o_proj``; no exchange is needed.
* :func:`kda_output_prefill_kernel` (``T`` a multiple of 128 per program, above that
  bound): each program takes a contiguous half of the rows, all ``H`` columns of them,
  and the whole ``o_proj^T``. At ``T = 1024`` the 16 MiB fp32 output (8 MiB in bf16)
  dominates the call, so each program loads ``o_proj^T`` one 512-column bank at a time
  and writes each ``[128, 512]`` output block as soon as it is formed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron import envs
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.functional.glue import glue_selected
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Largest token count the decode kernel serves; larger calls take the prefill kernel.
KDA_OUTPUT_MAX_TOKENS = 128

#: The switch (registered in :mod:`vllm_neuron.envs`): ``1`` loads ``o_proj^T`` by DMA
#: transpose, ``0`` by a plain DMA and tensor-engine transposes.
DMA_TRANSPOSE_ENV = "VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE"

_P = 128
_BANK = 512  # fp32 columns per PSUM bank
#: PSUM banks per output store in the prefill kernel: each partition then writes one
#: contiguous 4 x 512 fp32 = 8 KiB run per store instead of four 2 KiB runs (4 KiB
#: runs when the kernel stores bf16).
_STORE_BANKS = 4
_DGE_ROWS = 16

__all__ = [
    "KDA_OUTPUT_MAX_TOKENS",
    "dispatch_counters",
    "kda_gated_projection",
    "kda_gated_projection_torch",
    "kda_output_kernel",
    "kda_output_prefill_kernel",
    "reset_dispatch_counters",
]


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _gated_rows(c, gate, gain, eps):
    """``c * rsqrt(mean(c^2) + eps) * gain * sigmoid(gate)``, fp32, per ``kdim`` group.

    Args:
        c: ``[rows, groups, kdim]`` fp32 SBUF tile of the recurrence output.
        gate: ``[rows, groups * kdim]`` fp32 SBUF tile of the output gate.
        gain: ``[rows, kdim]`` SBUF tile of the norm gain (every row the same).
        eps: the norm epsilon.

    Returns:
        ``[rows, groups, kdim]`` fp32 SBUF tile, in the torch expression's order.
    """
    rows, groups, kdim = c.shape
    sq = nl.ndarray((rows, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=c, data2=c, op=nl.multiply)
    rstd = _tile(rows, groups)
    nisa.tensor_reduce(dst=rstd, op=nl.add, data=sq, axis=(2,))
    # mean = sum * (1/kdim): exact for a power-of-two kdim (128 served; admitted only).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.multiply, operand0=1.0 / float(kdim),
                       op1=nl.add, operand1=float(eps))
    # rsqrt on GpSimd, the higher-precision engine for it (see functional/norm.py).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    shaped = nl.ndarray((rows, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=shaped, data1=c, data2=rstd.expand_dim(2).broadcast(2, kdim),
                       op=nl.multiply)
    nisa.tensor_tensor(dst=shaped, data1=shaped,
                       data2=gain.expand_dim(1).broadcast(1, groups), op=nl.multiply)
    sig = nl.ndarray((rows, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sig.reshape((rows, groups * kdim)), op=nl.sigmoid, data=gate)
    nisa.tensor_tensor(dst=shaped, data1=shaped, data2=sig, op=nl.multiply)
    return shaped


def _split_bf16(flat):
    """``(hi, lo)`` bf16 tiles with ``hi = bf16(flat)``, ``lo = bf16(flat - hi)``."""
    rows, cols = flat.shape
    hi = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hi, src=flat)
    rem = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rem, data1=flat, data2=hi, op=nl.subtract)
    lo = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lo, src=rem)
    return hi, lo


def _load_o_proj_t(ot, o_proj, h0, use_dma):
    """``ot[j, h] = o_proj[h0 + h, j]`` for the ``hp = ot.shape[1]`` rows from ``h0``.

    ``use_dma``: one DMA transpose (bf16 only); else a plain DMA of the natural rows
    and tensor-engine transposes, eight bf16 (four fp32) blocks per PSUM bank.
    """
    width, hp = ot.shape
    if use_dma:
        # src[a, b, j] = o_proj[h0 + 16 b + a, j] -> dst[j, b, a]: one DMA.
        nisa.dma_transpose(
            dst=ot.reshape((width, hp // _DGE_ROWS, _DGE_ROWS)),
            src=o_proj.ap(pattern=[[width, _DGE_ROWS], [_DGE_ROWS * width, hp // _DGE_ROWS],
                                   [1, width]], offset=h0 * width))
        return
    nblk = hp // _P
    natural = nl.ndarray((_P, nblk, width), dtype=o_proj.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=natural, src=o_proj.ap(
        pattern=[[width, _P], [_P * width, nblk], [1, width]], offset=h0 * width))
    per = 4 if o_proj.dtype == nl.float32 else 8  # 128-column transposes per 2 KiB bank
    for g0 in range(0, nblk, per):
        gn = min(per, nblk - g0)
        flipped = nl.ndarray((width, gn * _P), dtype=o_proj.dtype, buffer=nl.psum)
        for j in range(gn):
            nisa.nc_transpose(dst=flipped[:, j * _P:(j + 1) * _P],
                              data=natural[:, g0 + j, :])
        nisa.tensor_copy(dst=ot[:, g0 * _P:(g0 + gn) * _P], src=flipped)


@nki.jit
def kda_output_kernel(core, out_gate, o_norm, o_proj, HEADS: int = 1, EPS: float = 1e-6,
                      DMA_TRANSPOSE: bool = True):
    """``attn_out [T, H]`` fp32, before the TP reduction (decode shapes, ``T <= 128``).

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
    flat = _gated_rows(c, gate, gain, EPS).reshape((tokens, width))

    fp32_proj = o_proj.dtype == nl.float32
    if fp32_proj:
        # g^T [W, T] in fp32, for one fp32 pass.
        g_ps = nl.ndarray((width, tokens), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_ps, data=flat)
        gt = nl.ndarray((width, 1, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gt.reshape((width, tokens)), src=g_ps)
    else:
        # g = hi + lo in bf16, then g^T [W, 2, T].
        hi, lo = _split_bf16(flat)
        # Each bf16 PSUM slot padded to an even width: a slot must start 4-byte aligned.
        gt_ps = nl.ndarray((width, 2, tokens + tokens % 2), dtype=nl.bfloat16,
                           buffer=nl.psum)
        nisa.nc_transpose(dst=gt_ps[:, 0, 0:tokens], data=hi)
        nisa.nc_transpose(dst=gt_ps[:, 1, 0:tokens], data=lo)
        gt = nl.ndarray((width, 2, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gt, src=gt_ps[:, :, 0:tokens])

    # ---- o_proj^T [W, hp] for this program's output columns. --------------- #
    ot = nl.ndarray((width, hp), dtype=o_proj.dtype, buffer=nl.sbuf)
    _load_o_proj_t(ot, o_proj, h0, DMA_TRANSPOSE and not fp32_proj)

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


@nki.jit
def kda_output_prefill_kernel(core, out_gate, o_norm, o_proj, HEADS: int = 1,
                              EPS: float = 1e-6, DMA_TRANSPOSE: bool = True,
                              BF16_OUT: bool = False):
    """``attn_out [T, H]`` fp32 (bf16 with ``BF16_OUT``), before the TP reduction, at prefill.

    Each program takes ``T / programs`` contiguous rows in 128-row tiles: the gated
    RMSNorm of all its tiles at once (tokens on the partitions, the tiles side by side
    on the free axis), and per tile ``g^T`` as ``hi``/``lo`` bf16 by tensor-engine
    transposes. Then, per group of ``_STORE_BANKS`` 512-column PSUM banks of the output:
    that group's columns of ``o_proj^T`` are loaded, and each tile takes the two passes
    per bank and writes its ``[128, 512 * _STORE_BANKS]`` block in one store. The output
    stream bounds the call; it starts after one group of ``o_proj^T`` and not after all of
    it, and each partition's store is one contiguous run of the group's columns (8 KiB
    for four banks), not one 2 KiB run per bank (with one bank per store, a profile
    reads 2 KiB DMA packets keeping all 16 DMA engines busy).

    Args:
        core: ``[T, W]`` fp32 recurrence output, ``W = HEADS * kdim <= 128``, ``T`` a
            multiple of ``128 * programs``.
        out_gate: ``[T, W]`` fp32.
        o_norm: ``[kdim]`` gain (bf16 or fp32).
        o_proj: ``[H, W]`` bf16, this rank's row-parallel slice.
        DMA_TRANSPOSE: load ``o_proj^T`` by DMA transpose (else tensor engine).
        BF16_OUT: round each fp32 sum to bf16 in the copy that reads it out of PSUM, to
            nearest even, and store bf16 (else store the fp32 sums). That is the one
            rounding a bf16 cast of the fp32 output makes, so the bits are the same.
    """
    tokens, width = core.shape
    hidden, w2 = o_proj.shape
    kdim = width // HEADS
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(tokens % (_P * programs) == 0, "T splits into 128-row tiles per program")
    kernel_assert(w2 == width and width <= _P and kdim * HEADS == width, "W <= 128")
    kernel_assert(hidden % _BANK == 0, "H in 512-column PSUM banks")
    kernel_assert(o_proj.dtype == nl.bfloat16, "bf16 o_proj")
    rows = tokens // programs
    r0 = program * rows
    tiles = rows // _P
    out_dtype = nl.bfloat16 if BF16_OUT else nl.float32
    out = nl.ndarray((tokens, hidden), dtype=out_dtype, buffer=nl.shared_hbm)

    # ---- Gated RMSNorm of this program's rows: c[p, i] = core[r0 + 128 i + p]. #
    c = nl.ndarray((_P, tiles * HEADS, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c.reshape((_P, tiles, width)),
                  src=core.ap(pattern=[[width, _P], [_P * width, tiles], [1, width]],
                              offset=r0 * width))
    gate = nl.ndarray((_P, tiles, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gate,
                  src=out_gate.ap(pattern=[[width, _P], [_P * width, tiles], [1, width]],
                                  offset=r0 * width))
    gain = nl.ndarray((_P, kdim), dtype=o_norm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gain, src=o_norm.ap(pattern=[[0, _P], [1, kdim]], offset=0))
    shaped = _gated_rows(c, gate.reshape((_P, tiles * width)), gain, EPS)
    hi, lo = _split_bf16(shaped.reshape((_P, tiles * width)))

    # g^T [W, tile, 2, 128]: each tile's hi and lo passes' stationaries.
    gt = nl.ndarray((width, tiles, 2, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(tiles):
        gt_ps = nl.ndarray((width, 2, _P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=gt_ps[:, 0, :], data=hi[:, i * width:(i + 1) * width])
        nisa.nc_transpose(dst=gt_ps[:, 1, :], data=lo[:, i * width:(i + 1) * width])
        nisa.tensor_copy(dst=gt[:, i, :, :], src=gt_ps)

    banks = hidden // _BANK
    group = math.gcd(banks, _STORE_BANKS)
    for g0 in range(0, banks, group):
        c0 = g0 * _BANK
        # o_proj^T [W, 512 * group] for this group's output columns.
        ot = nl.ndarray((width, group * _BANK), dtype=o_proj.dtype, buffer=nl.sbuf)
        _load_o_proj_t(ot, o_proj, c0, DMA_TRANSPOSE)
        for i in range(tiles):
            # Each bank's accumulator takes the same two passes in the same order as one
            # bank per store; only the store grouping changes, so every output bit is kept.
            block = nl.ndarray((_P, group * _BANK), dtype=out_dtype, buffer=nl.sbuf)
            for j in range(group):
                acc = nl.ndarray((_P, _BANK), dtype=nl.float32, buffer=nl.psum)
                for part in range(2):
                    nisa.nc_matmul(dst=acc, stationary=gt[:, i, part, :],
                                   moving=ot[:, j * _BANK:(j + 1) * _BANK],
                                   accumulate=(part > 0))
                # The PSUM reads alternate Vector and Scalar, the two engines that read PSUM;
                # a bf16 block takes its one rounding here.
                nisa.tensor_copy(dst=block[:, j * _BANK:(j + 1) * _BANK], src=acc,
                                 engine=nisa.engine.vector
                                 if ((g0 + j) * tiles + i) % 2 == 0 else nisa.engine.scalar)
            nisa.dma_copy(dst=out.ap(pattern=[[hidden, _P], [1, group * _BANK]],
                                     offset=(r0 + i * _P) * hidden + c0), src=block)
    return out


_KERNELS = {1: wrap_nki(kda_output_kernel), 2: wrap_nki(kda_output_kernel)[2]}
_PREFILL_KERNELS = {1: wrap_nki(kda_output_prefill_kernel),
                    2: wrap_nki(kda_output_prefill_kernel)[2]}


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
    """2 programs (both LNC2 cores) under ``NEURON_LOGICAL_NC_CONFIG=2``, 1 when unset or 1.

    :func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair` refuses a setting other
    than unset, 1 or 2 (``LaunchGridError``).
    """
    return 2 if lnc_pair() else 1


def _serves_prefill(tokens: int, o_proj: Tensor) -> bool:
    """True when the prefill kernel takes a ``tokens``-row call (above the decode bound)."""
    return (tokens > KDA_OUTPUT_MAX_TOKENS and tokens % (_P * launch_programs()) == 0
            and o_proj.dtype == torch.bfloat16 and int(o_proj.shape[0]) % _BANK == 0)


def kda_gated_projection_admits(core: Tensor, out_gate: Tensor, attn,
                                phase: str | None = None) -> bool:
    """True when :func:`kda_gated_projection` serves this call on a kernel.

    Args:
        core: ``[T, W]`` fp32 attention core, ``W = heads * K``.
        out_gate: ``[T, W]`` fp32 output gate (before its sigmoid).
        attn: the KDA attention module (``o_norm_weight [K]``, ``o_proj_weight [H, W]``).
        phase: the step's ``"prefill"`` or ``"decode"``; None when it is not known,
            and then only a rule without a phase selects the kernel.

    Either kernel serves ``W <= 128``, ``K`` a power of two, bf16 or fp32 gain, on an
    NKI device or the simulator, when ``VLLM_NEURON_GLUE_FUSED`` selects ``kda_output``
    for this row count and phase. The decode kernel takes ``1 <= T <=
    KDA_OUTPUT_MAX_TOKENS`` with bf16 or fp32 ``o_proj`` and ``H`` in 128-blocks per
    program; the prefill kernel takes a larger ``T`` that splits into 128-row tiles per
    program, with bf16 ``o_proj`` and ``H`` in 512-column banks. Anything else is
    counted as declined.
    """
    o_proj, gain = attn.o_proj_weight, attn.o_norm_weight
    heads, kdim = int(attn.num_kv_heads_per_rank), int(attn.head_dim)
    width = heads * kdim
    tokens = int(core.shape[0]) if core.dim() == 2 else 0
    ok = (
        core.dim() == 2 and out_gate.dim() == 2 and o_proj.dim() == 2
        and glue_selected("kda_output", tokens, phase)
        and core.dtype == torch.float32 and out_gate.dtype == torch.float32
        and o_proj.dtype in (torch.bfloat16, torch.float32)
        and gain.dtype in (torch.bfloat16, torch.float32)
        and tuple(gain.shape) == (kdim,) and kdim & (kdim - 1) == 0
        and tuple(core.shape) == tuple(out_gate.shape) == (tokens, width)
        and width <= _P and int(o_proj.shape[1]) == width
        and ((1 <= tokens <= KDA_OUTPUT_MAX_TOKENS
              and int(o_proj.shape[0]) % (_P * launch_programs()) == 0)
             or _serves_prefill(tokens, o_proj))
        and can_run_kernel(core)
    )
    if not ok:
        _count_declined()
    return ok


def kda_gated_projection_torch(core: Tensor, out_gate: Tensor, attn) -> Tensor:
    """The torch route's ``_gated_output`` up to the reduction, verbatim: ``[T, H]`` fp32."""
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
                         phase: str | None = None,
                         out_dtype: torch.dtype | None = None) -> Tensor:
    """The gated RMSNorm of ``core`` and its output projection, on a kernel or in torch.

    Args:
        core: ``[T, W]`` fp32 attention core.
        out_gate: ``[T, W]`` fp32 output gate.
        attn: the KDA attention module.
        phase: the step's ``"prefill"`` or ``"decode"``, for ``VLLM_NEURON_GLUE_FUSED``.
        out_dtype: None (the default) for the route's fp32 result as it is, or
            ``torch.bfloat16`` for a caller that rounds the partial to bf16 next (its
            wire dtype). The prefill kernel then rounds in its store (``BF16_OUT``);
            the decode kernel and the torch route cast their fp32 result. Either way
            the bits are those of a bf16 cast of the fp32 partial.

    Returns:
        ``attn_out [T, H]``, this rank's partial sum before the TP reduction: fp32, or
        bf16 with ``out_dtype=torch.bfloat16``. When :func:`kda_gated_projection_admits`
        says no, it is the torch expression (:func:`kda_gated_projection_torch`).

    Raises:
        ValueError: ``out_dtype`` is neither None nor bf16.
    """
    if out_dtype not in (None, torch.bfloat16):
        raise ValueError(
            f"kda_gated_projection: out_dtype is None (the route's fp32 result) or "
            f"torch.bfloat16; got {out_dtype}"
        )
    rounds = out_dtype is torch.bfloat16
    if not kda_gated_projection_admits(core, out_gate, attn, phase):
        return _stored(kda_gated_projection_torch(core, out_gate, attn), rounds)
    _count_nki_dispatch()
    operands = {
        "core": core.contiguous(), "out_gate": out_gate.contiguous(),
        "o_norm": attn.o_norm_weight.contiguous(), "o_proj": attn.o_proj_weight.contiguous(),
        "HEADS": int(attn.num_kv_heads_per_rank), "EPS": float(attn.rms_norm_eps),
        "DMA_TRANSPOSE": envs.VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE,
    }
    if int(core.shape[0]) <= KDA_OUTPUT_MAX_TOKENS:
        return _stored(_KERNELS[launch_programs()](**operands), rounds)
    return _PREFILL_KERNELS[launch_programs()](**operands, BF16_OUT=rounds)


def _stored(partial: Tensor, rounds: bool) -> Tensor:
    """The fp32 ``partial`` as the caller stores it: cast to bf16 when it rounds, else
    the same tensor. A traced graph then holds no cast unless the caller rounds, and
    the partial's node keeps the name of the expression that made it."""
    return partial.to(torch.bfloat16) if rounds else partial
