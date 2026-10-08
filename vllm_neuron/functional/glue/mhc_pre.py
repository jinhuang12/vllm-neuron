# SPDX-License-Identifier: Apache-2.0
"""mHC "pre" up to the Sinkhorn, and the pre-weighted collapse, in one NKI launch.

``Glm5NextHyperConnection.mhc_pre`` at 0a08ff4 runs this region as torch ops::

    flat   = residual.reshape(T, S*H).float()          # [T, K], K = S*H
    mixes  = flat @ fn.float().t()                       # [T, M], M = 2S + S*S
    mixes *= rsqrt(flat.square().sum(-1) / K + rms_eps)
    pre    = sigmoid(mixes[:, :S]   * scale[0] + base[:S]) + hc_eps
    post   = sigmoid(mixes[:, S:2S] * scale[1] + base[S:2S]) * post_mult
    comb   = softmax(mixes[:, 2S:].view(T, S, S) * scale[2] + base[2S:]) + hc_eps
    layer_input = (pre[..., None] * residual.float()).sum(1).to(residual.dtype)

and the profile charges all of it, with the per-step ``fn.float().t()`` (a
``[24, 16384]`` cast and transpose) and the fp32 streams copy, to compiler ops. This
kernel reads the bf16 streams and the bf16 ``fn`` as they are and returns
``(post, comb_start, layer_input)``; the Sinkhorn stays its own kernel. fp32 streams or
``fn`` (the CPU fixtures' dtypes) run the same steps on fp32 tiles.

Method, per program (``P`` programs split the hidden axis, ``Hp = H / P``):

1. ``fn``'s ``[M*S, Hp]`` slice and the streams' ``[T*S, Hp]`` slice are loaded with
   (row, stream) on the partitions -- every DMA row is one contiguous ``Hp`` run -- and
   transposed on the tensor engine, 128 hidden values at a time, into one SBUF tile
   ``mov[128 (h), Hp/128, S, M + T]``: for each contraction block ``(hb, s)`` the
   ``M`` columns of ``fn^T`` followed by the ``T`` columns of ``residual^T``.
2. ``acc[T, M + T] += mov[:, hb, s, M:]^T @ mov[:, hb, s, :]`` over the ``S * Hp/128``
   blocks: columns ``:M`` are this program's partial ``mixes``, the ``[T, T]`` rest is
   the streams' Gram block, whose diagonal is the square sum. bf16 x bf16 products are
   exact in the fp32 accumulator, so both differ from the torch values only by
   summation order.
3. Under two programs the two ``[T, M + T]`` partials are swapped (``sendrecv``) and
   added, so both programs hold the whole-``K`` values; ``a + b`` and ``b + a`` are the
   same fp32 number, so the two programs agree bit for bit.
4. The RMS scale, the gates and the softmax run with tokens on the partitions, in the
   torch expression's operation order. Program 0 stores ``post`` and ``comb_start``.
5. The collapse runs with hidden on the partitions, on the transposed streams already
   in ``mov``: ``pre`` is broadcast to all 128 partitions by a ones matmul (1.0 * a plus
   zeros, exact in fp32), each product is one fp32 multiply and the four streams are
   summed in fp32 and rounded once to bf16, as the torch expression does. The result
   goes back through the tensor engine to token rows and is stored as this program's
   hidden half.

Both LNC2 cores: yes. Each program does half the contraction (step 2), half the
transposes and half the collapse; steps 3-4 are ``T x (M + T)`` values, done on both.

``T`` is bounded by :data:`MHC_PRE_MAX_TOKENS` (the stationary ``residual^T`` block of
step 2 is at most 128 columns): a decode batch or a prefill chunk of at most 128 rows.
A larger call (the served 1024-row prefill chunk) keeps the torch route through
:func:`mhc_pre_admits`.
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

from vllm_neuron.functional.glue import glue_selected
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Largest token count served: ``residual^T`` blocks are the matmul's stationary.
MHC_PRE_MAX_TOKENS = 128

#: Tokens per transpose input: ``S * 32 = 128`` partitions at ``S = 4``.
_CHUNK = 32

#: Hidden blocks per PSUM tile of transposes (at most one 2 KiB bank of bf16).
_GROUP = 8


def _group(dtype):
    """Hidden blocks of ``dtype`` per 2 KiB PSUM bank of transposes: 8 bf16, 4 fp32."""
    return _GROUP if dtype == nl.bfloat16 else _GROUP // 2

_P = 128

__all__ = [
    "MHC_PRE_MAX_TOKENS",
    "dispatch_counters",
    "launch_programs",
    "mhc_pre_admits",
    "mhc_pre_fused",
    "mhc_pre_kernel",
    "reset_dispatch_counters",
]


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


@nki.jit
def mhc_pre_kernel(residual, fn, hc_scale, hc_base, RMS_EPS: float = 1e-6,
                   HC_EPS: float = 1e-6, POST_MULT: float = 2.0):
    """``(post [T, S, 1] fp32, comb_start [T, S, S] fp32, layer_input [T, H])``.

    ``layer_input`` is in the streams' dtype.

    Args:
        residual: ``[T, S, H]`` bf16 or fp32 streams, ``1 <= T <= 128``.
        fn: ``[M, S*H]`` bf16 or fp32 projection, ``M = 2S + S*S``.
        hc_scale: ``[3]`` fp32 head scales.
        hc_base: ``[M]`` fp32 head biases.
        RMS_EPS, HC_EPS, POST_MULT: the site's ``rms_eps``, ``hc_eps`` and
            ``post_mult_value``.
    """
    tokens, streams, hidden = residual.shape
    mix, width = fn.shape
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(1 <= tokens <= MHC_PRE_MAX_TOKENS, "1 <= T <= 128")
    kernel_assert(mix == 2 * streams + streams * streams, "fn rows are 2S + S*S")
    kernel_assert(width == streams * hidden, "fn columns are S*H")
    kernel_assert(mix * streams <= _P and streams * _CHUNK <= _P, "S <= 4")
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(hidden % (_P * programs) == 0, "H splits into 128-blocks per program")
    hp = hidden // programs
    h0 = program * hp
    nhb = hp // _P
    cols = mix + tokens
    ss = streams * streams
    # The matmul operands' dtype: bf16 when both are bf16 (the served case), else fp32
    # (the bf16 side is widened exactly by the copy out of PSUM).
    work = nl.bfloat16 if (residual.dtype == nl.bfloat16 and fn.dtype == nl.bfloat16) \
        else nl.float32

    post_out = nl.ndarray((tokens, streams, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_out = nl.ndarray((tokens, streams, streams), dtype=nl.float32,
                          buffer=nl.shared_hbm)
    x_out = nl.ndarray((tokens, hidden), dtype=residual.dtype, buffer=nl.shared_hbm)

    # ---- 1. fn^T and residual^T, hidden on the partitions. ----------------- #
    # mov[:, hb, s, 0:mix] = fn[:, s*H + h0 + hb*128 + p]^T,
    # mov[:, hb, s, mix + t] = residual[t, s, h0 + hb*128 + p].
    mov = nl.ndarray((_P, nhb, streams, cols), dtype=work, buffer=nl.sbuf)
    # fn rows (m, s) on partition m*S + s: one stride, H, over all M*S rows.
    f_rows = nl.ndarray((mix * streams, hp), dtype=fn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=f_rows, src=fn.ap(pattern=[[hidden, mix * streams], [1, hp]],
                                        offset=h0))
    f_group = _group(fn.dtype)
    for g0 in range(0, nhb, f_group):
        gn = min(f_group, nhb - g0)
        flipped = nl.ndarray((_P, gn, mix * streams), dtype=fn.dtype, buffer=nl.psum)
        for j in range(gn):
            hb = g0 + j
            nisa.nc_transpose(dst=flipped[:, j, :], data=f_rows[:, hb * _P:(hb + 1) * _P])
        nisa.tensor_copy(
            dst=mov[:, g0:g0 + gn, :, 0:mix],
            src=flipped.reshape((_P, gn, mix, streams)).permute((0, 1, 3, 2)))
    r_group = _group(residual.dtype)
    for t0 in range(0, tokens, _CHUNK):
        tn = min(_CHUNK, tokens - t0)
        rows = streams * tn
        # Streams rows (t, s) on partition t*S + s: one stride, H.
        r_rows = nl.ndarray((rows, hp), dtype=residual.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=r_rows, src=residual.ap(pattern=[[hidden, rows], [1, hp]],
                                                  offset=t0 * streams * hidden + h0))
        for g0 in range(0, nhb, r_group):
            gn = min(r_group, nhb - g0)
            flipped = nl.ndarray((_P, gn, rows), dtype=residual.dtype, buffer=nl.psum)
            for j in range(gn):
                hb = g0 + j
                nisa.nc_transpose(dst=flipped[:, j, :],
                                  data=r_rows[:, hb * _P:(hb + 1) * _P])
            nisa.tensor_copy(
                dst=mov[:, g0:g0 + gn, :, mix + t0:mix + t0 + tn],
                src=flipped.reshape((_P, gn, tn, streams)).permute((0, 1, 3, 2)))

    # ---- 2. Partial mixes and the Gram block over this program's K. -------- #
    acc = nl.ndarray((tokens, cols), dtype=nl.float32, buffer=nl.psum)
    blocks = nhb * streams
    for hb in range(nhb):
        for s in range(streams):
            k = hb * streams + s
            nisa.nc_matmul(dst=acc, stationary=mov[:, hb, s, mix:cols],
                           moving=mov[:, hb, s, 0:cols], accumulate=(k > 0))
    mine = _tile(tokens, cols)
    nisa.tensor_copy(dst=mine, src=acc)

    # ---- 3. Whole-K values on both programs. ------------------------------- #
    if programs == 2:
        theirs = _tile(tokens, cols)
        nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - program,
                      recv_from_rank=1 - program, pipe_id=0)
        total = _tile(tokens, cols)
        nisa.tensor_tensor(dst=total, data1=mine, data2=theirs, op=nl.add)
    else:
        total = mine

    # ---- 4. RMS scale, gates, softmax: tokens on the partitions. ----------- #
    ident = nl.shared_identity_matrix(n=_P, dtype=nl.float32)
    diag = _tile(tokens, tokens)
    nisa.tensor_tensor(dst=diag, data1=total[:, mix:cols], data2=ident[0:tokens, 0:tokens],
                       op=nl.multiply)
    sqrsum = _tile(tokens, 1)
    nisa.tensor_reduce(dst=sqrsum, op=nl.add, data=diag, axis=(1,))
    rstd = _tile(tokens, 1)
    # sqrsum / K + eps, as sqrsum * (1/K): the front end has no tensor_scalar divide,
    # and 1/K is exact (so the product is the quotient) for the served K = 16384 and
    # any power of two; mhc_pre_admits admits nothing else.
    nisa.tensor_scalar(dst=rstd, data=sqrsum, op0=nl.multiply, operand0=1.0 / float(width),
                       op1=nl.add, operand1=float(RMS_EPS))
    # rsqrt on GpSimd, the higher-precision engine for it (see functional/norm.py).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    mixes = _tile(tokens, mix)
    nisa.tensor_scalar(dst=mixes, data=total[:, 0:mix], op0=nl.multiply,
                       operand0=rstd[:, 0:1])

    # scale[head(m)] and base[m] as rows on partition 0, broadcast to the tokens by a
    # K=1 ones matmul (1.0 * a, exact).
    rows_in = _tile(1, 2 * mix)
    nisa.dma_copy(dst=rows_in[:, 0:streams], src=hc_scale.ap(pattern=[[0, 1], [0, streams]],
                                                             offset=0))
    nisa.dma_copy(dst=rows_in[:, streams:2 * streams],
                  src=hc_scale.ap(pattern=[[0, 1], [0, streams]], offset=1))
    nisa.dma_copy(dst=rows_in[:, 2 * streams:mix],
                  src=hc_scale.ap(pattern=[[0, 1], [0, ss]], offset=2))
    nisa.dma_copy(dst=rows_in[:, mix:2 * mix], src=hc_base.ap(pattern=[[0, 1], [1, mix]],
                                                              offset=0))
    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    spread = nl.ndarray((tokens, 2 * mix), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=spread, stationary=ones[0:1, 0:tokens], moving=rows_in,
                   accumulate=False)
    logits = _tile(tokens, mix)
    nisa.tensor_tensor(dst=logits, data1=mixes, data2=spread[:, 0:mix], op=nl.multiply)
    nisa.tensor_tensor(dst=logits, data1=logits, data2=spread[:, mix:2 * mix], op=nl.add)

    gates = _tile(tokens, 2 * streams)
    nisa.activation(dst=gates, op=nl.sigmoid, data=logits[:, 0:2 * streams])
    pre = _tile(tokens, streams)
    nisa.tensor_scalar(dst=pre, data=gates[:, 0:streams], op0=nl.add,
                       operand0=float(HC_EPS))
    post = _tile(tokens, streams)
    nisa.tensor_scalar(dst=post, data=gates[:, streams:2 * streams], op0=nl.multiply,
                       operand0=float(POST_MULT))

    # Its own [T, S, S] tile: a reshaped view of the logits slice has partition step 0
    # at T = 1, which the BIR verifier refuses.
    scores = nl.ndarray((tokens, streams, streams), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores.reshape((tokens, ss)), src=logits[:, 2 * streams:mix])
    peak = _tile(tokens, streams)
    nisa.tensor_reduce(dst=peak, op=nl.maximum, data=scores, axis=(2,))
    shifted = nl.ndarray((tokens, streams, streams), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=shifted, data1=scores,
                       data2=peak.expand_dim(2).broadcast(2, streams), op=nl.subtract)
    nisa.activation(dst=shifted, op=nl.exp, data=shifted)
    denom = _tile(tokens, streams)
    nisa.tensor_reduce(dst=denom, op=nl.add, data=shifted, axis=(2,))
    # e / sum as e * (1/sum): the front end has no divide; within one fp32 ulp.
    inv_denom = _tile(tokens, streams)
    nisa.reciprocal(dst=inv_denom, data=denom)
    comb = nl.ndarray((tokens, streams, streams), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=comb, data1=shifted,
                       data2=inv_denom.expand_dim(2).broadcast(2, streams), op=nl.multiply)
    nisa.tensor_scalar(dst=comb, data=comb, op0=nl.add, operand0=float(HC_EPS))

    if program == 0:
        nisa.dma_copy(dst=post_out.ap(pattern=[[streams, tokens], [1, streams]], offset=0),
                      src=post)
        nisa.dma_copy(dst=comb_out, src=comb)

    # ---- 5. Collapse with hidden on the partitions. ------------------------ #
    # spread_pre[p, s, t] = pre[t, s] on every partition: ones^T @ diag(pre[:, s]).
    eye_pre = nl.ndarray((tokens, streams, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=eye_pre,
                       data1=ident[0:tokens, 0:tokens].expand_dim(1).broadcast(1, streams),
                       data2=pre.expand_dim(2).broadcast(2, tokens), op=nl.multiply)
    pre_ps = nl.ndarray((_P, streams * tokens), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pre_ps, stationary=ones[0:tokens, :],
                   moving=eye_pre.reshape((tokens, streams * tokens)), accumulate=False)
    pre_all = nl.ndarray((_P, streams, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pre_all, src=pre_ps.reshape((_P, streams, tokens)))

    terms = nl.ndarray((_P, nhb, tokens, streams), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=terms,
        data1=mov[:, :, :, mix:cols].permute((0, 1, 3, 2)),
        data2=pre_all.permute((0, 2, 1)).expand_dim(1).broadcast(1, nhb),
        op=nl.multiply)
    collapsed = nl.ndarray((_P, nhb, tokens), dtype=residual.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=collapsed, op=nl.add, data=terms, axis=(3,))

    out_rows = nl.ndarray((tokens, hp), dtype=residual.dtype, buffer=nl.sbuf)
    for g0 in range(0, nhb, r_group):
        gn = min(r_group, nhb - g0)
        flipped = nl.ndarray((tokens, gn * _P), dtype=residual.dtype, buffer=nl.psum)
        for j in range(gn):
            nisa.nc_transpose(dst=flipped[:, j * _P:(j + 1) * _P],
                              data=collapsed[:, g0 + j, :])
        nisa.tensor_copy(dst=out_rows[:, g0 * _P:(g0 + gn) * _P], src=flipped)
    nisa.dma_copy(dst=x_out.ap(pattern=[[hidden, tokens], [1, hp]], offset=h0),
                  src=out_rows)
    return post_out, comb_out, x_out


_KERNELS = {1: wrap_nki(mhc_pre_kernel), 2: wrap_nki(mhc_pre_kernel)[2]}


@dataclass
class _DispatchCounters:
    """``nki_dispatch``: kernel launches. ``declined``: admission said no."""

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


def mhc_pre_admits(residual: Tensor, fn: Tensor, hc_scale: Tensor, hc_base: Tensor,
                   phase: str | None = None) -> bool:
    """True when :func:`mhc_pre_fused` serves this site call.

    Args:
        residual: ``[T, S, H]`` streams, bf16 (served) or fp32 (the CPU fixtures').
        fn: ``[M, S*H]`` projection, ``M = 2S + S*S``, bf16 or fp32.
        hc_scale: ``[3]`` fp32 head scales.
        hc_base: ``[M]`` fp32 head biases.
        phase: the step's ``"prefill"`` or ``"decode"``; None when it is not known,
            and then only a rule without a phase selects the kernel.

    The kernel serves ``1 <= T <= MHC_PRE_MAX_TOKENS``, streams that fit its partition
    layout (``S * 32 <= 128`` and ``M * S <= 128``, so ``S <= 4``), ``H`` in 128-blocks
    per program and ``S * H`` a power of two (so the kernel's ``* (1/K)`` is the
    reference's ``/ K``), on an NKI device or the simulator, when
    ``VLLM_NEURON_GLUE_FUSED`` selects ``mhc_pre`` for this row count and phase.
    Anything else keeps the torch route and is counted as declined.
    """
    ok = False
    if residual.dim() == 3 and fn.dim() == 2:
        tokens, streams, hidden = (int(v) for v in residual.shape)
        mix = 2 * streams + streams * streams
        programs = launch_programs()
        ok = (
            glue_selected("mhc_pre", tokens, phase)
            and residual.dtype in (torch.bfloat16, torch.float32)
            and fn.dtype in (torch.bfloat16, torch.float32)
            and hc_scale.dtype == torch.float32
            and hc_base.dtype == torch.float32
            and 1 <= tokens <= MHC_PRE_MAX_TOKENS
            and streams >= 1 and streams * _CHUNK <= _P and mix * streams <= _P
            and hidden % (_P * programs) == 0
            and (streams * hidden) & (streams * hidden - 1) == 0
            and tuple(fn.shape) == (mix, streams * hidden)
            and tuple(hc_scale.shape) == (3,)
            and tuple(hc_base.shape) == (mix,)
            and can_run_kernel(residual)
        )
    if not ok:
        _count_declined()
    return ok


def mhc_pre_fused(residual: Tensor, fn: Tensor, hc_scale: Tensor, hc_base: Tensor, *,
                  rms_eps: float, hc_eps: float, post_mult: float
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """``(post_mix [T, S, 1] fp32, comb_start [T, S, S] fp32, layer_input [T, H])``.

    ``comb_start`` is the softmax plus ``hc_eps``, the Sinkhorn's input; the caller
    runs the Sinkhorn. ``layer_input`` is in the streams' dtype.
    """
    _count_nki_dispatch()
    return _KERNELS[launch_programs()](
        residual=residual.contiguous(),
        fn=fn.contiguous(),
        hc_scale=hc_scale.contiguous(),
        hc_base=hc_base.contiguous(),
        RMS_EPS=float(rms_eps),
        HC_EPS=float(hc_eps),
        POST_MULT=float(post_mult),
    )
