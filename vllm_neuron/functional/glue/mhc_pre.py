# SPDX-License-Identifier: Apache-2.0
"""mHC "pre" up to the Sinkhorn, the collapse and the FFN norm, in one NKI launch.

``Glm5NextHyperConnection.mhc_pre``'s torch route runs this region as torch ops::

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

At the feed-forward site the sub-block first normalises ``layer_input``
(``Glm5NextModel._rms_norm``)::

    x      = layer_input.float()
    rstd   = rsqrt(x.square().mean(-1) + norm_eps)
    normed = (x * rstd * gain.float()).to(layer_input.dtype)

:func:`mhc_pre_norm_kernel` also returns ``normed``, from the rounded ``layer_input`` it
stores, so no torch op is left between the streams and the sub-block.

Shapes and layout. ``residual`` is ``[T, S, H]`` (row-major, so token ``t``'s stream
``s`` is one contiguous ``H`` run), ``fn`` is ``[M, S*H]``, ``hc_scale`` ``[3]``,
``hc_base`` ``[M]`` and the norm gain ``[H]``. The outputs are ``post [T, S, 1]`` and
``comb_start [T, S, S]`` in fp32, and ``layer_input``, ``normed`` and ``scaled``
``[T, H]`` in the streams' dtype.

Method. ``P`` programs split the hidden axis (``Hp = H / P``), and the tokens are walked
in tiles of :data:`MHC_PRE_TOKEN_TILE` rows (the last tile may be shorter). Once per
call, ``fn``'s ``[M*S, Hp]`` slice is loaded with (row, stream) on the partitions and
transposed on the tensor engine, 128 hidden values at a time, into the first ``M``
columns of the SBUF tile ``mov[128 (h), Hp/128, S, M + tile]``. Then, per token tile:

1. The tile's streams are loaded with the token on the partition, ``R[rows, S, Hp]``
   -- every DMA row is ``S`` contiguous ``Hp`` runs -- and each ``[rows, 128]`` block
   ``R[:, s, hb]`` is transposed into the last columns of ``mov``: for each contraction
   block ``(hb, s)`` the ``M`` columns of ``fn^T`` followed by the tile's columns of
   ``residual^T``. The copies out of PSUM alternate between the scalar and the vector
   engine.
2. ``acc[rows, M + rows] += mov[:, hb, s, M:]^T @ mov[:, hb, s, :]`` over the
   ``S * Hp/128`` blocks: columns ``:M`` are this program's partial ``mixes``, the
   ``[rows, rows]`` rest is the streams' Gram block, whose diagonal is the square sum.
   The ``residual^T`` block is the matmul's stationary operand, which is why a tile is
   at most ``gemm_stationary_fmax`` tokens. bf16 x bf16 products are exact in the fp32
   accumulator, so both differ from the torch values only by summation order.
3. Under two programs the two partials are swapped (``sendrecv``) and added, so both
   programs hold the whole-``K`` values; ``a + b`` and ``b + a`` are the same fp32
   number, so the two programs agree bit for bit.
4. The RMS scale, the gates and the softmax run with tokens on the partitions, in the
   torch expression's operation order. Program 0 stores ``post`` and ``comb_start``.
5. The collapse runs on ``R``, tokens on the partitions, with ``pre[t, s]`` a
   per-partition scalar: ``acc = R[:, 0] * pre_0``, then ``acc = R[:, s] * pre_s + acc``
   for ``s = 1..S-1``, each product one fp32 multiply and each sum one fp32 add, the
   last rounded once to the streams' dtype, as the torch expression does. The result
   rows are this program's hidden half of ``layer_input``, stored as they are.
6. The norm (:func:`mhc_pre_norm_kernel` only): the square sum of the rounded
   ``layer_input`` is the diagonal of its Gram block (step 2's method, on step 5's rows
   transposed to hidden-major), the two programs' halves are swapped and added as in
   step 3, and each row is scaled by ``rsqrt(sum * (1/H) + norm_eps)`` and then by the
   gain in one fp32 instruction, in ``_rms_norm``'s operation order, and rounded once.
   With ``ROUTER_SCALE`` the rows are also scaled by the same ``rsqrt`` alone and rounded
   once: ``scaled``, the MoE router's input (its gain is applied in the router).

Both LNC2 cores: yes. Each program does half the contraction (step 2), half the
transposes, half the collapse and half the norm; steps 3-4 are ``rows x (M + rows)``
values, done on both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.functional.glue import glue_selected
from vllm_neuron.utils.neuron_utils import can_run_kernel

_P = nl.tile_size.pmax

#: Tokens per tile: a tile's ``residual^T`` block is the stationary operand of step 2,
#: and steps 4-6 hold the tile's tokens on the partitions.
MHC_PRE_TOKEN_TILE = min(nl.tile_size.gemm_stationary_fmax, _P)

#: Hidden blocks per PSUM tile of transposes (at most one 2 KiB bank of bf16).
_GROUP = 8


def _group(dtype):
    """Hidden blocks of ``dtype`` per 2 KiB PSUM bank of transposes: 8 bf16, 4 fp32."""
    return _GROUP if dtype == nl.bfloat16 else _GROUP // 2


__all__ = [
    "MHC_PRE_TOKEN_TILE",
    "dispatch_counters",
    "launch_programs",
    "mhc_pre_admits",
    "mhc_pre_fused",
    "mhc_pre_kernel",
    "mhc_pre_norm_kernel",
    "normed_dispatches",
    "reset_dispatch_counters",
]


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _swap_add(mine, rows, cols, programs, program):
    """``mine`` plus the other program's ``mine`` (the whole-``K`` value), or ``mine``."""
    if programs == 1:
        return mine
    theirs = _tile(rows, cols)
    nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - program,
                  recv_from_rank=1 - program, pipe_id=0)
    total = _tile(rows, cols)
    nisa.tensor_tensor(dst=total, data1=mine, data2=theirs, op=nl.add)
    return total


def _diagonal_sum(gram, rows, ident):
    """``[rows, 1]``: the diagonal of a ``[rows, rows]`` Gram block (times 1, plus 0s)."""
    diag = _tile(rows, rows)
    nisa.tensor_tensor(dst=diag, data1=gram, data2=ident[0:rows, 0:rows], op=nl.multiply)
    total = _tile(rows, 1)
    nisa.tensor_reduce(dst=total, op=nl.add, data=diag, axis=(1,))
    return total


def _rstd(sqrsum, rows, inv_count, eps):
    """``[rows, 1]``: ``rsqrt(sqrsum * inv_count + eps)``.

    ``* inv_count`` stands for the reference's ``/ count``: the front end has no
    tensor_scalar divide, and ``1 / count`` is exact (so the product is the quotient)
    for a power-of-two count, the only kind :func:`mhc_pre_admits` admits.
    """
    rstd = _tile(rows, 1)
    nisa.tensor_scalar(dst=rstd, data=sqrsum, op0=nl.multiply, operand0=inv_count,
                       op1=nl.add, operand1=float(eps))
    # rsqrt on GpSimd, the higher-precision engine for it (see functional/norm.py).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    return rstd


def _transpose_into(dst, rows_t, col0, rows_n, nhb, group):
    """``dst[:, hb, s, col0 + t] = rows_t[t, s, hb*128 + p]`` on partition ``p``.

    ``rows_t`` is ``[rows_n, S, nhb*128]`` with the token on the partition. Each
    ``[rows_n, 128]`` block goes through the tensor engine (exact), ``group`` blocks per
    PSUM tile, and each tile's copy into ``dst`` alternates between the scalar and the
    vector engine. A block's PSUM columns are padded to an even count, so every block
    starts 4-byte aligned (a tensor-engine write rule) for an odd ``rows_n`` too.
    """
    streams = rows_t.shape[1]
    padded = rows_n + rows_n % 2
    turn = 0
    for s in range(streams):
        for g0 in range(0, nhb, group):
            gn = min(group, nhb - g0)
            flipped = nl.ndarray((_P, gn, padded), dtype=rows_t.dtype, buffer=nl.psum)
            for j in range(gn):
                hb = g0 + j
                nisa.nc_transpose(dst=flipped[:, j, 0:rows_n],
                                  data=rows_t[:, s, hb * _P:(hb + 1) * _P])
            target = dst[:, g0:g0 + gn, s, col0:col0 + rows_n]
            if turn % 2 == 0:
                nisa.activation(dst=target, op=nl.copy, data=flipped[:, :, 0:rows_n])
            else:
                nisa.tensor_copy(dst=target, src=flipped[:, :, 0:rows_n],
                                 engine=nisa.engine.vector)
            turn += 1


def _mhc_pre(residual, fn, hc_scale, hc_base, norm_gain, rms_eps, hc_eps, post_mult,
             norm_eps, router_scale=False):
    """The body of both kernels; ``norm_gain`` None skips step 6 (see the module doc).

    ``router_scale`` (with ``norm_gain`` only) adds step 6's ``scaled`` output.
    """
    tokens, streams, hidden = residual.shape
    mix, width = fn.shape
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(tokens >= 1, "T >= 1")
    kernel_assert(mix == 2 * streams + streams * streams, "fn rows are 2S + S*S")
    kernel_assert(width == streams * hidden, "fn columns are S*H")
    kernel_assert(mix * streams <= _P, "S <= 4")
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(hidden % (_P * programs) == 0, "H splits into 128-blocks per program")
    if norm_gain is not None:
        kernel_assert(tuple(norm_gain.shape) == (hidden,), "the norm gain is [H]")
    kernel_assert(norm_gain is not None or not router_scale, "scaled comes with the norm")
    hp = hidden // programs
    h0 = program * hp
    nhb = hp // _P
    ss = streams * streams
    tile_rows = min(MHC_PRE_TOKEN_TILE, tokens)
    # The matmul operands' dtype: bf16 when both are bf16 (the served case), else fp32
    # (the bf16 side is widened exactly by the copy out of PSUM).
    work = nl.bfloat16 if (residual.dtype == nl.bfloat16 and fn.dtype == nl.bfloat16) \
        else nl.float32

    post_out = nl.ndarray((tokens, streams, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_out = nl.ndarray((tokens, streams, streams), dtype=nl.float32,
                          buffer=nl.shared_hbm)
    x_out = nl.ndarray((tokens, hidden), dtype=residual.dtype, buffer=nl.shared_hbm)
    normed_out = None if norm_gain is None else nl.ndarray(
        (tokens, hidden), dtype=residual.dtype, buffer=nl.shared_hbm)
    scaled_out = nl.ndarray((tokens, hidden), dtype=residual.dtype,
                            buffer=nl.shared_hbm) if router_scale else None

    # ---- Once per call: fn^T, the head rows, the constants. --------------- #
    # mov[:, hb, s, 0:mix] = fn[:, s*H + h0 + hb*128 + p]^T; step 1 of each tile fills
    # mov[:, hb, s, mix + t] = residual[t0 + t, s, h0 + hb*128 + p].
    mov = nl.ndarray((_P, nhb, streams, mix + tile_rows), dtype=work, buffer=nl.sbuf)
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

    ident = nl.shared_identity_matrix(n=_P, dtype=nl.float32)
    ones = nl.ndarray((1, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    # scale[head(m)] and base[m] as rows on partition 0, broadcast to a tile's tokens by
    # a K=1 ones matmul (1.0 * a, exact).
    rows_in = _tile(1, 2 * mix)
    nisa.dma_copy(dst=rows_in[:, 0:streams], src=hc_scale.ap(pattern=[[0, 1], [0, streams]],
                                                             offset=0))
    nisa.dma_copy(dst=rows_in[:, streams:2 * streams],
                  src=hc_scale.ap(pattern=[[0, 1], [0, streams]], offset=1))
    nisa.dma_copy(dst=rows_in[:, 2 * streams:mix],
                  src=hc_scale.ap(pattern=[[0, 1], [0, ss]], offset=2))
    nisa.dma_copy(dst=rows_in[:, mix:2 * mix], src=hc_base.ap(pattern=[[0, 1], [1, mix]],
                                                              offset=0))
    spread_ps = nl.ndarray((tile_rows, 2 * mix), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=spread_ps, stationary=ones[0:1, 0:tile_rows], moving=rows_in,
                   accumulate=False)
    spread = _tile(tile_rows, 2 * mix)
    nisa.tensor_copy(dst=spread, src=spread_ps)
    if norm_gain is not None:
        # This program's gain slice on every token partition (partition stride 0).
        gain_rows = nl.ndarray((tile_rows, hp), dtype=norm_gain.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gain_rows, src=norm_gain.ap(pattern=[[0, tile_rows], [1, hp]],
                                                      offset=h0))

    r_group = _group(residual.dtype)
    for t0 in range(0, tokens, MHC_PRE_TOKEN_TILE):
        rows_n = min(MHC_PRE_TOKEN_TILE, tokens - t0)
        cols = mix + rows_n

        # ---- 1. This tile's streams, token rows, and their transposes. ------- #
        streams_t = nl.ndarray((rows_n, streams, hp), dtype=residual.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=streams_t,
                      src=residual.ap(pattern=[[streams * hidden, rows_n], [hidden, streams],
                                               [1, hp]],
                                      offset=t0 * streams * hidden + h0))
        _transpose_into(mov, streams_t, mix, rows_n, nhb, r_group)

        # ---- 2. Partial mixes and the Gram block over this program's K. ---- #
        acc = nl.ndarray((rows_n, cols), dtype=nl.float32, buffer=nl.psum)
        for hb in range(nhb):
            for s in range(streams):
                k = hb * streams + s
                nisa.nc_matmul(dst=acc, stationary=mov[:, hb, s, mix:cols],
                               moving=mov[:, hb, s, 0:cols], accumulate=(k > 0))
        mine = _tile(rows_n, cols)
        nisa.tensor_copy(dst=mine, src=acc)

        # ---- 3. Whole-K values on both programs. --------------------------- #
        total = _swap_add(mine, rows_n, cols, programs, program)

        # ---- 4. RMS scale, gates, softmax: tokens on the partitions. ------- #
        rstd = _rstd(_diagonal_sum(total[:, mix:cols], rows_n, ident), rows_n,
                     1.0 / float(width), rms_eps)
        mixes = _tile(rows_n, mix)
        nisa.tensor_scalar(dst=mixes, data=total[:, 0:mix], op0=nl.multiply,
                           operand0=rstd[:, 0:1])
        logits = _tile(rows_n, mix)
        nisa.tensor_tensor(dst=logits, data1=mixes, data2=spread[0:rows_n, 0:mix],
                           op=nl.multiply)
        nisa.tensor_tensor(dst=logits, data1=logits, data2=spread[0:rows_n, mix:2 * mix],
                           op=nl.add)

        gates = _tile(rows_n, 2 * streams)
        nisa.activation(dst=gates, op=nl.sigmoid, data=logits[:, 0:2 * streams])
        pre = _tile(rows_n, streams)
        nisa.tensor_scalar(dst=pre, data=gates[:, 0:streams], op0=nl.add,
                           operand0=float(hc_eps))
        post = _tile(rows_n, streams)
        nisa.tensor_scalar(dst=post, data=gates[:, streams:2 * streams], op0=nl.multiply,
                           operand0=float(post_mult))

        # Its own [rows, S, S] tile: a reshaped view of the logits slice has partition
        # step 0 at one row, which the BIR verifier refuses.
        scores = nl.ndarray((rows_n, streams, streams), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=scores.reshape((rows_n, ss)), src=logits[:, 2 * streams:mix])
        peak = _tile(rows_n, streams)
        nisa.tensor_reduce(dst=peak, op=nl.maximum, data=scores, axis=(2,))
        shifted = nl.ndarray((rows_n, streams, streams), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=shifted, data1=scores,
                           data2=peak.expand_dim(2).broadcast(2, streams), op=nl.subtract)
        nisa.activation(dst=shifted, op=nl.exp, data=shifted)
        denom = _tile(rows_n, streams)
        nisa.tensor_reduce(dst=denom, op=nl.add, data=shifted, axis=(2,))
        # e / sum as e * (1/sum): the front end has no divide; within one fp32 ulp.
        inv_denom = _tile(rows_n, streams)
        nisa.reciprocal(dst=inv_denom, data=denom)
        comb = nl.ndarray((rows_n, streams, streams), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=comb, data1=shifted,
                           data2=inv_denom.expand_dim(2).broadcast(2, streams),
                           op=nl.multiply)
        nisa.tensor_scalar(dst=comb, data=comb, op0=nl.add, operand0=float(hc_eps))

        if program == 0:
            nisa.dma_copy(dst=post_out.ap(pattern=[[streams, rows_n], [1, streams]],
                                          offset=t0 * streams),
                          src=post)
            nisa.dma_copy(dst=comb_out.ap(pattern=[[ss, rows_n], [streams, streams],
                                                   [1, streams]], offset=t0 * ss),
                          src=comb)

        # ---- 5. Collapse with tokens on the partitions. --------------------- #
        acc_li = nl.ndarray((rows_n, hp), dtype=nl.float32, buffer=nl.sbuf)
        out_rows = nl.ndarray((rows_n, hp), dtype=residual.dtype, buffer=nl.sbuf)
        for s_i in range(streams):
            into = out_rows if s_i == streams - 1 else acc_li
            if s_i == 0:
                nisa.tensor_scalar(dst=into, data=streams_t[:, 0, :], op0=nl.multiply,
                                   operand0=pre[:, 0:1])
            else:
                nisa.scalar_tensor_tensor(dst=into, data=streams_t[:, s_i, :],
                                          op0=nl.multiply, operand0=pre[:, s_i:s_i + 1],
                                          op1=nl.add, operand1=acc_li)
        nisa.dma_copy(dst=x_out.ap(pattern=[[hidden, rows_n], [1, hp]],
                                   offset=t0 * hidden + h0),
                      src=out_rows)

        # ---- 6. The FFN norm of the rounded layer_input. ------------------- #
        if norm_gain is not None:
            collapsed = nl.ndarray((_P, nhb, 1, rows_n), dtype=residual.dtype,
                                   buffer=nl.sbuf)
            _transpose_into(collapsed, out_rows.reshape((rows_n, 1, hp)), 0, rows_n, nhb,
                            r_group)
            gram = nl.ndarray((rows_n, rows_n), dtype=nl.float32, buffer=nl.psum)
            for hb in range(nhb):
                nisa.nc_matmul(dst=gram, stationary=collapsed[:, hb, 0, :],
                               moving=collapsed[:, hb, 0, :], accumulate=(hb > 0))
            sqrsum = _swap_add(_diagonal_sum(gram, rows_n, ident), rows_n, 1, programs,
                               program)
            norm_rstd = _rstd(sqrsum, rows_n, 1.0 / float(hidden), norm_eps)
            normed_rows = nl.ndarray((rows_n, hp), dtype=residual.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=normed_rows, data=out_rows, op0=nl.multiply,
                                      operand0=norm_rstd[:, 0:1], op1=nl.multiply,
                                      operand1=gain_rows[0:rows_n, :])
            nisa.dma_copy(dst=normed_out.ap(pattern=[[hidden, rows_n], [1, hp]],
                                            offset=t0 * hidden + h0),
                          src=normed_rows)
            if router_scale:
                scaled_rows = nl.ndarray((rows_n, hp), dtype=residual.dtype,
                                         buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scaled_rows, data=out_rows, op0=nl.multiply,
                                   operand0=norm_rstd[:, 0:1])
                nisa.dma_copy(dst=scaled_out.ap(pattern=[[hidden, rows_n], [1, hp]],
                                                offset=t0 * hidden + h0),
                              src=scaled_rows)
    if norm_gain is None:
        return post_out, comb_out, x_out
    if router_scale:
        return post_out, comb_out, x_out, normed_out, scaled_out
    return post_out, comb_out, x_out, normed_out


@nki.jit
def mhc_pre_kernel(residual, fn, hc_scale, hc_base, RMS_EPS: float = 1e-6,
                   HC_EPS: float = 1e-6, POST_MULT: float = 2.0):
    """``(post [T, S, 1] fp32, comb_start [T, S, S] fp32, layer_input [T, H])``.

    ``layer_input`` is in the streams' dtype.

    Args:
        residual: ``[T, S, H]`` bf16 or fp32 streams, ``T >= 1``.
        fn: ``[M, S*H]`` bf16 or fp32 projection, ``M = 2S + S*S``.
        hc_scale: ``[3]`` fp32 head scales.
        hc_base: ``[M]`` fp32 head biases.
        RMS_EPS, HC_EPS, POST_MULT: the site's ``rms_eps``, ``hc_eps`` and
            ``post_mult_value``.
    """
    return _mhc_pre(residual, fn, hc_scale, hc_base, None, RMS_EPS, HC_EPS, POST_MULT,
                    None)


@nki.jit
def mhc_pre_norm_kernel(residual, fn, hc_scale, hc_base, norm_gain, RMS_EPS: float,
                        HC_EPS: float, POST_MULT: float, NORM_EPS: float,
                        ROUTER_SCALE: bool = False):
    """:func:`mhc_pre_kernel`'s three outputs, then ``normed [T, H]`` (and ``scaled``).

    ``normed = layer_input * rsqrt(mean(layer_input**2) + NORM_EPS) * norm_gain``,
    computed in fp32 from the returned (rounded) ``layer_input`` and returned in the
    streams' dtype: ``Glm5NextModel._rms_norm`` of ``layer_input``.

    With ``ROUTER_SCALE`` a fifth output, ``scaled [T, H]`` in the streams' dtype, is
    ``layer_input * rsqrt(mean(layer_input**2) + NORM_EPS)`` rounded once: the same fp32
    ``rsqrt`` as ``normed``'s, one fp32 multiply, no gain. It is
    :func:`~vllm_neuron.functional.moe.router_prefill.router_rms_scale` of
    ``layer_input``, the router's input up to the order of the square sum.

    Args:
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, POST_MULT: as
            :func:`mhc_pre_kernel`.
        norm_gain: ``[H]`` bf16 or fp32 gain of the sub-block's input RMSNorm.
        NORM_EPS: that norm's epsilon.
        ROUTER_SCALE: also return ``scaled``.
    """
    return _mhc_pre(residual, fn, hc_scale, hc_base, norm_gain, RMS_EPS, HC_EPS,
                    POST_MULT, NORM_EPS, ROUTER_SCALE)


_KERNELS = {1: wrap_nki(mhc_pre_kernel), 2: wrap_nki(mhc_pre_kernel)[2]}
_NORM_KERNELS = {1: wrap_nki(mhc_pre_norm_kernel), 2: wrap_nki(mhc_pre_norm_kernel)[2]}


@dataclass
class _DispatchCounters:
    """``nki_dispatch``: kernel launches that returned (``normed``: those with the norm).

    ``declined``: admission said no.
    """

    nki_dispatch: int = 0
    declined: int = 0
    normed: int = 0


_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.declined = 0
    _COUNTERS.normed = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, declined)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.declined


def normed_dispatches() -> int:
    """Launches of :func:`mhc_pre_norm_kernel` since the last reset.

    Each is also counted in ``nki_dispatch``.
    """
    return _COUNTERS.normed


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_normed() -> None:
    _COUNTERS.normed += 1


@torch._dynamo.assume_constant_result
def _count_declined() -> None:
    _COUNTERS.declined += 1


def launch_programs() -> int:
    """2 programs (both LNC2 cores) under ``NEURON_LOGICAL_NC_CONFIG=2``, 1 when unset or 1.

    :func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair` refuses a setting other
    than unset, 1 or 2 (``LaunchGridError``).
    """
    return 2 if lnc_pair() else 1


def mhc_pre_admits(residual: Tensor, fn: Tensor, hc_scale: Tensor, hc_base: Tensor,
                   phase: str | None = None, norm_gain: Tensor | None = None) -> bool:
    """True when :func:`mhc_pre_fused` serves this site call.

    Args:
        residual: ``[T, S, H]`` streams, bf16 (served) or fp32 (the CPU fixtures').
        fn: ``[M, S*H]`` projection, ``M = 2S + S*S``, bf16 or fp32.
        hc_scale: ``[3]`` fp32 head scales.
        hc_base: ``[M]`` fp32 head biases.
        phase: the step's ``"prefill"``, ``"decode"`` or ``"verify"``, which the mHC
            layer tells from the row count
            (:func:`~vllm_neuron.functional.glue.phase_of_rows`); None when it is not
            known, and then only a rule without a phase selects the kernel.
        norm_gain: the ``[H]`` gain the call also asks the norm with, or None.

    The kernel serves any ``T >= 1`` (in tiles of :data:`MHC_PRE_TOKEN_TILE` rows),
    streams that fit its partition layout (``M * S <= 128``, so ``S <= 4``), ``H`` in
    128-blocks per program and ``S * H`` a power of two (so ``S`` and ``H`` are too, and
    the kernel's ``* (1/K)`` and ``* (1/H)`` are the reference's ``/ K`` and ``/ H``), and
    a bf16 or fp32 ``[H]`` norm gain, on an NKI device or the simulator, when
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
            and tokens >= 1
            and streams >= 1 and mix * streams <= _P
            and hidden % (_P * programs) == 0
            and (streams * hidden) & (streams * hidden - 1) == 0
            and tuple(fn.shape) == (mix, streams * hidden)
            and tuple(hc_scale.shape) == (3,)
            and tuple(hc_base.shape) == (mix,)
            and (norm_gain is None or (norm_gain.dtype in (torch.bfloat16, torch.float32)
                                       and tuple(norm_gain.shape) == (hidden,)))
            and can_run_kernel(residual)
        )
    if not ok:
        _count_declined()
    return ok


def mhc_pre_fused(residual: Tensor, fn: Tensor, hc_scale: Tensor, hc_base: Tensor, *,
                  rms_eps: float, hc_eps: float, post_mult: float,
                  norm_gain: Tensor | None = None, norm_eps: float | None = None,
                  router_scale: bool = False) -> tuple[Tensor | None, ...]:
    """``(post_mix [T, S, 1] fp32, comb_start [T, S, S] fp32, layer_input [T, H], normed)``.

    ``comb_start`` is the softmax plus ``hc_eps``, the Sinkhorn's input; the caller
    runs the Sinkhorn. ``layer_input`` is in the streams' dtype. ``normed`` is
    ``layer_input``'s RMSNorm with ``norm_gain`` and ``norm_eps``
    (:func:`mhc_pre_norm_kernel`), in the same dtype, or None without ``norm_gain``.
    With ``router_scale`` a fifth value follows: ``scaled [T, H]``, the same norm
    without the gain (``mhc_pre_norm_kernel``'s ``ROUTER_SCALE`` output).

    Raises:
        ValueError: on a ``norm_gain`` without its ``norm_eps``, or ``router_scale``
            without ``norm_gain``.
    """
    if norm_gain is not None and norm_eps is None:
        raise ValueError("mhc_pre_fused: norm_gain needs its norm_eps")
    if router_scale and norm_gain is None:
        raise ValueError("mhc_pre_fused: router_scale needs norm_gain")
    operands = dict(
        residual=residual.contiguous(),
        fn=fn.contiguous(),
        hc_scale=hc_scale.contiguous(),
        hc_base=hc_base.contiguous(),
        RMS_EPS=float(rms_eps),
        HC_EPS=float(hc_eps),
        POST_MULT=float(post_mult),
    )
    if norm_gain is None:
        outputs = (*_KERNELS[launch_programs()](**operands), None)
    else:
        outputs = tuple(_NORM_KERNELS[launch_programs()](
            **operands, norm_gain=norm_gain.contiguous(), NORM_EPS=float(norm_eps),
            ROUTER_SCALE=bool(router_scale)))
        _count_normed()
    # Counted once the launch has returned: a launch that raises served nothing.
    _count_nki_dispatch()
    return outputs
