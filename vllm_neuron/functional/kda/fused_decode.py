# SPDX-License-Identifier: Apache-2.0
"""KDA decode in one NKI launch: conv carrier update, gate, and state step.

At 5938748 one decode token cost each KDA layer three kernels and a dozen torch
ops between them: the ``nkilib`` depthwise conv (384 per-channel DMA round trips,
192 in series per core, ~350 us), the gate clamp and the decode-state step, plus
the carrier selects, casts, transposes and copies of the model glue. This module
does that whole region for ``B`` requests in one dispatch::

    hist   = 0 if start == 0 else conv_state          # NaN-safe select
    padded = [hist; q_in | k_in | v_in]                # [R + 1, C]
    q,k,v  = silu(sum_s w[c, s] * padded[s, c])        # depthwise, R + 1 taps
    conv_state' = padded[1:] if real else padded[:R]   # bank dtype and layout
    gk     = lower * sigmoid(exp(A_log) * (raw_gate + dt_bias)) * mask
    beta   = sigmoid(raw_beta) * mask
    S      = 0 if start == 0 else S                    # [V, K] per head
    S      = S * exp(gk);  delta = (v - S kn) * beta;  S += delta kn^T
    o      = S qn                                      # kn, qn L2-normalised

which is ``decode_state``'s step and ``gate_clamp``'s gate unchanged, fed by the
same conv. Inputs and outputs keep the runner's carrier layouts: the conv carrier
``[B, R, C]`` (``SD``) or ``[B, C, R]`` (``DS``) in its own dtype, the recurrent
carrier ``[B, H, V, K]`` float32.

Layout inside the kernel. Channels sit on partitions for the conv (taps on the
free axis, so the depthwise sum is a free-axis reduce); the carrier rows are
loaded as contiguous rows and turned by the tensor engine, which is bit-exact for
transposes on this target. Each state is turned to ``[K, V]`` so the per-key
decay and ``-beta * kn`` are per-partition scalars. ``S kn - v`` is one PSUM
accumulation that lands broadcast on every partition: ``kn`` broadcast along the
free axis as the stationary operand, plus a one-hot row selecting ``-v``. The
update is then a single ``scalar_tensor_tensor``.

Both LNC2 cores. Under ``NEURON_LOGICAL_NC_CONFIG=2`` the kernel launches with a
grid of 2 and program ``p`` owns value rows ``[p V/2, (p+1) V/2)`` of every state,
the matching ``o`` columns, and half the conv-carrier channels. Each program
computes the whole conv and gate (``q``, ``k`` and the decay are needed by every
value row; that part is a few hundred elements), so the split halves the state
traffic and the per-request tensor-engine work.

A refused geometry raises :class:`KdaFusedDecodeError`; when no device or
simulator exists, a counted torch reference serves the call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.kda.chunked_recurrence import L2_NORM_EPS, MAX_TILE
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Partition count, the bound on every transposed extent here.
PMAX = 128

#: Free-axis float32 capacity of one PSUM bank, the bound on a matmul output.
PSUM_BANK_F32 = 512

#: Environment switch for the model call site; ``0`` routes decode back to the
#: three 5938748 kernels.
FUSED_DECODE_ENV = "VLLM_NEURON_KDA_FUSED_DECODE"

__all__ = [
    "FUSED_DECODE_ENV",
    "FusedDecodeOutputs",
    "KdaFusedDecodeError",
    "can_run_fused_decode",
    "fused_decode_dispatch_counters",
    "fused_decode_enabled",
    "fused_decode_grid",
    "fused_decode_kernel_identity",
    "kda_fused_decode",
    "kda_fused_decode_kernel",
    "kda_fused_decode_torch_reference",
    "requests_per_chunk",
    "reset_fused_decode_dispatch_counters",
]


class KdaFusedDecodeError(ValueError):
    """A geometry or operand this kernel refuses, named rather than coerced."""


class FusedDecodeOutputs(NamedTuple):
    """The step's ``core`` ``[B, H*V]`` and both advanced carriers."""

    core: Tensor
    conv_state: Tensor
    recurrent_state: Tensor


@dataclass
class _FusedDecodeCounters:
    nki_dispatch: int = 0
    torch_fallback: int = 0


_COUNTERS = _FusedDecodeCounters()


def reset_fused_decode_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def fused_decode_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def requests_per_chunk(batch: int, taps: int, heads: int) -> int:
    """Requests the kernel serves per pass.

    The conv rows tile holds ``taps`` rows per request and the value-rows tile
    ``heads`` rows per request, and both are turned by the tensor engine, which
    serves ``PMAX`` partitions.
    """
    return max(1, min(batch, PMAX // max(taps, heads)))


# --------------------------------------------------------------------------- #
# The kernel
# --------------------------------------------------------------------------- #
def _load_taps(dst, w_hbm, stream, kdim, heads, taps):
    """One stream's ``[H*K, taps]`` conv taps into ``dst[:, stream*H:(stream+1)*H]``."""
    nisa.dma_copy(
        dst=dst[0:kdim, stream * heads : (stream + 1) * heads, 0:taps],
        src=w_hbm.ap(pattern=[[taps, kdim], [kdim * taps, heads], [1, taps]], offset=0),
    )


def _load_token_rows(dst, x_hbm, stream, hrows, srows, c0, bc, width):
    """One stream's token rows into the conv rows tile, after the carrier rows."""
    nisa.dma_copy(
        dst=dst[hrows:srows, stream * width : (stream + 1) * width],
        src=x_hbm[c0 : c0 + bc, 0:width],
    )


@nki.jit
def kda_fused_decode_kernel(
    q_in,
    k_in,
    v_in,
    raw_gate,
    raw_beta,
    conv_state,
    q_w,
    k_w,
    v_w,
    a_log,
    dt_bias,
    rec_state,
    start_pos,
    real_tokens,
    row_mask,
    lower,
    dim_first,
):
    """One decode token for each of ``B`` requests; see the module docstring.

    ``q_in``/``k_in``/``v_in``/``raw_gate`` are ``[B, H*K]`` float32,
    ``raw_beta`` ``[B, H]`` float32, ``conv_state`` ``[B, R, C]`` (``[B, C, R]``
    when ``dim_first``), ``q_w``/``k_w``/``v_w`` ``[H*K, R+1]``, ``a_log``
    ``[1, H]``, ``dt_bias`` ``[1, H*K]``, ``rec_state`` ``[B, H, V, K]`` float32,
    ``start_pos`` ``[1, B]`` int32, ``real_tokens`` ``[1, B]`` int32 or None,
    ``row_mask`` ``[1, B]`` float32 or None. Returns ``(core [B, H*V],
    rec_out [B, H, V, K], conv_out like conv_state)``.
    """
    batch, width = q_in.shape
    _, heads, vdim, kdim = rec_state.shape
    taps = q_w.shape[1]
    rows = taps - 1
    channels = 3 * width
    blocks = 3 * heads
    conv_dtype = conv_state.dtype
    n_prog = nl.num_programs(0)
    prog = nl.program_id(0)
    vpart = vdim // n_prog
    cpart = channels // n_prog
    chunk = min(batch, PMAX // max(taps, heads))
    if chunk < 1:
        chunk = 1
    scale = float(kdim) ** -0.5

    core_hbm = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.shared_hbm)
    rec_hbm = nl.ndarray(
        (batch, heads, vdim, kdim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_hbm = nl.ndarray(conv_state.shape, dtype=conv_dtype, buffer=nl.shared_hbm)

    # ---- launch constants ------------------------------------------------- #
    ones = nl.ndarray((kdim, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    eye_n = heads * chunk
    eye = nl.shared_identity_matrix(n=eye_n, dtype=nl.float32)

    # Conv taps, channel on partitions: [K, blocks, taps, 1], float32.
    w_raw = nl.ndarray((kdim, blocks, taps), dtype=q_w.dtype, buffer=nl.sbuf)
    _load_taps(w_raw, q_w, 0, kdim, heads, taps)
    _load_taps(w_raw, k_w, 1, kdim, heads, taps)
    _load_taps(w_raw, v_w, 2, kdim, heads, taps)
    w32 = nl.ndarray((kdim, blocks, taps, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w32.reshape((kdim, blocks, taps)), src=w_raw)

    # exp(A_log) on every partition, and dt_bias as one column per head.
    a_raw = nl.ndarray((kdim, heads), dtype=a_log.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_raw, src=a_log.broadcast(0, kdim))
    exp_a = nl.ndarray((kdim, heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_a, data=a_raw, op=nl.exp)
    bias_raw = nl.ndarray((kdim, heads), dtype=dt_bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias_raw, src=dt_bias.ap(pattern=[[1, kdim], [kdim, heads]], offset=0)
    )
    bias = nl.ndarray((kdim, heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bias, src=bias_raw)

    for chunk_index in range(-(-batch // chunk)):
        c0 = chunk_index * chunk
        bc = min(chunk, batch - c0)
        hb = heads * bc
        srows = taps * bc
        hrows = rows * bc

        # ---- per-request step operands, on every partition ---------------- #
        pos = nl.ndarray((PMAX, bc), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pos, src=start_pos[0:1, c0 : c0 + bc].broadcast(0, PMAX))
        keep = nl.ndarray((PMAX, bc), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=pos, op0=nl.not_equal, operand0=0)
        if real_tokens != None:  # noqa: E711 -- NKI traces `!=`, not `is not`
            real = nl.ndarray((PMAX, bc), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=real, src=real_tokens[0:1, c0 : c0 + bc].broadcast(0, PMAX)
            )
            shift = nl.ndarray((PMAX, bc), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=shift, data=real, op0=nl.not_equal, operand0=0)
        if row_mask != None:  # noqa: E711
            mask = nl.ndarray((PMAX, bc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=mask, src=row_mask[0:1, c0 : c0 + bc].broadcast(0, PMAX))

        # ---- conv rows: carrier rows (r, b) then this token's rows (b) ------ #
        hist = nl.ndarray((hrows, channels), dtype=conv_dtype, buffer=nl.sbuf)
        for r in range(rows):
            if dim_first:
                src = conv_state[c0 : c0 + bc, 0:channels, r]
            else:
                src = conv_state[c0 : c0 + bc, r, 0:channels]
            nisa.dma_copy(dst=hist[r * bc : (r + 1) * bc, 0:channels], src=src)
        rows32 = nl.ndarray((srows, channels), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rows32[0:hrows, 0:channels], src=hist)
        _load_token_rows(rows32, q_in, 0, hrows, srows, c0, bc, width)
        _load_token_rows(rows32, k_in, 1, hrows, srows, c0, bc, width)
        _load_token_rows(rows32, v_in, 2, hrows, srows, c0, bc, width)

        # Carrier taps are kept only where the request continues; the token tap
        # always. A predicated copy over zeros, so a NaN carrier cannot leak.
        tap_keep = nl.ndarray((kdim, taps, bc), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.memset(dst=tap_keep, value=1)
        nisa.tensor_copy(
            dst=tap_keep[0:kdim, 0:rows, 0:bc],
            src=keep.reshape((PMAX, 1, bc))[0:kdim].broadcast(1, rows),
        )
        xcol = nl.ndarray((kdim, blocks, taps, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=xcol, value=0.0)
        for nb in range(blocks):
            turned = nl.ndarray((kdim, srows), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=turned, data=rows32[0:srows, nb * kdim : (nb + 1) * kdim]
            )
            nisa.tensor_copy_predicated(
                dst=xcol[0:kdim, nb],
                src=turned.reshape((kdim, taps, bc)),
                predicate=tap_keep,
            )

        # ---- depthwise conv + silu ------------------------------------------ #
        prod = nl.ndarray((kdim, blocks, bc, taps), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=prod.permute((0, 1, 3, 2)),
            data1=xcol,
            data2=w32.broadcast(3, bc),
            op=nl.multiply,
        )
        acc = nl.ndarray((kdim, blocks, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=acc, data=prod, op=nl.add, axis=(3,))
        act = nl.ndarray((kdim, blocks, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=act, data=acc, op=nl.silu)

        # ---- the advanced conv carrier, in its own dtype and layout --------- #
        xb = nl.ndarray((kdim, blocks, taps, bc), dtype=conv_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xb, src=xcol)
        new_hist = nl.ndarray((kdim, blocks, rows, bc), dtype=conv_dtype, buffer=nl.sbuf)
        if real_tokens != None:  # noqa: E711
            nisa.tensor_copy(dst=new_hist, src=xb[0:kdim, 0:blocks, 0:rows, 0:bc])
            nisa.tensor_copy_predicated(
                dst=new_hist,
                src=xb[0:kdim, 0:blocks, 1:taps, 0:bc],
                predicate=shift.reshape((PMAX, 1, 1, bc))[0:kdim]
                .broadcast(1, blocks)
                .broadcast(2, rows),
            )
        else:
            nisa.tensor_copy(dst=new_hist, src=xb[0:kdim, 0:blocks, 1:taps, 0:bc])
        out_rows = nl.ndarray((hrows, channels), dtype=conv_dtype, buffer=nl.sbuf)
        for nb in range(blocks):
            back = nl.ndarray((hrows, kdim), dtype=conv_dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=back, data=new_hist[0:kdim, nb].reshape((kdim, hrows))
            )
            nisa.tensor_copy(
                dst=out_rows[0:hrows, nb * kdim : (nb + 1) * kdim], src=back
            )
        for r in range(rows):
            if dim_first:
                dst = conv_hbm[c0 : c0 + bc, nl.ds(prog * cpart, cpart), r]
            else:
                dst = conv_hbm[c0 : c0 + bc, r, nl.ds(prog * cpart, cpart)]
            nisa.dma_copy(
                dst=dst, src=out_rows[r * bc : (r + 1) * bc, nl.ds(prog * cpart, cpart)]
            )

        # ---- kn, qn: L2 norms over K, summed by a ones matmul --------------- #
        sq = nl.ndarray((kdim, 2 * heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=sq,
            data1=act[0:kdim, 0 : 2 * heads],
            data2=act[0:kdim, 0 : 2 * heads],
            op=nl.multiply,
        )
        sums = nl.ndarray((kdim, 2 * heads * bc), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=sums,
            stationary=ones,
            moving=sq.reshape((kdim, 2 * heads * bc)),
            is_stationary_onezero=True,
        )
        den = nl.ndarray((kdim, 2 * heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=den.reshape((kdim, 2 * heads * bc)),
            data=sums,
            op0=nl.add,
            operand0=L2_NORM_EPS,
        )
        inv_root = nl.ndarray((kdim, 2 * heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_root, data=den, op=nl.rsqrt)
        qkn = nl.ndarray((kdim, 2 * heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=qkn, data1=act[0:kdim, 0 : 2 * heads], data2=inv_root, op=nl.multiply
        )
        nisa.tensor_scalar(
            dst=qkn[0:kdim, 0:heads],
            data=qkn[0:kdim, 0:heads],
            op0=nl.multiply,
            operand0=scale,
        )

        # ---- gate: lower * sigmoid(exp(A_log) * (g + bias)) * mask ----------- #
        graw = nl.ndarray((bc, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=graw, src=raw_gate[c0 : c0 + bc, 0:width])
        squashed = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        for h in range(heads):
            gate_t = nl.ndarray((kdim, bc), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=gate_t, data=graw[0:bc, h * kdim : (h + 1) * kdim])
            biased = nl.ndarray((kdim, bc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=biased, data=gate_t, op0=nl.add, operand0=bias[0:kdim, h : h + 1]
            )
            nisa.activation(
                dst=squashed[0:kdim, h],
                data=biased,
                op=nl.sigmoid,
                scale=exp_a[0:kdim, h : h + 1],
            )
        gk = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gk, data=squashed, op0=nl.multiply, operand0=lower)
        beta_raw = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_raw,
            src=raw_beta.ap(
                pattern=[[0, kdim], [1, heads], [heads, bc]], offset=c0 * heads
            ),
        )
        beta = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=beta, data=beta_raw, op=nl.sigmoid)
        if row_mask != None:  # noqa: E711
            row = mask.reshape((PMAX, 1, bc))[0:kdim].broadcast(1, heads)
            gk_m = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gk_m, data1=gk, data2=row, op=nl.multiply)
            beta_m = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=beta_m, data1=beta, data2=row, op=nl.multiply)
            gk = gk_m
            beta = beta_m
        decay = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=decay, data=gk, op=nl.exp)
        # -beta * kn, the per-key factor of the rank-1 update.
        neg_bk = nl.ndarray((kdim, heads, bc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=neg_bk,
            data=beta,
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.multiply,
            operand1=qkn[0:kdim, heads : 2 * heads],
        )

        # ---- -v as rows (h, b) on partitions, for the one-hot accumulation -- #
        v_t = nl.ndarray((hb, vdim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=v_t, data=act[0:kdim, 2 * heads : 3 * heads].reshape((kdim, hb))
        )
        neg_v = nl.ndarray((hb, vdim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_v, data=v_t, op0=nl.multiply, operand0=-1.0)

        # ---- the states: this program's value rows of every request --------- #
        st = nl.ndarray((vpart, bc, heads, kdim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=st,
            src=rec_state[
                c0 : c0 + bc, 0:heads, nl.ds(prog * vpart, vpart), 0:kdim
            ].permute((2, 0, 1, 3)),
        )
        st_keep = nl.ndarray((vpart, bc, heads, kdim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=st_keep, value=0.0)
        nisa.tensor_copy_predicated(
            dst=st_keep,
            src=st,
            predicate=keep.reshape((PMAX, bc, 1, 1))[0:vpart]
            .broadcast(2, heads)
            .broadcast(3, kdim),
        )
        st_out = nl.ndarray((vpart, bc, heads, kdim), dtype=nl.float32, buffer=nl.sbuf)
        o_row = nl.ndarray((1, bc, heads, vpart), dtype=nl.float32, buffer=nl.sbuf)
        for b in range(bc):
            for h in range(heads):
                j = h * bc + b
                turned_in = nl.ndarray((kdim, vpart), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=turned_in, data=st_keep[0:vpart, b, h, 0:kdim])
                decayed = nl.ndarray((kdim, vpart), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=decayed,
                    data=turned_in,
                    op0=nl.multiply,
                    operand0=decay[0:kdim, h, b : b + 1],
                )
                # (S kn - v) on every partition: kn broadcast as the stationary,
                # then a one-hot row adds -v.
                resid = nl.ndarray((kdim, vpart), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=resid,
                    stationary=qkn[0:kdim, heads + h, b : b + 1].broadcast(1, kdim),
                    moving=decayed,
                    accumulate=False,
                )
                nisa.nc_matmul(
                    dst=resid,
                    stationary=eye[0:hb, j : j + 1].broadcast(1, kdim),
                    moving=neg_v[0:hb, nl.ds(prog * vpart, vpart)],
                    accumulate=True,
                    is_stationary_onezero=True,
                )
                updated = nl.ndarray((kdim, vpart), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=updated,
                    data=resid,
                    op0=nl.multiply,
                    operand0=neg_bk[0:kdim, h, b : b + 1],
                    op1=nl.add,
                    operand1=decayed,
                )
                o_ps = nl.ndarray((1, vpart), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=o_ps,
                    stationary=qkn[0:kdim, h, b : b + 1],
                    moving=updated,
                    accumulate=False,
                )
                nisa.tensor_copy(dst=o_row[0:1, b, h, 0:vpart], src=o_ps)
                turned_out = nl.ndarray((vpart, kdim), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=turned_out, data=updated)
                nisa.tensor_copy(dst=st_out[0:vpart, b, h, 0:kdim], src=turned_out)
        nisa.dma_copy(
            dst=rec_hbm[
                c0 : c0 + bc, 0:heads, nl.ds(prog * vpart, vpart), 0:kdim
            ].permute((2, 0, 1, 3)),
            src=st_out,
        )
        nisa.dma_copy(
            dst=core_hbm[c0 : c0 + bc, 0:width]
            .reshape((1, bc, heads, vdim))[0:1, 0:bc, 0:heads, nl.ds(prog * vpart, vpart)],
            src=o_row,
        )

    return core_hbm, rec_hbm, conv_hbm


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def fused_decode_enabled() -> bool:
    """Whether the model call site takes this kernel for a one-token decode."""
    return os.environ.get(FUSED_DECODE_ENV, "1") != "0"


def fused_decode_grid(vdim: int) -> tuple[int, ...]:
    """``(2,)`` on an LNC2 runtime when the value rows split evenly, else ``()``."""
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and vdim % 2 == 0:
        return (2,)
    return ()


def can_run_fused_decode(reference: Tensor) -> bool:
    """Is a device or the simulator available for the NKI route?"""
    return can_run_kernel(reference)


def _require(cond: bool, problems: list[str], message: str) -> None:
    if not cond:
        problems.append(message)


def _per_request(value, batch: int, dtype: torch.dtype, device, name: str):
    """A request-axis operand as a ``[1, B]`` tensor, or ``None``."""
    if value is None:
        return None
    if not torch.is_tensor(value):
        return torch.full((1, batch), value, dtype=dtype, device=device)
    if value.numel() != batch:
        raise KdaFusedDecodeError(
            f"{name} carries {value.numel()} value(s) for {batch} request(s); it "
            f"holds one per request"
        )
    return value.reshape(1, batch).to(device=device, dtype=dtype)


def kda_fused_decode(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    raw_gate: Tensor,
    raw_beta: Tensor,
    *,
    conv_state: Tensor,
    recurrent_state: Tensor,
    q_conv1d_weight: Tensor,
    k_conv1d_weight: Tensor,
    v_conv1d_weight: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    gate_lower_bound: float,
    conv_state_dim_first: bool,
    start_position: Tensor | int,
    real_tokens: Tensor | int | None = None,
    row_mask: Tensor | None = None,
) -> FusedDecodeOutputs:
    """One decode token for each of ``B`` requests, in one kernel dispatch.

    Args:
        q_in, k_in, v_in: ``[B, H*K]`` float32, this token's projections, before
            the conv.
        raw_gate: ``[B, H*K]`` float32, the low-rank gate projection.
        raw_beta: ``[B, H]`` float32, before its sigmoid.
        conv_state: ``[B, R, C]`` (``[B, C, R]`` when ``conv_state_dim_first``),
            the conv carrier in the bank's dtype; ``C = 3*H*K``.
        recurrent_state: ``[B, H, V, K]`` float32, index 2 the value extent.
        q_conv1d_weight, k_conv1d_weight, v_conv1d_weight: ``H*K * (R+1)``
            elements each, channel-major.
        A_log: ``[H]``; dt_bias: ``[H*K]``; gate_lower_bound: the checkpoint's.
        start_position: per-request position, ``0`` opens the sequence.
        real_tokens: per-request real-token count (0 for a padding request) or
            ``None`` for "every request carries its token".
        row_mask: ``[B, 1]`` float or ``None``; multiplies the gate and beta.

    Returns:
        :class:`FusedDecodeOutputs`. Neither input carrier is written; the caller
        copies the returned carriers into its bank.
    """
    problems: list[str] = []
    if q_in.dim() != 2:
        raise KdaFusedDecodeError(f"q_in must be [B, H*K], got {tuple(q_in.shape)}")
    batch, width = (int(x) for x in q_in.shape)
    if recurrent_state.dim() != 4:
        raise KdaFusedDecodeError(
            f"recurrent_state must be [B, H, V, K], got {tuple(recurrent_state.shape)}"
        )
    _, heads, vdim, kdim = (int(x) for x in recurrent_state.shape)
    taps = int(q_conv1d_weight.numel()) // max(width, 1)
    rows = taps - 1
    channels = 3 * width
    expected_conv = (batch, channels, rows) if conv_state_dim_first else (
        batch, rows, channels
    )
    _require(
        tuple(recurrent_state.shape) == (batch, heads, kdim, kdim)
        and heads * kdim == width,
        problems,
        f"recurrent_state {tuple(recurrent_state.shape)} must be "
        f"[{batch}, H, K, K] with H*K = {width}",
    )
    _require(
        recurrent_state.dtype == torch.float32,
        problems,
        f"recurrent_state dtype {recurrent_state.dtype} must be float32, the "
        f"recurrent carrier's",
    )
    _require(
        tuple(conv_state.shape) == expected_conv,
        problems,
        f"conv_state {tuple(conv_state.shape)} must be {expected_conv}",
    )
    _require(
        conv_state.dtype in (torch.bfloat16, torch.float32),
        problems,
        f"conv_state dtype {conv_state.dtype} must be bfloat16 or float32",
    )
    for name, tensor, shape in (
        ("k_in", k_in, (batch, width)),
        ("v_in", v_in, (batch, width)),
        ("raw_gate", raw_gate, (batch, width)),
        ("raw_beta", raw_beta, (batch, heads)),
    ):
        _require(
            tuple(tensor.shape) == shape,
            problems,
            f"{name} {tuple(tensor.shape)} must be {shape}",
        )
    for name, tensor in (
        ("q_conv1d_weight", q_conv1d_weight),
        ("k_conv1d_weight", k_conv1d_weight),
        ("v_conv1d_weight", v_conv1d_weight),
    ):
        _require(
            int(tensor.numel()) == width * taps and taps >= 2,
            problems,
            f"{name} holds {tensor.numel()} values; it must hold {width} channels "
            f"x {taps} taps",
        )
    _require(A_log.numel() == heads, problems, f"A_log must hold {heads} value(s)")
    _require(dt_bias.numel() == width, problems, f"dt_bias must hold {width} values")
    _require(
        1 <= kdim <= MAX_TILE,
        problems,
        f"head_dim={kdim} must be in [1, {MAX_TILE}]: it is a partition extent",
    )
    _require(
        taps <= PMAX and heads <= PMAX,
        problems,
        f"taps={taps} and heads={heads} must each fit {PMAX} partitions",
    )
    if problems:
        raise KdaFusedDecodeError(
            "kda_fused_decode cannot serve this call: " + "; ".join(problems)
        )

    device = q_in.device
    start = _per_request(start_position, batch, torch.int32, device, "start_position")
    real = _per_request(real_tokens, batch, torch.int32, device, "real_tokens")
    mask = _per_request(row_mask, batch, torch.float32, device, "row_mask")

    if not can_run_fused_decode(q_in):
        _count_torch_fallback()
        logger.debug(
            "kda_fused_decode: NKI route unavailable, using the torch reference "
            "(reference only, not the shipped path)"
        )
        return kda_fused_decode_torch_reference(
            q_in, k_in, v_in, raw_gate, raw_beta,
            conv_state=conv_state, recurrent_state=recurrent_state,
            q_conv1d_weight=q_conv1d_weight, k_conv1d_weight=k_conv1d_weight,
            v_conv1d_weight=v_conv1d_weight, A_log=A_log, dt_bias=dt_bias,
            gate_lower_bound=gate_lower_bound,
            conv_state_dim_first=conv_state_dim_first,
            start_position=start, real_tokens=real, row_mask=mask,
        )

    _count_nki_dispatch()
    call = wrap_nki(kda_fused_decode_kernel)
    grid = fused_decode_grid(vdim)
    if grid:
        call = call[grid]
    core, rec_out, conv_out = call(
        q_in=q_in.to(torch.float32).contiguous(),
        k_in=k_in.to(torch.float32).contiguous(),
        v_in=v_in.to(torch.float32).contiguous(),
        raw_gate=raw_gate.to(torch.float32).contiguous(),
        raw_beta=raw_beta.to(torch.float32).contiguous(),
        conv_state=conv_state.contiguous(),
        q_w=q_conv1d_weight.reshape(width, taps).contiguous(),
        k_w=k_conv1d_weight.reshape(width, taps).contiguous(),
        v_w=v_conv1d_weight.reshape(width, taps).contiguous(),
        a_log=A_log.reshape(1, heads).contiguous(),
        dt_bias=dt_bias.reshape(1, width).contiguous(),
        rec_state=recurrent_state.contiguous(),
        start_pos=start.contiguous(),
        real_tokens=None if real is None else real.contiguous(),
        row_mask=None if mask is None else mask.contiguous(),
        lower=float(gate_lower_bound),
        dim_first=bool(conv_state_dim_first),
    )
    return FusedDecodeOutputs(core=core, conv_state=conv_out, recurrent_state=rec_out)


def kda_fused_decode_torch_reference(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    raw_gate: Tensor,
    raw_beta: Tensor,
    *,
    conv_state: Tensor,
    recurrent_state: Tensor,
    q_conv1d_weight: Tensor,
    k_conv1d_weight: Tensor,
    v_conv1d_weight: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    gate_lower_bound: float,
    conv_state_dim_first: bool,
    start_position: Tensor,
    real_tokens: Tensor | None = None,
    row_mask: Tensor | None = None,
) -> FusedDecodeOutputs:
    """The same step in torch: the counted fallback, never the shipped path.

    ``start_position``/``real_tokens``/``row_mask`` are the ``[1, B]`` forms
    :func:`kda_fused_decode` builds.
    """
    batch, width = (int(x) for x in q_in.shape)
    _, heads, vdim, kdim = (int(x) for x in recurrent_state.shape)
    taps = int(q_conv1d_weight.numel()) // width
    rows = taps - 1
    opening = (start_position.reshape(batch) == 0).reshape(batch, 1, 1)

    stored = conv_state.transpose(1, 2) if conv_state_dim_first else conv_state
    history = torch.where(
        opening,
        torch.zeros((), dtype=torch.float32, device=q_in.device),
        stored.to(torch.float32),
    )
    token = torch.cat((q_in, k_in, v_in), dim=-1).to(torch.float32)
    padded = torch.cat((history, token.unsqueeze(1)), dim=1)  # [B, taps, C]
    filt = torch.cat(
        (
            q_conv1d_weight.to(torch.float32).reshape(width, taps),
            k_conv1d_weight.to(torch.float32).reshape(width, taps),
            v_conv1d_weight.to(torch.float32).reshape(width, taps),
        ),
        dim=0,
    )
    act = torch.nn.functional.silu((padded * filt.t().unsqueeze(0)).sum(dim=1))
    if real_tokens is None:
        new_rows = padded[:, 1:]
    else:
        moved = (real_tokens.reshape(batch) != 0).reshape(batch, 1, 1)
        new_rows = torch.where(moved, padded[:, 1:], padded[:, :rows])
    new_rows = new_rows.to(conv_state.dtype)
    conv_out = new_rows.transpose(1, 2) if conv_state_dim_first else new_rows

    q, k, v = (part.reshape(batch, heads, kdim) for part in act.split(width, dim=-1))
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)
    exp_a = torch.exp(A_log.to(torch.float32).reshape(1, heads, 1))
    biased = raw_gate.to(torch.float32).reshape(batch, heads, kdim) + dt_bias.to(
        torch.float32
    ).reshape(1, heads, kdim)
    gk = torch.sigmoid(biased * exp_a) * float(gate_lower_bound)
    beta = torch.sigmoid(raw_beta.to(torch.float32))
    if row_mask is not None:
        gk = gk * row_mask.reshape(batch, 1, 1)
        beta = beta * row_mask.reshape(batch, 1)

    state = torch.where(
        opening.reshape(batch, 1, 1, 1),
        torch.zeros((), dtype=torch.float32, device=q_in.device),
        recurrent_state.to(torch.float32),
    )
    state = state * torch.exp(gk).unsqueeze(2)
    delta = (v - (state @ kn.unsqueeze(-1)).squeeze(-1)) * beta.unsqueeze(-1)
    state = state + delta.unsqueeze(-1) * kn.unsqueeze(2)
    o = (state @ qn.unsqueeze(-1)).squeeze(-1)
    return FusedDecodeOutputs(
        core=o.reshape(batch, heads * vdim).contiguous(),
        conv_state=conv_out.contiguous(),
        recurrent_state=state.contiguous(),
    )


def fused_decode_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel, read off the unwrapped ``.func``."""
    func = getattr(kda_fused_decode_kernel, "func", None)
    target = func if func is not None else kda_fused_decode_kernel
    return target.__module__, target.__qualname__
