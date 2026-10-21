# SPDX-License-Identifier: Apache-2.0
"""The KDA input projections in one NKI launch, at decode and at prefill.

The torch route (``Glm5NextKDAAttention.forward``, and
``_fused_decode_requests``) computes, with ``x = hidden_states.float()``::

    q_in, k_in, v_in = x @ q_w.float().t(), x @ k_w.float().t(), x @ v_w.float().t()
    raw_beta = x @ b_w.float().t()
    raw_gate = (x @ f_a_w.float().t()) @ f_b_w.float().t()
    out_gate = (x @ g_a_w.float().t()) @ g_b_w.float().t()

so every call casts and transposes 641 x 4096 weight values to fp32 and runs fp32
matmuls on them. The kernels read bf16 activations and weights instead:

* :func:`kda_projections_kernel` (``T <= KDA_PROJECTIONS_MAX_TOKENS``) reads the eight
  weights as stored. fp32 operands (the CPU fixtures' dtype) run the same steps on fp32
  tiles, with the weights' transposes on the tensor engine.
* :func:`kda_projections_prefill_kernel` (a larger ``T`` in 128-row tiles per program,
  bf16 only) reads the weights contraction-major, as :func:`prefill_operands` lays
  them out once at load (the model's ``prepare_projection_weights`` stores them on the
  module under :data:`PREFILL_OPERANDS_ATTR`). Each program takes half of the rows and
  all of the weights, so the programs exchange nothing; the method is in its
  docstring.

Method of the decode kernel. Both programs (``P`` under LNC2) take ``H / P`` of the
contraction:

1. ``x^T`` by tensor-engine transposes; the weights' transposes ``W^T[h, r]`` are
   loaded straight from HBM by DMA transposes (16 weight rows per DMA, the hardware
   DGE transpose shape), or, with ``DMA_TRANSPOSE=False``, by a plain DMA and
   tensor-engine transposes.
2. ``[T, 641]`` first-stage partials by bf16 matmuls (bf16 x bf16 is exact in the fp32
   accumulator: the fp32 reference up to summation order), accumulated over this
   program's 128-row blocks, then swapped between the programs and added.
3. The low-rank second stage on the fp32 bottleneck values, as an fp32 matmul (the
   reference's own precision): program 0 forms ``raw_gate``, program 1 ``out_gate``
   (one program forms both).

Both LNC2 cores: yes, in both kernels. The decode kernel's steps 1-2 split the
contraction and step 3 splits the two gates; the prefill kernel splits the rows.
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

from vllm_neuron import envs
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.functional.glue import glue_selected
from vllm_neuron.functional.kda.gate_clamp import kda_gate
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Largest token count the decode kernel serves (the activations' transpose is its
#: matmul stationary); larger calls take the prefill kernel.
KDA_PROJECTIONS_MAX_TOKENS = 128

#: The module attribute the load-time prep stores the prefill kernel's operands on:
#: ``(wt, expand_t)`` from :func:`prefill_operands`. A plain attribute, not a buffer,
#: so ``state_dict()`` does not carry a second copy of the weights.
PREFILL_OPERANDS_ATTR = "_prepared_prefill_projection_operands"

#: The switch (registered in :mod:`vllm_neuron.envs`): ``1`` loads the weights'
#: transposes by DMA transpose, ``0`` by a plain DMA and tensor-engine transposes.
DMA_TRANSPOSE_ENV = "VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE"

_P = 128
_BANK = 512  # fp32 columns per PSUM bank
_DGE_ROWS = 16  # weight rows per hardware-DGE transpose
_BANK_BYTES = 2048  # bytes per partition of one PSUM bank
_XBAR_STEP = 16  # a DMA transpose's output row stride: a multiple of 32 bytes of bf16
_PER_BF16_BANK = 8  # 128-column bf16 transposes per PSUM bank
_WEIGHT_BLOCKS_PER_DMA = 4  # contraction blocks of the prefill weights per load DMA

__all__ = [
    "KDA_PROJECTIONS_MAX_TOKENS",
    "PREFILL_OPERANDS_ATTR",
    "dispatch_counters",
    "input_rms_norm",
    "kda_projections",
    "kda_projections_admits",
    "kda_projections_kernel",
    "kda_projections_prefill_kernel",
    "prefill_operands",
    "reset_dispatch_counters",
]


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _even(n):
    """``n`` rounded up to even: a bf16 PSUM slot must start 4-byte aligned."""
    return n + (n % 2)


def _per_bank(dtype, cols):
    """How many ``cols``-wide transposes of ``dtype`` fill one PSUM bank (at least 1)."""
    size = 2 if dtype == nl.bfloat16 else 4
    return max(1, _BANK_BYTES // (size * max(cols, 1)))


def _load_transposed(dst, weight, h0, nkb, use_dma):
    """``dst[c, kb, r] = weight[r, h0 + kb*128 + c]`` for this program's ``nkb`` blocks.

    ``dst`` may be wider than ``weight`` (fp32 from bf16): the copy out of PSUM widens.
    """
    rows, hidden = weight.shape
    if use_dma:
        for r0 in range(0, rows, _DGE_ROWS):
            n = min(_DGE_ROWS, rows - r0)
            nisa.dma_transpose(
                dst=dst[:, :, r0:r0 + n],
                src=weight.ap(pattern=[[hidden, n], [_P, nkb], [1, _P]],
                              offset=r0 * hidden + h0))
        return
    natural = nl.ndarray((rows, nkb * _P), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=natural, src=weight.ap(pattern=[[hidden, rows], [1, nkb * _P]],
                                             offset=h0))
    slot = _even(rows)
    per = _per_bank(weight.dtype, slot)
    for g0 in range(0, nkb, per):
        gn = min(per, nkb - g0)
        flipped = nl.ndarray((_P, gn, slot), dtype=weight.dtype, buffer=nl.psum)
        for j in range(gn):
            nisa.nc_transpose(dst=flipped[:, j, 0:rows],
                              data=natural[:, (g0 + j) * _P:(g0 + j + 1) * _P])
        nisa.tensor_copy(dst=dst[:, g0:g0 + gn, 0:rows], src=flipped[:, :, 0:rows])


@nki.jit
def kda_projections_kernel(x, q_w, k_w, v_w, b_w, f_a_w, f_b_w, g_a_w, g_b_w,
                           DMA_TRANSPOSE: bool = True):
    """``(q_in, k_in, v_in, raw_gate, raw_beta, out_gate)``, fp32, ``[T, *]``.

    Every operand is bf16 (served) or fp32; the first stage runs in bf16 when all of
    ``x`` and the six first-stage weights are bf16, else in fp32.

    Args:
        x: ``[T, H]`` activations (the normed attention input), ``T <= 128``.
        q_w, k_w, v_w: ``[W, H]``, ``W <= 128`` (heads x head_dim on this rank).
        b_w: ``[heads, H]``.
        f_a_w, g_a_w: ``[R, H]`` bottlenecks, ``R <= 128``.
        f_b_w, g_b_w: ``[W, R]`` expansions.
        DMA_TRANSPOSE: load the weights' transposes by DMA (else tensor engine); bf16
            weights only, so fp32 ones always take the tensor engine.
    """
    tokens, hidden = x.shape
    width = q_w.shape[0]
    heads = b_w.shape[0]
    rank = f_a_w.shape[0]
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(1 <= tokens <= KDA_PROJECTIONS_MAX_TOKENS, "1 <= T <= 128")
    kernel_assert(width <= _P and rank <= _P and heads <= _P, "W, R, heads <= 128")
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(hidden % (_P * programs) == 0, "H splits into 128-blocks per program")
    hp = hidden // programs
    h0 = program * hp
    nkb = hp // _P

    # First-stage columns: q | k | v | f_a | g_a | b.
    stage1 = [q_w, k_w, v_w, f_a_w, g_a_w, b_w]
    starts = [0, width, 2 * width, 3 * width, 3 * width + rank, 3 * width + 2 * rank]
    cols = 3 * width + 2 * rank + heads
    all_bf16 = (x.dtype == nl.bfloat16 and q_w.dtype == nl.bfloat16
                and k_w.dtype == nl.bfloat16 and v_w.dtype == nl.bfloat16
                and b_w.dtype == nl.bfloat16 and f_a_w.dtype == nl.bfloat16
                and g_a_w.dtype == nl.bfloat16)
    work = nl.bfloat16 if all_bf16 else nl.float32
    use_dma = DMA_TRANSPOSE and all_bf16

    q_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    k_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    v_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    gate_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    beta_out = nl.ndarray((tokens, heads), dtype=nl.float32, buffer=nl.shared_hbm)
    og_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- 1. x^T [128, nkb, T] and W^T [128, nkb, cols]. -------------------- #
    x_rows = nl.ndarray((tokens, hp), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_rows, src=x.ap(pattern=[[hidden, tokens], [1, hp]], offset=h0))
    xt = nl.ndarray((_P, nkb, tokens), dtype=work, buffer=nl.sbuf)
    slot = _even(tokens)
    per = _per_bank(x.dtype, slot)
    for g0 in range(0, nkb, per):
        gn = min(per, nkb - g0)
        flipped = nl.ndarray((_P, gn, slot), dtype=x.dtype, buffer=nl.psum)
        for j in range(gn):
            nisa.nc_transpose(dst=flipped[:, j, 0:tokens],
                              data=x_rows[:, (g0 + j) * _P:(g0 + j + 1) * _P])
        nisa.tensor_copy(dst=xt[:, g0:g0 + gn, :], src=flipped[:, :, 0:tokens])
    # Row stride padded to a multiple of _XBAR_STEP for the DMA transposes' output.
    wt = nl.ndarray((_P, nkb, -(-cols // _XBAR_STEP) * _XBAR_STEP), dtype=work,
                    buffer=nl.sbuf)
    for i in range(6):
        c0 = starts[i]
        n = stage1[i].shape[0]
        _load_transposed(wt[:, :, c0:c0 + n], stage1[i], h0, nkb, use_dma)

    # ---- 2. First stage over this program's blocks, then the other half. ---- #
    mine = _tile(tokens, cols)
    for c0 in range(0, cols, _BANK):
        cn = min(_BANK, cols - c0)
        acc = nl.ndarray((tokens, cn), dtype=nl.float32, buffer=nl.psum)
        for kb in range(nkb):
            nisa.nc_matmul(dst=acc, stationary=xt[:, kb, :], moving=wt[:, kb, c0:c0 + cn],
                           accumulate=(kb > 0))
        nisa.tensor_copy(dst=mine[:, c0:c0 + cn], src=acc)
    if programs == 2:
        theirs = _tile(tokens, cols)
        nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - program,
                      recv_from_rank=1 - program, pipe_id=0)
        total = _tile(tokens, cols)
        nisa.tensor_tensor(dst=total, data1=mine, data2=theirs, op=nl.add)
    else:
        total = mine

    if program == 0:
        nisa.dma_copy(dst=q_out, src=total[:, 0:width])
        nisa.dma_copy(dst=k_out, src=total[:, width:2 * width])
        nisa.dma_copy(dst=v_out, src=total[:, 2 * width:3 * width])
        nisa.dma_copy(dst=beta_out, src=total[:, starts[5]:cols])

    # ---- 3. Low-rank gates, fp32: program 0 raw_gate, the last out_gate. ---- #
    expands = [f_b_w, g_b_w]
    gate_outs = [gate_out, og_out]
    for which in range(2):
        if programs == 1 or program == which:
            low = starts[3 + which]
            expand = expands[which]
            out = gate_outs[which]
            # bottleneck^T [R, T], fp32.
            bt_ps = nl.ndarray((rank, tokens), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=bt_ps, data=total[:, low:low + rank])
            bt = _tile(rank, tokens)
            nisa.tensor_copy(dst=bt, src=bt_ps)
            # expand^T [R, W] as fp32 (a bf16 expansion's values, exactly).
            e_rows = nl.ndarray((width, rank), dtype=expand.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=e_rows, src=expand)
            et_ps = nl.ndarray((rank, width), dtype=expand.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=et_ps, data=e_rows)
            et = nl.ndarray((rank, width), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=et, src=et_ps)
            g_ps = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=g_ps, stationary=bt, moving=et, accumulate=False)
            g_sb = _tile(tokens, width)
            nisa.tensor_copy(dst=g_sb, src=g_ps)
            nisa.dma_copy(dst=out, src=g_sb)
    return q_out, k_out, v_out, gate_out, beta_out, og_out


def _rms_norm_rows(rows, gain_rows, eps):
    """Norm ``rows`` in place: ``bf16((x * rsqrt(mean(x^2) + eps)) * gain)``, fp32 inside.

    Args:
        rows: ``[128, H]`` bf16 SBUF rows of the un-normed layer input; they are
            overwritten with the normed rows.
        gain_rows: ``[128, H]`` bf16 SBUF tile, the norm gain on every partition.
        eps: the norm epsilon.

    The steps are :func:`input_rms_norm`'s, in its order: the squares summed in fp32
    (Scalar, one pass), the mean as a multiply by ``1 / H`` (exact for the power-of-two
    ``H`` the model has) plus ``eps``, rsqrt (both on GpSimd, the higher-precision
    engine for rsqrt, as in ``kda_output``), then ``x * rstd`` and ``* gain`` in fp32
    with one bf16 rounding (Vector, one pass).

    The mean and ``eps`` are on GpSimd, not on the default engine: the compiler
    gives that step to Scalar, where it waits behind the squares of every later
    tile, and the Vector pass of the first tile waits with it.
    """
    parts, hidden = rows.shape
    squares = nl.ndarray((parts, hidden), dtype=nl.float32, buffer=nl.sbuf)
    rstd = _tile(parts, 1)
    nisa.activation_reduce(dst=squares, op=nl.square, data=rows, reduce_op=nl.add,
                           reduce_res=rstd)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.multiply, operand0=1.0 / float(hidden),
                       op1=nl.add, operand1=float(eps), engine=nisa.engine.gpsimd)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    nisa.scalar_tensor_tensor(dst=rows, data=rows, op0=nl.multiply, operand0=rstd,
                              op1=nl.multiply, operand1=gain_rows)


def _gate_rows(dst, raw, bias_rows, decay, lower):
    """``dst = lower * sigmoid(exp(a_log_h) * (raw + bias))`` on one head's token rows.

    Args:
        dst: ``[128, K]`` fp32 SBUF tile, head ``h``'s gate rows.
        raw: ``[128, K]`` fp32 PSUM tile, the same rows of ``raw_gate``.
        bias_rows: ``[128, K]`` fp32 SBUF tile, head ``h``'s gate bias on every partition.
        decay: ``[128, 1]`` fp32 SBUF tile, ``exp(a_log_h)`` on every partition.
        lower: the checkpoint's ``gate_lower_bound``.

    The steps of :func:`~vllm_neuron.functional.kda.gate_clamp.kda_gate_clamp_kernel`, in
    its order and in fp32: the bias added (here on the PSUM read, Vector), the columns
    scaled by the head's decay rate, the sigmoid (Scalar), then the bound as a factor.
    That kernel holds its tile transposed, so that the per-channel bias is a
    per-partition scalar. Here the partitions are tokens: the bias is a row on every
    partition and the decay rate is the per-partition scalar, so no transpose is paid.
    """
    nisa.tensor_tensor(dst=dst, data1=raw, data2=bias_rows, op=nl.add,
                       engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=dst, data=dst, op0=nl.multiply, operand0=decay)
    nisa.activation(dst=dst, data=dst, op=nl.sigmoid)
    nisa.tensor_scalar(dst=dst, data=dst, op0=nl.multiply, operand0=lower)


def _psum_reader(index):
    """The engine of the ``index``-th PSUM read: Vector and Scalar (the two that read PSUM)
    take turns."""
    return nisa.engine.vector if index % 2 == 0 else nisa.engine.scalar


@nki.jit
def kda_projections_prefill_kernel(x, wt, expand_t, gain=None, EPS=None, a_log=None,
                                   gate_bias=None, LOWER=None):
    """``(q_in, k_in, v_in, raw_gate, raw_beta, out_gate)``, fp32, ``[T, *]`` (prefill shapes).

    With the gate operands the fourth output is the KDA gate instead of ``raw_gate``.

    Args:
        x: ``[T, H]`` bf16 activations, ``T`` a multiple of ``128 * programs``.
        wt: ``[H, C]`` bf16, the six first-stage weights contraction-major, columns
            ``q | k | v | b`` of each head in turn, then ``f_a | g_a``
            (``C = 3 W + heads + 2 R``), as :func:`prefill_operands` lays them out.
        expand_t: ``[R, 2 W]`` bf16, the two expansions transposed, ``f_b^T | g_b^T``.
        gain: None, or ``[H]`` bf16, the layer's input RMSNorm gain. Then ``x`` is the
            un-normed layer input and each row is normed on chip first, as
            :func:`input_rms_norm` does (:func:`_rms_norm_rows`), so the normed
            activations never go to HBM.
        EPS: None, or the input RMSNorm epsilon, the checkpoint's ``rms_norm_eps``.
            Given with ``gain`` and only with it; read only with ``gain``.
        a_log: None, or ``[1, heads]`` fp32, each head's decay exponent ``A_log``. Given
            with ``gate_bias`` and ``LOWER``, and only with them. Then each gate tile is
            gated on chip before it is stored (:func:`_gate_rows`), so ``raw_gate`` never
            goes to HBM.
        gate_bias: None, or ``[1, W]`` fp32, the per-key-channel gate bias (``dt_bias``).
        LOWER: None, or the checkpoint's ``gate_lower_bound``, a compile-time scalar.

    Each program takes ``T / programs`` contiguous rows, in groups of up to 512 tokens
    (the bottlenecks' moving operand fills one fp32 PSUM bank). Per group: ``x^T`` by
    tensor-engine transposes (``x`` arrives row-major and the tensor engine contracts
    over the partitions), then, contraction block by block, ``x @ [q|k|v|b]^T`` with
    ``x^T`` stationary (tokens on the output partitions, the call's layout) and the two
    bottlenecks ``f_a x^T``, ``g_a x^T`` with the weight stationary (the bottleneck on
    the partitions, which is the second stage's stationary layout, so it needs no
    transpose). The second stage is an fp32 matmul on the fp32 bottleneck, the
    reference's own precision.

    A rank that holds several heads' shards runs the first stage head by head, one
    head's ``q | k | v | b`` columns per pass (the bottlenecks, replicated and ``R``
    wide whatever the head count, beside the first pass), and the second stage head by
    head too. Each head's matmuls are then the instructions a one-head call on that
    head's weights issues, on the same operand values, so each head's outputs are that
    call's.
    """
    tokens, hidden = x.shape
    contraction, cols = wt.shape
    rank, expand_cols = expand_t.shape
    width = expand_cols // 2
    heads = cols - 3 * width - 2 * rank
    kdim = width // max(heads, 1)  # one head's q, k, v and gate columns
    direct = 3 * kdim + 1  # one head's q | k | v | b columns
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(contraction == hidden and hidden % _P == 0, "wt is [H, C], H in 128-blocks")
    kernel_assert(tokens % (_P * programs) == 0, "T splits into 128-row tiles per program")
    kernel_assert(rank <= _P and heads >= 1 and kdim * heads == width and direct <= _BANK,
                  "R <= 128, W in heads blocks, one head's q|k|v|b in one PSUM bank")
    kernel_assert(x.dtype == nl.bfloat16 and wt.dtype == nl.bfloat16
                  and expand_t.dtype == nl.bfloat16, "bf16 operands")
    kernel_assert(gain is None or (tuple(gain.shape) == (hidden,) and gain.dtype == nl.bfloat16),
                  "gain is [H] bf16")
    kernel_assert((gain is None) == (EPS is None), "EPS is given with gain, and only with it")
    gated = a_log is not None
    kernel_assert(gated == (gate_bias is not None) == (LOWER is not None),
                  "a_log, gate_bias and LOWER are given together")
    kernel_assert(not gated or (tuple(a_log.shape) == (1, heads) and a_log.dtype == nl.float32
                                and tuple(gate_bias.shape) == (1, width)
                                and gate_bias.dtype == nl.float32 and width % heads == 0),
                  "a_log is [1, heads] fp32, gate_bias [1, W] fp32, W in heads blocks")
    rows = tokens // programs
    r0 = program * rows
    blocks = hidden // _P

    q_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    k_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    v_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    gate_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    beta_out = nl.ndarray((tokens, heads), dtype=nl.float32, buffer=nl.shared_hbm)
    og_out = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.shared_hbm)
    direct_outs = [q_out, k_out, v_out, beta_out]
    direct_cols = [kdim, kdim, kdim, 1]  # one head's columns of each
    out_cols = [width, width, width, heads]
    gate_outs = [gate_out, og_out]

    # ---- Weights, once per program: wt_sb[p, kb, c] = wt[128 kb + p, c]. ---- #
    wt_sb = nl.ndarray((_P, blocks, cols), dtype=wt.dtype, buffer=nl.sbuf)
    for k0 in range(0, blocks, _WEIGHT_BLOCKS_PER_DMA):
        kn = min(_WEIGHT_BLOCKS_PER_DMA, blocks - k0)
        nisa.dma_copy(dst=wt_sb[:, k0:k0 + kn, :],
                      src=wt.ap(pattern=[[cols, _P], [_P * cols, kn], [1, cols]],
                                offset=k0 * _P * cols))
    e_rows = nl.ndarray((rank, expand_cols), dtype=expand_t.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=e_rows, src=expand_t)
    expand = nl.ndarray((rank, expand_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=expand, src=e_rows)  # exact: bf16 values in fp32
    if gain is not None:
        # The norm gain on every partition: one stride-0 read of the [H] vector.
        gain_rows = nl.ndarray((_P, hidden), dtype=gain.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gain_rows, src=gain.ap(pattern=[[0, _P], [1, hidden]], offset=0))
    if gated:
        # The gate's per-channel bias and each head's decay rate on every partition, by
        # stride-0 reads: the rows here are tokens, so both are the same on every row.
        bias_rows = nl.ndarray((_P, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_rows, src=gate_bias.ap(pattern=[[0, _P], [1, width]], offset=0))
        a_rows = _tile(_P, heads)
        nisa.dma_copy(dst=a_rows, src=a_log.ap(pattern=[[0, _P], [1, heads]], offset=0))
        decay = _tile(_P, heads)
        nisa.activation(dst=decay, data=a_rows, op=nl.exp)

    reads = 0
    for g0 in range(0, rows, _BANK):
        span = min(_BANK, rows - g0)
        tiles = span // _P
        t0 = r0 + g0
        # ---- x^T [128, blocks, span] by tensor-engine transposes. ---------- #
        x_rows = nl.ndarray((_P, tiles, hidden), dtype=x.dtype, buffer=nl.sbuf)
        for i in range(tiles):
            nisa.dma_copy(dst=x_rows[:, i, :],
                          src=x.ap(pattern=[[hidden, _P], [1, hidden]],
                                   offset=(t0 + i * _P) * hidden))
            if gain is not None:
                _rms_norm_rows(x_rows[:, i, :], gain_rows, EPS)
        xt = nl.ndarray((_P, blocks, span), dtype=x.dtype, buffer=nl.sbuf)
        for i in range(tiles):
            for k0 in range(0, blocks, _PER_BF16_BANK):
                kn = min(_PER_BF16_BANK, blocks - k0)
                flipped = nl.ndarray((_P, kn, _P), dtype=x.dtype, buffer=nl.psum)
                for j in range(kn):
                    nisa.nc_transpose(dst=flipped[:, j, :],
                                      data=x_rows[:, i, (k0 + j) * _P:(k0 + j + 1) * _P])
                nisa.tensor_copy(dst=xt[:, k0:k0 + kn, i * _P:(i + 1) * _P], src=flipped,
                                 engine=_psum_reader(reads))
                reads += 1

        # ---- First stage, contraction-outer: one PSUM bank per accumulation group, #
        # one head's q | k | v | b per pass, the bottlenecks beside the first pass.
        acc_low = []
        for head in range(heads):
            p0 = head * direct
            acc_direct = []
            for i in range(tiles):
                acc_direct.append(nl.ndarray((_P, _BANK), dtype=nl.float32, buffer=nl.psum))
            if head == 0:
                for which in range(2):
                    acc_low.append(nl.ndarray((rank, _BANK), dtype=nl.float32,
                                              buffer=nl.psum))
            for kb in range(blocks):
                for i in range(tiles):
                    nisa.nc_matmul(dst=acc_direct[i][:, 0:direct],
                                   stationary=xt[:, kb, i * _P:(i + 1) * _P],
                                   moving=wt_sb[:, kb, p0:p0 + direct], accumulate=(kb > 0))
                if head == 0:
                    for which in range(2):
                        low = heads * direct + which * rank
                        nisa.nc_matmul(dst=acc_low[which][:, 0:span],
                                       stationary=wt_sb[:, kb, low:low + rank],
                                       moving=xt[:, kb, :], accumulate=(kb > 0))

            # ---- This head's q | k | v | b rows out. ---------------------- #
            for i in range(tiles):
                res = nl.ndarray((_P, direct), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=res, src=acc_direct[i][:, 0:direct],
                                 engine=_psum_reader(reads))
                reads += 1
                for which in range(4):
                    c0 = which * kdim
                    cn = direct_cols[which]
                    row = out_cols[which]
                    nisa.dma_copy(dst=direct_outs[which].ap(
                                      pattern=[[row, _P], [1, cn]],
                                      offset=(t0 + i * _P) * row + head * cn),
                                  src=res[:, c0:c0 + cn])

        # ---- Second stage, fp32: (bottleneck^T)^T @ expand^T, per 128-row tile ---- #
        # and head.
        for which in range(2):
            bt = nl.ndarray((rank, span), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bt, src=acc_low[which][:, 0:span],
                             engine=_psum_reader(reads))
            reads += 1
            for i in range(tiles):
                for head in range(heads):
                    h0 = head * kdim
                    e0 = which * width + h0
                    g_ps = nl.ndarray((_P, kdim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=g_ps, stationary=bt[:, i * _P:(i + 1) * _P],
                                   moving=expand[:, e0:e0 + kdim], accumulate=False)
                    g_sb = nl.ndarray((_P, kdim), dtype=nl.float32, buffer=nl.sbuf)
                    if gated and which == 0:
                        _gate_rows(g_sb, g_ps, bias_rows[:, h0:h0 + kdim],
                                   decay[:, head:head + 1], LOWER)
                    else:
                        nisa.tensor_copy(dst=g_sb, src=g_ps, engine=_psum_reader(reads))
                    reads += 1
                    nisa.dma_copy(dst=gate_outs[which].ap(pattern=[[width, _P], [1, kdim]],
                                                          offset=(t0 + i * _P) * width + h0),
                                  src=g_sb)
    return q_out, k_out, v_out, gate_out, beta_out, og_out


_KERNELS = {1: wrap_nki(kda_projections_kernel), 2: wrap_nki(kda_projections_kernel)[2]}
_PREFILL_KERNELS = {1: wrap_nki(kda_projections_prefill_kernel),
                    2: wrap_nki(kda_projections_prefill_kernel)[2]}


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


def _weights(attn) -> tuple[Tensor, ...]:
    return (attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight,
            attn.b_proj_weight, attn.f_a_proj_weight, attn.f_b_proj_weight,
            attn.g_a_proj_weight, attn.g_b_proj_weight)


def prefill_operands(q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, b_proj: Tensor,
                     f_a_proj: Tensor, f_b_proj: Tensor, g_a_proj: Tensor,
                     g_b_proj: Tensor) -> tuple[Tensor, Tensor]:
    """The prefill kernel's two operands, dense, from the eight ``[out, in]`` weights.

    Returns:
        ``(wt, expand_t)``: ``wt [H, C]`` is, head by head, that head's rows of
        ``q | k | v | b``, then ``f_a | g_a``, stacked on the output axis and
        transposed (``C = 3 W + heads + 2 R``); ``expand_t [R, 2 W]`` is
        ``f_b^T | g_b^T``. Same dtype as the weights. The head count is ``b``'s row
        count; with one head ``wt`` is ``q | k | v | b | f_a | g_a``.

    Head by head, so that each head's first-stage columns are one contiguous run, the
    run a one-head rank holds: the prefill kernel's pass over one head is then a
    one-head call's matmul. It runs where the weights are; the load-time prep passes
    host copies, because a device tensor refuses the transposed copy.

    Raises:
        ValueError: ``q``'s rows do not split evenly over the heads.
    """
    heads = int(b_proj.shape[0])
    kdim = int(q_proj.shape[0]) // heads
    if kdim * heads != int(q_proj.shape[0]):
        raise ValueError(f"q's {int(q_proj.shape[0])} rows do not split over {heads} heads")
    per_head = []
    for head in range(heads):
        rows = slice(head * kdim, (head + 1) * kdim)
        per_head += [q_proj[rows], k_proj[rows], v_proj[rows], b_proj[head:head + 1]]
    wt = torch.cat(per_head + [f_a_proj, g_a_proj], dim=0)
    return wt.t().contiguous(), torch.cat((f_b_proj.t(), g_b_proj.t()), dim=1).contiguous()


def _serves_prefill(hidden_states: Tensor, attn, width: int, heads: int, rank: int) -> bool:
    """True when the prefill kernel takes this call (above the decode bound).

    It needs bf16 activations in 128-row tiles per program and the operands the
    load-time prep stored, of the shapes :func:`prefill_operands` builds from these
    weights; one head's ``q | k | v | b`` must fit one fp32 PSUM bank.
    """
    tokens, hidden = (int(v) for v in hidden_states.shape)
    operands = getattr(attn, PREFILL_OPERANDS_ATTR, None)
    if operands is None or heads < 1:
        return False
    wt, expand_t = operands
    kdim = width // heads
    return (tokens > KDA_PROJECTIONS_MAX_TOKENS and tokens % (_P * launch_programs()) == 0
            and hidden % _P == 0 and kdim * heads == width and 3 * kdim + 1 <= _BANK
            and hidden_states.dtype == wt.dtype == expand_t.dtype == torch.bfloat16
            and tuple(wt.shape) == (hidden, 3 * width + heads + 2 * rank)
            and tuple(expand_t.shape) == (rank, 2 * width))


def kda_projections_admits(hidden_states: Tensor, attn, phase: str | None = None) -> bool:
    """True when :func:`kda_projections` serves this call on a kernel.

    Args:
        hidden_states: ``[T, H]`` activations, bf16 (served) or fp32.
        attn: the KDA attention module; its eight projection weights are read, and
            at prefill the operands its load-time prep stored.
        phase: the step's ``"prefill"`` or ``"decode"``; None when it is not known,
            and then only a rule without a phase selects the kernel.

    Either kernel serves weights of the shapes ``kda_projections_torch`` multiplies
    (``q/k/v [W, H]``, ``b [Bw, H]``, the gates' ``[R, H]`` then ``[W, R]``, with
    ``Bw`` and ``R`` at most 128), on an NKI device or the simulator, when
    ``VLLM_NEURON_GLUE_FUSED`` selects ``kda_projections`` for this row count and phase.
    The decode kernel takes ``1 <= T <= KDA_PROJECTIONS_MAX_TOKENS``, bf16 or fp32, with
    ``H`` in 128-blocks per program and ``W`` at most 128; the prefill kernel takes a
    larger ``T`` in 128-row tiles per program, bf16, with the prepared operands, and a
    ``W`` above 128 when it holds several heads' shards (:func:`_serves_prefill`).
    Anything else is counted as declined.
    """
    ok = False
    weights = _weights(attn)
    if hidden_states.dim() == 2 and all(w.dim() == 2 for w in weights):
        tokens, hidden = (int(v) for v in hidden_states.shape)
        q_w, k_w, v_w, b_w, f_a, f_b, g_a, g_b = weights
        width, rank, heads = int(q_w.shape[0]), int(f_a.shape[0]), int(b_w.shape[0])
        ok = (
            glue_selected("kda_projections", tokens, phase)
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
            and all(w.dtype in (torch.bfloat16, torch.float32) for w in weights)
            and ((1 <= tokens <= KDA_PROJECTIONS_MAX_TOKENS
                  and hidden % (_P * launch_programs()) == 0 and width <= _P)
                 or _serves_prefill(hidden_states, attn, width, heads, rank))
            and rank <= _P and heads <= _P
            and all(tuple(w.shape) == (width, hidden) for w in (q_w, k_w, v_w))
            and int(b_w.shape[1]) == hidden
            and all(tuple(w.shape) == (rank, hidden) for w in (f_a, g_a))
            and all(tuple(w.shape) == (width, rank) for w in (f_b, g_b))
            and can_run_kernel(hidden_states)
        )
    if not ok:
        _count_declined()
    return ok


def kda_projections_torch(hidden_states: Tensor, attn) -> tuple[Tensor, ...]:
    """The torch route's expressions, verbatim: the route when the kernel declines."""
    x = hidden_states.to(torch.float32)

    def project(weight: Tensor) -> Tensor:
        return x @ weight.to(torch.float32).t()

    q_in = project(attn.q_proj_weight)
    k_in = project(attn.k_proj_weight)
    v_in = project(attn.v_proj_weight)
    raw_beta = project(attn.b_proj_weight)
    raw_gate = (x @ attn.f_a_proj_weight.to(torch.float32).t()) @ (
        attn.f_b_proj_weight.to(torch.float32).t()
    )
    out_gate = (x @ attn.g_a_proj_weight.to(torch.float32).t()) @ (
        attn.g_b_proj_weight.to(torch.float32).t()
    )
    return q_in, k_in, v_in, raw_gate, raw_beta, out_gate


def input_rms_norm(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
    """The KDA layer's pre-attention RMSNorm, ``x / sqrt(mean(x**2) + eps) * gain``.

    The expression of ``Glm5NextKDALayer._input_norm``, step for step: fp32 inside,
    the result in the input's dtype.
    """
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * weight.to(torch.float32)
    return normed.to(hidden_states.dtype)


def _gate_operands(attn, heads: int, width: int) -> dict:
    """The prefill kernel's gate operands from the module's ``A_log``, ``dt_bias`` and
    ``gate_lower_bound``: ``[1, heads]`` and ``[1, W]`` fp32, and the bound."""
    a_log, bias = attn.A_log, attn.dt_bias
    if a_log.numel() != heads or bias.numel() != width:
        raise ValueError(
            f"the gate needs one A_log per head and one dt_bias per key channel: "
            f"{heads} and {width}; got {a_log.numel()} and {bias.numel()}"
        )
    return {"a_log": a_log.to(torch.float32).reshape(1, heads).contiguous(),
            "gate_bias": bias.to(torch.float32).reshape(1, width).contiguous(),
            "LOWER": float(attn.gate_lower_bound)}


def kda_projections(hidden_states: Tensor, attn, phase: str | None = None,
                    norm: tuple[Tensor, float] | None = None,
                    gate: bool = False) -> tuple[Tensor, ...]:
    """The six KDA input projections of ``hidden_states``, on a kernel or in torch.

    Args:
        hidden_states: ``[T, H]`` activations, bf16 or fp32; with ``norm``, the
            un-normed layer input.
        attn: the KDA attention module (its projection weights, and above
            ``KDA_PROJECTIONS_MAX_TOKENS`` rows the operands its load-time prep stored).
        phase: the step's ``"prefill"`` or ``"decode"``, for ``VLLM_NEURON_GLUE_FUSED``.
        norm: None, or the layer's input RMSNorm ``(gain [H], eps)``. Then the
            projections are of :func:`input_rms_norm` of ``hidden_states``: on the
            prefill kernel, which norms the rows on chip, when it serves the call and
            the gain is bf16; otherwise normed here first, then as without ``norm``.
        gate: whether the fourth output is the KDA gate of ``raw_gate``, with the
            module's ``A_log``, ``dt_bias`` and ``gate_lower_bound``
            (:func:`~vllm_neuron.functional.kda.gate_clamp.kda_gate`). On the prefill
            kernel it is applied on chip; on every other route, by ``kda_gate``.

    Returns:
        ``(q_in, k_in, v_in, raw_gate, raw_beta, out_gate)``, each ``[T, rows]`` fp32,
        ``rows`` the weight's output rows; with ``gate``, the gate in place of
        ``raw_gate``. When :func:`kda_projections_admits` says no, these are the torch
        expressions (:func:`kda_projections_torch`).
    """
    admitted = kda_projections_admits(hidden_states, attn, phase)
    prefill = admitted and int(hidden_states.shape[0]) > KDA_PROJECTIONS_MAX_TOKENS
    if norm is not None and not (prefill and norm[0].dtype == torch.bfloat16
                                 and tuple(norm[0].shape) == (int(hidden_states.shape[1]),)):
        hidden_states = input_rms_norm(hidden_states, *norm)
        norm = None
    if not admitted:
        projected = kda_projections_torch(hidden_states, attn)
    elif prefill:
        _count_nki_dispatch()
        wt, expand_t = getattr(attn, PREFILL_OPERANDS_ATTR)
        fused_norm = {} if norm is None else {"gain": norm[0].contiguous(),
                                              "EPS": float(norm[1])}
        fused_gate = {}
        if gate:
            width = int(expand_t.shape[1]) // 2
            fused_gate = _gate_operands(attn, int(attn.b_proj_weight.shape[0]), width)
        return _PREFILL_KERNELS[launch_programs()](
            x=hidden_states.contiguous(), wt=wt, expand_t=expand_t, **fused_norm,
            **fused_gate)
    else:
        _count_nki_dispatch()
        q_w, k_w, v_w, b_w, f_a, f_b, g_a, g_b = (w.contiguous() for w in _weights(attn))
        projected = _KERNELS[launch_programs()](
            x=hidden_states.contiguous(), q_w=q_w, k_w=k_w, v_w=v_w, b_w=b_w,
            f_a_w=f_a, f_b_w=f_b, g_a_w=g_a, g_b_w=g_b,
            DMA_TRANSPOSE=envs.VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE,
        )
    if not gate:
        return projected
    q_in, k_in, v_in, raw_gate, raw_beta, out_gate = projected
    return (q_in, k_in, v_in,
            kda_gate(raw_gate, attn.A_log, lower=attn.gate_lower_bound, bias=attn.dt_bias),
            raw_beta, out_gate)
