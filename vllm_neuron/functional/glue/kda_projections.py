# SPDX-License-Identifier: Apache-2.0
"""The KDA decode input projections in one NKI launch, on the bf16 weights as stored.

0a08ff4 (``Glm5NextKDAAttention.forward`` at one token, and
``_fused_decode_requests``) computes, with ``x = hidden_states.float()``::

    q_in, k_in, v_in = x @ q_w.float().t(), x @ k_w.float().t(), x @ v_w.float().t()
    raw_beta = x @ b_w.float().t()
    raw_gate = (x @ f_a_w.float().t()) @ f_b_w.float().t()
    out_gate = (x @ g_a_w.float().t()) @ g_b_w.float().t()

so every step casts and transposes 641 x 4096 weight values to fp32 and runs fp32
matmuls on them. This kernel reads the bf16 activations and weights as they are. fp32
operands (the CPU fixtures' dtype) run the same steps on fp32 tiles, with the weights'
transposes on the tensor engine.

Method. Both programs (``P`` under LNC2) take ``H / P`` of the contraction:

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

Both LNC2 cores: yes; steps 1-2 split the contraction, step 3 splits the two gates.
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

from vllm_neuron.functional.glue import glue_fused_enabled
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Largest token count served (the activations' transpose is the matmul stationary).
KDA_PROJECTIONS_MAX_TOKENS = 128

#: ``1`` loads the weights' transposes by DMA, ``0`` by tensor-engine transposes.
DMA_TRANSPOSE_ENV = "VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE"

_P = 128
_BANK = 512  # fp32 columns per PSUM bank
_DGE_ROWS = 16  # weight rows per hardware-DGE transpose
_BANK_BYTES = 2048  # bytes per partition of one PSUM bank
_XBAR_STEP = 16  # a DMA transpose's output row stride: a multiple of 32 bytes of bf16

__all__ = [
    "KDA_PROJECTIONS_MAX_TOKENS",
    "dispatch_counters",
    "kda_projections",
    "kda_projections_admits",
    "kda_projections_kernel",
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


_KERNELS = {1: wrap_nki(kda_projections_kernel), 2: wrap_nki(kda_projections_kernel)[2]}


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


def _weights(attn) -> tuple[Tensor, ...]:
    return (attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight,
            attn.b_proj_weight, attn.f_a_proj_weight, attn.f_b_proj_weight,
            attn.g_a_proj_weight, attn.g_b_proj_weight)


def kda_projections_admits(hidden_states: Tensor, attn) -> bool:
    """True when :func:`kda_projections` serves this call: bf16 or fp32 operands."""
    ok = False
    weights = _weights(attn)
    if hidden_states.dim() == 2 and all(w.dim() == 2 for w in weights):
        tokens, hidden = (int(v) for v in hidden_states.shape)
        q_w, k_w, v_w, b_w, f_a, f_b, g_a, g_b = weights
        width, rank = int(q_w.shape[0]), int(f_a.shape[0])
        ok = (
            glue_fused_enabled()
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
            and all(w.dtype in (torch.bfloat16, torch.float32) for w in weights)
            and 1 <= tokens <= KDA_PROJECTIONS_MAX_TOKENS
            and hidden % (_P * launch_programs()) == 0
            and width <= _P and rank <= _P and int(b_w.shape[0]) <= _P
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
    """0a08ff4's torch expressions, verbatim: the route when the kernel declines."""
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


def kda_projections(hidden_states: Tensor, attn) -> tuple[Tensor, ...]:
    """``(q_in, k_in, v_in, raw_gate, raw_beta, out_gate)`` fp32, kernel or 0a08ff4 torch."""
    if not kda_projections_admits(hidden_states, attn):
        return kda_projections_torch(hidden_states, attn)
    _count_nki_dispatch()
    q_w, k_w, v_w, b_w, f_a, f_b, g_a, g_b = (w.contiguous() for w in _weights(attn))
    q_in, k_in, v_in, raw_gate, raw_beta, out_gate = _KERNELS[launch_programs()](
        x=hidden_states.contiguous(), q_w=q_w, k_w=k_w, v_w=v_w, b_w=b_w,
        f_a_w=f_a, f_b_w=f_b, g_a_w=g_a, g_b_w=g_b,
        DMA_TRANSPOSE=os.environ.get(DMA_TRANSPOSE_ENV, "1") != "0",
    )
    return q_in, k_in, v_in, raw_gate, raw_beta, out_gate
