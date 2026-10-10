# SPDX-License-Identifier: Apache-2.0
"""K2: the MTP draft iteration's output tail as one NKI kernel.

The traced head's draft loop (``model/glm5_next/mtp.py``) ends each iteration with::

    mixed = attended + ffn                       # bf16 add
    hidden = rms(mixed, shared_head_norm)        # fp32 interior, bf16 out
    logits = linear(hidden, head_rows)           # bf16 x bf16 -> bf16, this rank's shard
    pair = (logits.max(-1), logits.argmax(-1))   # fp32 [B, 2]: the gather's operand

:func:`mtp_tail_out` is that expression: ``hidden`` is the next iteration's previous
hidden state (and the draft collector's), ``pair`` is what
:func:`vllm_neuron.functional.draft_token.draft_token_ids` all-gathers to resolve the
global token id. The time is the head shard's bytes (the served ``[2420, 4096]`` bf16
slab, 18.9 MiB per rank), read once.

Kernel (``mtp_tail_out_kernel``). Rows are tiled in 128 (one row per partition): the
bf16 add is one rounding of the exact sum (as torch's), the norm is
:func:`~vllm_neuron.functional.mtp.common.rms_rows` (the reference's rounding
points), the normed rows are transposed onto the contraction axis. The shard rows
are streamed in 512-column chunks through the DMA engines' transpose and each chunk
is one ``nc_matmul`` accumulation over the ``H / 128`` contraction blocks per row
tile, rounded to bf16 into a ``[rows, Vs]`` logits tile. ``max8`` finds each row's
largest value and ``nc_find_index8`` its first index (the lowest index among equal
maxima, ``torch.argmax``'s convention). Under two programs (LNC2) each program takes
half of the shard rows, offsets its index by its first row, and the two pairs are
exchanged (``sendrecv``) and combined with the same rule: the larger max, the lower
index on a tie. Program 0 writes both outputs.

The torch route (:func:`mtp_tail_out_torch`) is the traced head's expression verbatim: the
CPU-mode and kill-switch path, and the reference the kernel is tested against.
"""

from __future__ import annotations

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.mtp.common import (
    PARTITIONS,
    DispatchCounters,
    MtpTailError,
    count_kernel,
    count_torch_route,
    even,
    launch_programs,
    load_rows_transposed,
    rms_rows,
    tile,
    transpose_rows,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

__all__ = [
    "MAX_ROWS_PER_PROGRAM",
    "MIN_ROWS_PER_PROGRAM",
    "MtpTailError",
    "dispatch_counters",
    "launch_programs",
    "mtp_tail_out",
    "mtp_tail_out_kernel",
    "mtp_tail_out_torch",
    "reset_dispatch_counters",
    "shard_pair",
]

#: ``max8`` / ``nc_find_index8`` scan between 8 and 16384 elements per partition: the
#: shard rows one program takes (``ceil(Vs / programs)``) must lie in that range.
MIN_ROWS_PER_PROGRAM = 8
MAX_ROWS_PER_PROGRAM = 16384
#: Values ``max8`` returns per row; the first is the maximum.
_TOP = 8


def _program_rows(rows: int, programs: int) -> int:
    """Shard rows per program, ``ceil(rows / programs)``."""
    return -(-rows // programs)


@nki.jit
def mtp_tail_out_kernel(attended, ffn, gain, head_rows, EPS: float):
    """``(hidden [B, H] bf16, pair [B, 2] fp32)``: the norm and the shard's (max, argmax).

    Args:
        attended: ``[B, H]`` bf16, the attention half's output; ``H`` a multiple of 128.
        ffn: ``[B, H]`` bf16, the feed-forward half's output.
        gain: ``[H]`` bf16, ``shared_head.norm``'s gain.
        head_rows: ``[Vs, H]`` bf16, this rank's rows of the head;
            ``ceil(Vs / programs)`` in ``[8, 16384]``.
        EPS: the norm's epsilon.
    """
    batch, hidden = attended.shape
    rows = head_rows.shape[0]
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(hidden % PARTITIONS == 0, "H is a multiple of 128")
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(head_rows.shape[1] == hidden, "head rows are [Vs, H]")
    share = _program_rows(rows, programs)
    v_lo = program * share
    v_hi = min(rows, v_lo + share)
    span = v_hi - v_lo
    kernel_assert(span >= MIN_ROWS_PER_PROGRAM, "at least 8 shard rows per program")
    kernel_assert(span <= MAX_ROWS_PER_PROGRAM, "at most 16384 shard rows per program")
    blocks = hidden // PARTITIONS
    work = attended.dtype
    hidden_out = nl.ndarray((batch, hidden), dtype=work, buffer=nl.shared_hbm)
    pair_out = nl.ndarray((batch, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    ntiles = -(-batch // PARTITIONS)

    # ---- 1. hidden = rms(attended + ffn) per row tile; hidden^T for the GEMV. ----- #
    xt = nl.ndarray((PARTITIONS, blocks, batch), dtype=work, buffer=nl.sbuf)
    for i in range(ntiles):
        b0 = i * PARTITIONS
        n = min(PARTITIONS, batch - b0)
        a_sb = nl.ndarray((n, hidden), dtype=work, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=attended[b0:b0 + n, :])
        f_sb = nl.ndarray((n, hidden), dtype=work, buffer=nl.sbuf)
        nisa.dma_copy(dst=f_sb, src=ffn[b0:b0 + n, :])
        # One rounding of the exact sum, as torch's bf16 add.
        mixed = nl.ndarray((n, hidden), dtype=work, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mixed, data1=a_sb, data2=f_sb, op=nl.add)
        mixed_f = tile(n, hidden)
        nisa.tensor_copy(dst=mixed_f, src=mixed)
        gain_sb = nl.ndarray((n, hidden), dtype=gain.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gain_sb, src=gain.ap(pattern=[[0, n], [1, hidden]]))
        normed = rms_rows(mixed_f, n, hidden, gain_sb, EPS, work)
        if program == 0:
            nisa.dma_copy(dst=hidden_out[b0:b0 + n, :], src=normed)
        transpose_rows(xt[:, :, b0:b0 + n], normed, n, blocks, 0)

    # ---- 2. The shard rows this program takes, 128 at a time as contiguous row ---- #
    #         tiles turned on the PE; one GEMV per (row tile, batch tile); bf16 logits.
    logits = []
    for i in range(ntiles):
        n = min(PARTITIONS, batch - i * PARTITIONS)
        logits.append(nl.ndarray((n, span), dtype=work, buffer=nl.sbuf))
    for c0 in range(0, span, PARTITIONS):
        cn = min(PARTITIONS, span - c0)
        wt = nl.ndarray((PARTITIONS, blocks, even(cn)), dtype=head_rows.dtype,
                        buffer=nl.sbuf)
        load_rows_transposed(wt, head_rows, v_lo + c0, cn, blocks)
        for i in range(ntiles):
            b0 = i * PARTITIONS
            n = min(PARTITIONS, batch - b0)
            acc = nl.ndarray((n, cn), dtype=nl.float32, buffer=nl.psum)
            for kb in range(blocks):
                nisa.nc_matmul(dst=acc, stationary=xt[:, kb, b0:b0 + n],
                               moving=wt[:, kb, 0:cn], accumulate=(kb > 0))
            nisa.tensor_copy(dst=logits[i][:, c0:c0 + cn], src=acc)

    # ---- 3. (max, first argmax) per row; across two programs the larger max wins, --- #
    #         the lower index on a tie; program 0 stores.
    for i in range(ntiles):
        b0 = i * PARTITIONS
        n = min(PARTITIONS, batch - b0)
        top = nl.ndarray((n, _TOP), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top, src=logits[i])
        top_w = nl.ndarray((n, _TOP), dtype=work, buffer=nl.sbuf)
        nisa.tensor_copy(dst=top_w, src=top)
        where = nl.ndarray((n, _TOP), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=where, data=logits[i], vals=top_w)
        mine = tile(n, 2)
        nisa.tensor_copy(dst=mine[:, 0:1], src=top[:, 0:1])
        nisa.tensor_scalar(dst=mine[:, 1:2], data=where[:, 0:1], op0=nl.add,
                           operand0=float(v_lo))
        if programs == 1:
            nisa.dma_copy(dst=pair_out[b0:b0 + n, :], src=mine)
        else:
            _combine_and_store(mine, n, program, pair_out, b0)
    return hidden_out, pair_out


def _combine_and_store(mine, rows, program, pair_out, b0):
    """Both programs' pairs combined -- the larger max, the lower index on a tie -- and
    program 0's copy stored to ``pair_out[b0:b0 + rows]``."""
    n = rows
    theirs = tile(n, 2)
    nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - program,
                  recv_from_rank=1 - program, pipe_id=0)
    # take = (theirs.max > mine.max) + (theirs.max == mine.max) * (theirs.idx < mine.idx)
    greater = tile(n, 1)
    nisa.tensor_tensor(dst=greater, data1=theirs[:, 0:1], data2=mine[:, 0:1], op=nl.greater)
    equal = tile(n, 1)
    nisa.tensor_tensor(dst=equal, data1=theirs[:, 0:1], data2=mine[:, 0:1], op=nl.equal)
    lower = tile(n, 1)
    nisa.tensor_tensor(dst=lower, data1=theirs[:, 1:2], data2=mine[:, 1:2], op=nl.less)
    tie_take = tile(n, 1)
    nisa.tensor_tensor(dst=tie_take, data1=equal, data2=lower, op=nl.multiply)
    take = tile(n, 1)
    nisa.tensor_tensor(dst=take, data1=greater, data2=tie_take, op=nl.add)
    keep = tile(n, 1)
    nisa.tensor_scalar(dst=keep, data=take, op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=1.0)
    # Exact select: x * 1 = x, x * 0 = 0, 0 + x = x.
    kept = tile(n, 2)
    nisa.tensor_scalar(dst=kept, data=mine, op0=nl.multiply, operand0=keep)
    taken = tile(n, 2)
    nisa.tensor_scalar(dst=taken, data=theirs, op0=nl.multiply, operand0=take)
    combined = tile(n, 2)
    nisa.tensor_tensor(dst=combined, data1=kept, data2=taken, op=nl.add)
    if program == 0:
        nisa.dma_copy(dst=pair_out[b0:b0 + n, :], src=combined)


_KERNELS = {1: wrap_nki(mtp_tail_out_kernel), 2: wrap_nki(mtp_tail_out_kernel)[2]}
_COUNTERS = DispatchCounters()
_count_kernel = count_kernel(_COUNTERS)
_count_torch_route = count_torch_route(_COUNTERS)


def reset_dispatch_counters() -> None:
    _COUNTERS.reset()


def dispatch_counters() -> tuple[int, int]:
    """``(kernel launches that returned, torch-route calls)`` since the last reset."""
    return _COUNTERS.pair()


def shard_pair(logits: Tensor) -> Tensor:
    """``[B, 2]`` fp32 ``(max, first argmax)`` of ``[B, Vs]`` fp32 shard logits."""
    local_max, local_arg = logits.max(dim=-1)
    return torch.stack([local_max, local_arg.to(torch.float32)], dim=-1)


def _checked(attended: Tensor, ffn: Tensor, gain: Tensor, head_rows: Tensor,
             programs: int) -> None:
    """Raise :class:`MtpTailError` naming the operand unless the geometry is the tail's."""
    if attended.dim() != 2 or attended.dtype != torch.bfloat16:
        raise MtpTailError(f"mtp_tail_out: attended must be [B, H] bf16, got "
                           f"{tuple(attended.shape)} {attended.dtype}")
    batch, hidden = attended.shape
    if hidden % PARTITIONS or batch < 1:
        raise MtpTailError(f"mtp_tail_out: H={hidden} must be a multiple of {PARTITIONS} "
                           f"and B={batch} at least 1")
    if tuple(ffn.shape) != (batch, hidden) or ffn.dtype != attended.dtype:
        raise MtpTailError(f"mtp_tail_out: ffn must be [B={batch}, H={hidden}] {attended.dtype}, "
                           f"got {tuple(ffn.shape)} {ffn.dtype}")
    if tuple(gain.shape) != (hidden,) or gain.dtype != attended.dtype:
        raise MtpTailError(f"mtp_tail_out: gain must be [H={hidden}] {attended.dtype}, got "
                           f"{tuple(gain.shape)} {gain.dtype}")
    if head_rows.dim() != 2 or head_rows.shape[1] != hidden or head_rows.dtype != attended.dtype:
        raise MtpTailError(f"mtp_tail_out: head_rows must be [Vs, H={hidden}] {attended.dtype}, "
                           f"got {tuple(head_rows.shape)} {head_rows.dtype}")
    rows = head_rows.shape[0]
    share = _program_rows(rows, programs)
    least = rows - share * (programs - 1)
    if least < MIN_ROWS_PER_PROGRAM or share > MAX_ROWS_PER_PROGRAM:
        raise MtpTailError(
            f"mtp_tail_out: {rows} head rows over {programs} program(s) gives a program "
            f"{share if share > MAX_ROWS_PER_PROGRAM else least} rows; the max/argmax scan "
            f"takes between {MIN_ROWS_PER_PROGRAM} and {MAX_ROWS_PER_PROGRAM} per program")


def mtp_tail_out_torch(attended: Tensor, ffn: Tensor, gain: Tensor, head_rows: Tensor, *,
                       eps: float) -> tuple[Tensor, Tensor]:
    """The tail as torch ops, the traced head's draft loop arithmetic bit for bit.

    ``hidden = rms(attended + ffn, gain)`` (fp32 interior, cast to the inputs' dtype);
    ``pair = shard_pair(linear(hidden, head_rows).float())``.
    """
    mixed = attended + ffn
    x = mixed.to(torch.float32)
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    hidden = (normed * gain.to(torch.float32)).to(mixed.dtype)
    logits = torch.nn.functional.linear(hidden.to(head_rows.dtype), head_rows).to(torch.float32)
    return hidden, shard_pair(logits)


def mtp_tail_out(attended: Tensor, ffn: Tensor, gain: Tensor, head_rows: Tensor, *,
                 eps: float) -> tuple[Tensor, Tensor]:
    """``(hidden [B, H] bf16, pair [B, 2] fp32)`` of the draft iteration's output tail.

    The kernel wherever NKI kernels run (:func:`can_run_kernel`); the torch route
    otherwise (CPU mode without the simulator, or the kernels' kill switch).

    Args:
        attended, ffn: ``[B, H]`` bf16, the two halves' outputs; ``H`` a multiple of 128.
        gain: ``[H]`` bf16, ``shared_head.norm``'s gain.
        head_rows: ``[Vs, H]`` bf16, this rank's rows of the head; per program
            (``ceil(Vs / programs)``) between 8 and 16384 rows.
        eps: the norm's epsilon.

    Raises:
        MtpTailError: on any other geometry or dtype, naming the operand.
    """
    programs = launch_programs()
    _checked(attended, ffn, gain, head_rows, programs)
    if not can_run_kernel(attended):
        outputs = mtp_tail_out_torch(attended, ffn, gain, head_rows, eps=eps)
        _count_torch_route()
        return outputs
    hidden, pair = _KERNELS[programs](
        attended=attended.contiguous(), ffn=ffn.contiguous(), gain=gain.contiguous(),
        head_rows=head_rows.contiguous(), EPS=float(eps))
    # Counted once the launch has returned: a launch that raises served nothing.
    _count_kernel()
    return hidden, pair
