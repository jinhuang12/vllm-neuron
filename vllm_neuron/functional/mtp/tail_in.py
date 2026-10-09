# SPDX-License-Identifier: Apache-2.0
"""K1': the MTP draft iteration's input tail as one NKI kernel, ``eh_proj`` row-sharded.

The head's ``_layer_input`` (``model/glm5_next/mtp.py``) computes, per draft row::

    embeds = table[token_id]; embeds = 0 where position == 0
    joined = cat([rms(embeds, enorm), rms(previous, hnorm)])        # [2H], bf16
    layer_input = eh_proj @ joined                                   # [H], bf16

:func:`mtp_tail_in` is that expression with ``eh_proj`` ROW-sharded over the tensor
parallel group: rank ``r`` holds rows ``[r * H / world, (r + 1) * H / world)`` and
returns its ``[B, H / world]`` slice of ``layer_input``; the caller all-gathers the
slices along the last dimension (rank-major, which is row order). Every rank reads
the whole embedding row and both norm gains (a few KiB) and ``1 / world`` of
``eh_proj`` (the served ``[64, 8192]`` bf16 slab, 1 MiB), the bytes that bound the
kernel's time.

Kernel (``mtp_tail_in_kernel``). Rows are tiled in 128 (one row per partition): the
embedding rows are gathered straight from HBM by an indirect DMA on the token ids,
the position-0 mask is a per-row fp32 factor (``position != 0``), both halves are
normed by :func:`~vllm_neuron.functional.mtp.common.rms_rows` (fp32 interior,
bf16 result -- the reference's rounding points) and transposed onto the contraction
axis as ``xt[c, kb, row] = joined[row, kb * 128 + c]``. The shard's rows are streamed
in column chunks of 512 through the DMA engines' transpose
(``wt[c, kb, col] = eh_proj_rows[col, kb * 128 + c]``), and each chunk is one
``nc_matmul`` accumulation over the ``2H / 128`` contraction blocks per row tile,
rounded to bf16 on the copy out of PSUM. Under two programs (LNC2) each program
takes half of the shard's rows (its output columns) and writes its own slice.

The torch route (:func:`mtp_tail_in_torch`) is the Stage A expression verbatim: the
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
    "MtpTailError",
    "dispatch_counters",
    "eh_proj_shard_rows",
    "mtp_tail_in",
    "mtp_tail_in_kernel",
    "mtp_tail_in_torch",
    "reset_dispatch_counters",
]

#: The absolute position whose embedding half the head zeroes (no previous token).
MASKED_POSITION = 0


def eh_proj_shard_rows(hidden: int, world_size: int) -> int:
    """Rows of ``eh_proj`` each rank holds: ``hidden / world_size``.

    Raises:
        MtpTailError: unless ``world_size`` is positive and divides ``hidden``.
    """
    if world_size < 1:
        raise MtpTailError(f"mtp_tail_in: the TP world size must be positive, got {world_size}")
    if hidden % world_size:
        raise MtpTailError(
            f"mtp_tail_in: the TP world size {world_size} does not divide the hidden size {hidden}")
    return hidden // world_size


@nki.jit
def mtp_tail_in_kernel(token_ids, table, positions, previous, enorm, hnorm, eh_proj_rows,
                       EPS: float):
    """``[B, R]`` bf16: this rank's rows of ``eh_proj(cat(enorm(mask(embed)), hnorm(prev)))``.

    Args:
        token_ids: ``[B]`` int32 ids into ``table``.
        table: ``[V, H]`` bf16 embedding table; ``H`` a multiple of 128.
        positions: ``[B]`` int32 absolute positions; rows at 0 draft from the hidden
            half alone.
        previous: ``[B, H]`` bf16 previous hidden state.
        enorm, hnorm: ``[H]`` bf16 norm gains of the two halves.
        eh_proj_rows: ``[R, 2H]`` bf16, this rank's rows of ``eh_proj``, ``R >= 1``.
        EPS: the norms' epsilon.
    """
    batch = token_ids.shape[0]
    hidden = table.shape[1]
    rows, width = eh_proj_rows.shape
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(hidden % PARTITIONS == 0, "H is a multiple of 128")
    kernel_assert(width == 2 * hidden, "eh_proj rows are [R, 2H]")
    kernel_assert(rows >= 1 and batch >= 1, "R >= 1 and B >= 1")
    kernel_assert(programs in (1, 2), "one or two programs")
    half = hidden // PARTITIONS
    blocks = 2 * half
    work = table.dtype
    out = nl.ndarray((batch, rows), dtype=eh_proj_rows.dtype, buffer=nl.shared_hbm)

    # ---- 1. joined^T for every row tile: xt[c, kb, b] = joined[b, kb * 128 + c]. -- #
    xt = nl.ndarray((PARTITIONS, blocks, batch), dtype=work, buffer=nl.sbuf)
    for b0 in range(0, batch, PARTITIONS):
        n = min(PARTITIONS, batch - b0)
        idx = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=token_ids.ap(pattern=[[1, n], [1, 1]], offset=b0))
        embeds = nl.ndarray((n, hidden), dtype=work, buffer=nl.sbuf)
        nisa.dma_copy(dst=embeds, src=table.ap(pattern=[[hidden, n], [1, hidden]],
                                               vector_offset=idx, indirect_dim=0))
        pos = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pos, src=positions.ap(pattern=[[1, n], [1, 1]], offset=b0))
        pos_f = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pos_f, src=pos)
        # keep[b] = 1.0 unless the position is MASKED_POSITION: the embedding's factor.
        keep = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=pos_f, op0=nl.not_equal,
                           operand0=float(MASKED_POSITION))
        embeds_f = tile(n, hidden)
        nisa.tensor_scalar(dst=embeds_f, data=embeds, op0=nl.multiply, operand0=keep)
        prev = nl.ndarray((n, hidden), dtype=previous.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=prev, src=previous[b0:b0 + n, :])
        prev_f = tile(n, hidden)
        nisa.tensor_copy(dst=prev_f, src=prev)
        # The gains, one copy per row (partition stride 0 in the source).
        enorm_sb = nl.ndarray((n, hidden), dtype=enorm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=enorm_sb, src=enorm.ap(pattern=[[0, n], [1, hidden]]))
        hnorm_sb = nl.ndarray((n, hidden), dtype=hnorm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=hnorm_sb, src=hnorm.ap(pattern=[[0, n], [1, hidden]]))
        e_normed = rms_rows(embeds_f, n, hidden, enorm_sb, EPS, work)
        h_normed = rms_rows(prev_f, n, hidden, hnorm_sb, EPS, work)
        transpose_rows(xt[:, :, b0:b0 + n], e_normed, n, half, 0)
        transpose_rows(xt[:, :, b0:b0 + n], h_normed, n, half, half)

    # ---- 2. This program's rows of eh_proj, 128 at a time as contiguous row tiles -- #
    #         turned on the PE; one GEMV per (row tile, batch tile).
    share = -(-rows // programs)
    c_lo = program * share
    c_hi = min(rows, c_lo + share)
    for c0 in range(c_lo, c_hi, PARTITIONS):
        cn = min(PARTITIONS, c_hi - c0)
        wt = nl.ndarray((PARTITIONS, blocks, even(cn)), dtype=eh_proj_rows.dtype,
                        buffer=nl.sbuf)
        load_rows_transposed(wt, eh_proj_rows, c0, cn, blocks)
        for b0 in range(0, batch, PARTITIONS):
            n = min(PARTITIONS, batch - b0)
            acc = nl.ndarray((n, cn), dtype=nl.float32, buffer=nl.psum)
            for kb in range(blocks):
                nisa.nc_matmul(dst=acc, stationary=xt[:, kb, b0:b0 + n],
                               moving=wt[:, kb, 0:cn], accumulate=(kb > 0))
            out_sb = nl.ndarray((n, cn), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb, src=acc)
            nisa.dma_copy(dst=out[b0:b0 + n, c0:c0 + cn], src=out_sb)
    return out


_KERNELS = {1: wrap_nki(mtp_tail_in_kernel), 2: wrap_nki(mtp_tail_in_kernel)[2]}
_COUNTERS = DispatchCounters()
_count_kernel = count_kernel(_COUNTERS)
_count_torch_route = count_torch_route(_COUNTERS)


def reset_dispatch_counters() -> None:
    _COUNTERS.reset()


def dispatch_counters() -> tuple[int, int]:
    """``(kernel launches that returned, torch-route calls)`` since the last reset."""
    return _COUNTERS.pair()


def _checked(token_ids: Tensor, table: Tensor, positions: Tensor, previous: Tensor,
             enorm: Tensor, hnorm: Tensor, eh_proj_rows: Tensor) -> None:
    """Raise :class:`MtpTailError` naming the operand unless the geometry is the tail's."""
    if token_ids.dim() != 1 or token_ids.dtype != torch.int32:
        raise MtpTailError(f"mtp_tail_in: token_ids must be a 1-D int32 tensor, got "
                           f"{tuple(token_ids.shape)} {token_ids.dtype}")
    batch = token_ids.shape[0]
    if table.dim() != 2 or table.dtype != torch.bfloat16:
        raise MtpTailError(f"mtp_tail_in: table must be [V, H] bf16, got "
                           f"{tuple(table.shape)} {table.dtype}")
    hidden = table.shape[1]
    if hidden % PARTITIONS:
        raise MtpTailError(f"mtp_tail_in: the table's H={hidden} is not a multiple of {PARTITIONS}")
    if tuple(positions.shape) != (batch,) or positions.dtype != torch.int32:
        raise MtpTailError(f"mtp_tail_in: positions must be [B={batch}] int32, got "
                           f"{tuple(positions.shape)} {positions.dtype}")
    if tuple(previous.shape) != (batch, hidden) or previous.dtype != table.dtype:
        raise MtpTailError(f"mtp_tail_in: previous must be [B={batch}, H={hidden}] {table.dtype}, "
                           f"got {tuple(previous.shape)} {previous.dtype}")
    for name, gain in (("enorm", enorm), ("hnorm", hnorm)):
        if tuple(gain.shape) != (hidden,) or gain.dtype != table.dtype:
            raise MtpTailError(f"mtp_tail_in: {name} must be [H={hidden}] {table.dtype}, got "
                               f"{tuple(gain.shape)} {gain.dtype}")
    if (eh_proj_rows.dim() != 2 or eh_proj_rows.shape[1] != 2 * hidden
            or eh_proj_rows.dtype != table.dtype):
        raise MtpTailError(f"mtp_tail_in: eh_proj_rows must be [R, 2H={2 * hidden}] {table.dtype}, "
                           f"got {tuple(eh_proj_rows.shape)} {eh_proj_rows.dtype}")
    if eh_proj_rows.shape[0] < 1 or batch < 1:
        raise MtpTailError(f"mtp_tail_in: needs at least one eh_proj row and one token, got "
                           f"{eh_proj_rows.shape[0]} rows and {batch} tokens")


def mtp_tail_in_torch(token_ids: Tensor, table: Tensor, positions: Tensor, previous: Tensor,
                      enorm: Tensor, hnorm: Tensor, eh_proj_rows: Tensor, *, eps: float
                      ) -> Tensor:
    """The tail as torch ops, the Stage A ``_layer_input`` arithmetic bit for bit.

    Each norm is ``(x32 * rsqrt(mean(x32**2) + eps)) * gain32`` cast to the input
    dtype; the GEMV is ``linear`` over the bf16 concatenation.
    """

    def rms(x: Tensor, gain: Tensor) -> Tensor:
        x32 = x.to(torch.float32)
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * gain.to(torch.float32)).to(x.dtype)

    embeds = table[token_ids.to(torch.int64)]
    embeds = torch.where(positions.reshape(-1, 1) == MASKED_POSITION,
                         torch.zeros((), dtype=embeds.dtype, device=embeds.device), embeds)
    joined = torch.cat([rms(embeds, enorm), rms(previous.to(embeds.dtype), hnorm)], dim=-1)
    return torch.nn.functional.linear(joined, eh_proj_rows)


def mtp_tail_in(token_ids: Tensor, table: Tensor, positions: Tensor, previous: Tensor,
                enorm: Tensor, hnorm: Tensor, eh_proj_rows: Tensor, *, eps: float) -> Tensor:
    """``[B, R]`` bf16: this rank's ``R`` rows of the draft iteration's ``layer_input``.

    The kernel wherever NKI kernels run (:func:`can_run_kernel`); the torch route
    otherwise (CPU mode without the simulator, or the kernels' kill switch).

    Args:
        token_ids: ``[B]`` int32.
        table: ``[V, H]`` bf16, ``H`` a multiple of 128.
        positions: ``[B]`` int32.
        previous: ``[B, H]`` bf16.
        enorm, hnorm: ``[H]`` bf16.
        eh_proj_rows: ``[R, 2H]`` bf16, this rank's rows (``R = H / world``).
        eps: the norms' epsilon.

    Raises:
        MtpTailError: on any other geometry or dtype, naming the operand.
    """
    _checked(token_ids, table, positions, previous, enorm, hnorm, eh_proj_rows)
    if not can_run_kernel(previous):
        out = mtp_tail_in_torch(token_ids, table, positions, previous, enorm, hnorm,
                                eh_proj_rows, eps=eps)
        _count_torch_route()
        return out
    out = _KERNELS[launch_programs()](
        token_ids=token_ids.contiguous(), table=table.contiguous(),
        positions=positions.contiguous(), previous=previous.contiguous(),
        enorm=enorm.contiguous(), hnorm=hnorm.contiguous(),
        eh_proj_rows=eh_proj_rows.contiguous(), EPS=float(eps))
    # Counted once the launch has returned: a launch that raises served nothing.
    _count_kernel()
    return out
