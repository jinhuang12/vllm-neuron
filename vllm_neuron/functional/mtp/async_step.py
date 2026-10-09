# SPDX-License-Identifier: Apache-2.0
"""The async drafter's per-step device corrections (speculative method "mtp", GLM-5.3-Flash).

Under ``--async-scheduling`` the scheduler advances a request by the whole verify step
(``1 + k`` tokens) before the rejection sampler's rows reach the host, and the rows of
one step are still a device future when the next step is built. The synchronous runner
corrected three things on the host once it had read the rows -- the indexer-ring cursor,
the KDA checkpoint commit and the resume row (``_update_states_after_model_execute``) --
and built every position-derived operand from host ints. Here both halves run on
device, one authored kernel each, so no host read sits between two steps:

* :func:`mtp_async_take` right after a step, on its output: the kept count per request
  (the non-placeholder prefix of the rejection sampler's row), the resume row
  ``kept - 1`` (what ``commit_kda_checkpoints`` returns, carried as a tensor), the last
  kept id and the next step's ``[B, 1 + k]`` input ids (that id, then the drafts);
* :func:`mtp_async_correct` when the next step is built: the scheduler's optimistic
  position pulled back by the rows the previous step rejected
  (``prev_width - 1 - prev_checkpoint_rows``), and from the true position the operands
  the translator otherwise derives from host ints -- the sparse carrier's per-row causal
  lengths and physical latent slots (the page gather through the request's block-table
  column), the recurrent carrier's start positions and checkpoint rows -- with the
  bucket's padding rows laid out as the host builder lays them (position 0 in the null
  block for the sparse family, position 1 and row 0 for the recurrent one).

The torch route (``*_torch``) is the contract and the CPU route; the kernels are held to
it bit for bit. Every value is an int32 count, id or position below ``2 ** 24``, so the
engines' fp32 arithmetic is exact; the latent slots leave the kernel as int32 pairs
(``[rows, 2]``: low word, zero high word) and are bit-viewed as the int64 the layers'
``index_copy_`` takes, since NKI has no 64-bit integer and a dtype cast on a device
tensor is a device op of its own. Dispatch counters (``dispatch_counters``) tell the
tests which route served.
"""

from __future__ import annotations

from typing import NamedTuple

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
    count_kernel,
    count_torch_route,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

__all__ = [
    "MtpAsyncStepError",
    "StepCarry",
    "StepCorrection",
    "StepTake",
    "dispatch_counters",
    "mtp_async_correct",
    "mtp_async_correct_kernel",
    "mtp_async_correct_torch",
    "mtp_async_take",
    "mtp_async_take_kernel",
    "mtp_async_take_torch",
    "reset_dispatch_counters",
]

#: The rejection sampler's marker for a rejected row (``nn/rejection_sampler.py``).
PLACEHOLDER_TOKEN_ID = -1
#: The largest value any operand or result may hold: fp32 is exact below it.
EXACT_BOUND = 1 << 24


def _on_the_host(tensor: Tensor) -> bool:
    """True when a value check costs no device read: a real CPU tensor.

    ``neuron_utils.values_are_readable`` answers True for a device tensor outside
    tracing (a read would sync the step this module exists to keep off the host), so
    the checks here run on the CPU route and in the tests, never on a device future.
    """
    return tensor.device.type == "cpu" and not isinstance(tensor, torch._subclasses.FakeTensor)


class MtpAsyncStepError(ValueError):
    """An operand that is not a verify step's: wrong geometry, dtype, or a value outside its range."""


class StepTake(NamedTuple):
    """What :func:`mtp_async_take` reads off one step's output, all int32 on its device."""

    #: ``[B]``: rows kept per request, ``1 .. W`` (the kept prefix of the sampler's row).
    valid_count: Tensor
    #: ``[B]``: ``valid_count - 1``, the checkpoint row the request resumes from.
    checkpoint_rows: Tensor
    #: ``[B]``: the last kept id, the next step's first input token.
    last_accepted: Tensor
    #: ``[B, 1 + k]``: ``last_accepted`` then the drafts, the next verify step's input ids.
    next_input_ids: Tensor


class StepCorrection(NamedTuple):
    """What :func:`mtp_async_correct` builds for a step of ``R = B + padding`` rows of ``width`` tokens."""

    #: ``[R]`` int32: each request's true position of row 0; a padding row's is 0.
    start_position: Tensor
    #: ``[R]`` int32: the same for the recurrent layers; a padding row's is 1 (it keeps its state).
    linear_start: Tensor
    #: ``[R * width]`` int32: row ``r * width + t``'s causal length, ``start_position[r] + t + 1``.
    seq_lens: Tensor
    #: ``[R * width]`` int64: row ``r * width + t``'s physical latent-bank row; ``None``
    #: when the step named no page table (a stack with no latent bank).
    latent_slots: Tensor | None
    #: ``[R]`` int32: the row each request's recurrent slot resumes from; a padding row's is 0.
    checkpoint_rows: Tensor


class StepCarry(NamedTuple):
    """What one step hands the next under the async drafter: its take, keyed by its requests.

    Written by the runner's state hook from :func:`mtp_async_take` of the step's output,
    read once by the translator of the step after it (:func:`mtp_async_correct`) and by
    the draft proposal (``next_input_ids`` is the next verify step's input, ``drafts`` the
    rejection sampler's draft ids). The tensors are the step's device futures or tensors;
    nothing here was read on the host.
    """

    #: The step's requests, in batch order; the next step must name the same ones.
    req_ids: tuple
    #: Rows per request the step carried: ``1 + k`` on a verify step, 1 otherwise.
    prev_width: int
    #: ``[rows]`` int32: ``kept - 1`` per row of the step (real requests first).
    checkpoint_rows: Tensor
    #: ``[rows]`` int32: the last kept id per row.
    last_accepted: Tensor
    #: ``[rows, 1 + k]`` int32: the next verify step's input ids.
    next_input_ids: Tensor
    #: ``[rows, k]`` int32: the drafts those ids carry (the root's, or a prefill's placeholders).
    drafts: Tensor


# ── take ───────────────────────────────────────────────────────────────────────────────


def _check_take(accepted: Tensor, drafts: Tensor) -> Tensor:
    """Return ``accepted`` as ``[B, W]`` int32 or raise naming the operand."""
    if accepted.dim() == 1:
        accepted = accepted.reshape(-1, 1)
    if accepted.dim() != 2 or accepted.dtype != torch.int32 or accepted.shape[0] < 1 or accepted.shape[1] < 1:
        raise MtpAsyncStepError(
            f"mtp_async_take: accepted must be the sampler's [B, W] int32 rows (or [B] for a "
            f"one-row step); got {tuple(accepted.shape)} {accepted.dtype}"
        )
    if drafts.dim() != 2 or drafts.dtype != torch.int32 or drafts.shape[1] < 1:
        raise MtpAsyncStepError(
            f"mtp_async_take: drafts must be the root's [B, k] int32 draft rows, k >= 1; got "
            f"{tuple(drafts.shape)} {drafts.dtype}"
        )
    if drafts.shape[0] != accepted.shape[0]:
        raise MtpAsyncStepError(
            f"mtp_async_take: drafts carry {int(drafts.shape[0])} row(s) for "
            f"{int(accepted.shape[0])} accepted row(s); one row per request"
        )
    if _on_the_host(accepted) and bool((accepted[:, 0] == PLACEHOLDER_TOKEN_ID).any()):
        raise MtpAsyncStepError(
            "mtp_async_take: a verify step keeps at least its first row (the correction or "
            "the first draft), and a row here starts with the placeholder"
        )
    return accepted


def mtp_async_take_torch(accepted: Tensor, drafts: Tensor) -> StepTake:
    """The take as torch ops: the contract, and the CPU route."""
    accepted = _check_take(accepted, drafts)
    batch = int(accepted.shape[0])
    valid_count = (accepted != PLACEHOLDER_TOKEN_ID).sum(dim=1).to(torch.int32)
    checkpoint_rows = valid_count - 1
    last_accepted = accepted[torch.arange(batch, device=accepted.device), checkpoint_rows.to(torch.int64)]
    next_input_ids = torch.cat([last_accepted.reshape(batch, 1), drafts], dim=1).contiguous()
    return StepTake(valid_count, checkpoint_rows, last_accepted.to(torch.int32), next_input_ids)


@nki.jit
def mtp_async_take_kernel(accepted, drafts):
    """``(valid [B, 1], rows [B, 1], last [B, 1], next [B, 1 + k])`` int32 from ``accepted [B, W]``, ``drafts [B, k]``.

    Per request (one partition each, tiles of 128): the kept count is the number of
    non-placeholder entries (the row is a kept prefix, so count = prefix length), the
    resume row is one less, the last kept id is the entry at that index (picked by an
    equality mask and summed: the other terms are zero), and the next ids are that id
    followed by the drafts. fp32 arithmetic throughout, exact below ``2 ** 24``.
    """
    batch, width = accepted.shape
    k = drafts.shape[1]
    kernel_assert(width >= 1 and k >= 1 and batch >= 1, "B >= 1, W >= 1, k >= 1")
    valid_out = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    rows_out = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    last_out = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    next_out = nl.ndarray((batch, 1 + k), dtype=nl.int32, buffer=nl.shared_hbm)
    for b0 in range(0, batch, PARTITIONS):
        n = min(PARTITIONS, batch - b0)
        acc_i = nl.ndarray((n, width), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=acc_i, src=accepted[b0:b0 + n, :])
        acc_f = nl.ndarray((n, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=acc_f, src=acc_i)
        # keep[b, w] = 1.0 where the entry is not the placeholder.
        keep = nl.ndarray((n, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=acc_f, op0=nl.not_equal,
                           operand0=float(PLACEHOLDER_TOKEN_ID))
        valid_f = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=valid_f, op=nl.add, data=keep, axis=(1,))
        rows_f = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rows_f, data=valid_f, op0=nl.add, operand0=-1.0)
        # The last kept entry: index == rows, one hit per request.
        index_f = nl.ndarray((n, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=index_f, pattern=[[1, width]], offset=0, channel_multiplier=0)
        hit = nl.ndarray((n, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hit, data=index_f, op0=nl.equal, operand0=rows_f)
        picked = nl.ndarray((n, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=picked, data1=acc_f, data2=hit, op=nl.multiply)
        last_f = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=last_f, op=nl.add, data=picked, axis=(1,))
        valid_i = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=valid_i, src=valid_f)
        rows_i = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rows_i, src=rows_f)
        last_i = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=last_i, src=last_f)
        nisa.dma_copy(dst=valid_out[b0:b0 + n, :], src=valid_i)
        nisa.dma_copy(dst=rows_out[b0:b0 + n, :], src=rows_i)
        nisa.dma_copy(dst=last_out[b0:b0 + n, :], src=last_i)
        nisa.dma_copy(dst=next_out[b0:b0 + n, 0:1], src=last_i)
        drafts_i = nl.ndarray((n, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=drafts_i, src=drafts[b0:b0 + n, :])
        nisa.dma_copy(dst=next_out[b0:b0 + n, 1:1 + k], src=drafts_i)
    return valid_out, rows_out, last_out, next_out


def mtp_async_take(accepted: Tensor, drafts: Tensor) -> StepTake:
    """:class:`StepTake` of one step's output; the kernel where kernels run, else the torch route.

    Args:
        accepted: ``[B, W]`` int32, the rejection sampler's rows (``W = 1 + k`` on a verify
            step; ``[B]`` or ``[B, 1]`` after a prefill or a one-row decode), a kept prefix
            of ids and ``-1`` after it. A device future is read by the kernel only.
        drafts: ``[B, k]`` int32, the root's drafts for the same requests, ``k >= 1``.

    Raises:
        MtpAsyncStepError: another geometry or dtype, or (when the values are on the host)
            a row that keeps nothing.
    """
    accepted = _check_take(accepted, drafts)
    if not can_run_kernel(accepted):
        take = mtp_async_take_torch(accepted, drafts)
        _count_torch_route()
        return take
    batch = int(accepted.shape[0])
    valid, rows, last, nxt = _TAKE_KERNEL(accepted=accepted.contiguous(), drafts=drafts.contiguous())
    _count_kernel()
    return StepTake(valid.reshape(batch), rows.reshape(batch), last.reshape(batch), nxt)


# ── correct ────────────────────────────────────────────────────────────────────────────


def _page_shift(page_size: int) -> int:
    page = int(page_size)
    if page < 1 or page & (page - 1):
        raise MtpAsyncStepError(
            f"mtp_async_correct: page_size must be a power of two (the slot is page << shift "
            f"| offset); got {page_size!r}"
        )
    return page.bit_length() - 1


def _check_correct(
    prev_checkpoint_rows: Tensor, optimistic_starts: Tensor, block_table_row: Tensor | None, *,
    prev_width: int, width: int, page_size: int, padding: int,
) -> tuple[int, int, int]:
    """Return ``(B, R, pages)`` or raise naming the operand."""
    if int(prev_width) < 1 or int(width) < 1:
        raise MtpAsyncStepError(
            f"mtp_async_correct: prev_width and width are rows per request, at least 1; got "
            f"prev_width={prev_width!r} width={width!r}"
        )
    if int(padding) < 0:
        raise MtpAsyncStepError(f"mtp_async_correct: padding must be >= 0; got {padding!r}")
    if prev_checkpoint_rows.dim() != 1 or prev_checkpoint_rows.dtype != torch.int32 or prev_checkpoint_rows.shape[0] < 1:
        raise MtpAsyncStepError(
            f"mtp_async_correct: prev_checkpoint_rows must be [B] int32, B >= 1; got "
            f"{tuple(prev_checkpoint_rows.shape)} {prev_checkpoint_rows.dtype}"
        )
    batch = int(prev_checkpoint_rows.shape[0])
    if tuple(optimistic_starts.shape) != (batch,) or optimistic_starts.dtype != torch.int32:
        raise MtpAsyncStepError(
            f"mtp_async_correct: optimistic_starts must be [B={batch}] int32; got "
            f"{tuple(optimistic_starts.shape)} {optimistic_starts.dtype}"
        )
    rows = batch + int(padding)
    if block_table_row is not None and (
        block_table_row.dim() != 2 or block_table_row.dtype != torch.int32
        or int(block_table_row.shape[1]) != rows
    ):
        raise MtpAsyncStepError(
            f"mtp_async_correct: block_table_row must be [pages, R] int32 with one column per "
            f"request and padding row (R = {batch} + {int(padding)} = {rows}); got "
            f"{tuple(block_table_row.shape)} {block_table_row.dtype}"
        )
    if rows > PARTITIONS:
        raise MtpAsyncStepError(
            f"mtp_async_correct: a step of {rows} rows exceeds the {PARTITIONS} partitions one "
            f"launch lays requests over"
        )
    pages = 0 if block_table_row is None else int(block_table_row.shape[0])
    if block_table_row is not None and pages < 1:
        raise MtpAsyncStepError("mtp_async_correct: block_table_row names no page")
    if _on_the_host(prev_checkpoint_rows) and _on_the_host(optimistic_starts):
        _check_values(prev_checkpoint_rows, optimistic_starts, block_table_row,
                      prev_width=int(prev_width), width=int(width), page_size=int(page_size))
    return batch, rows, pages


def _check_values(
    prev_checkpoint_rows: Tensor, optimistic_starts: Tensor, block_table_row: Tensor | None, *,
    prev_width: int, width: int, page_size: int,
) -> None:
    """The host's checks on readable values: the resume row, the true start, the pages.

    Run on both routes when the operands are on the host (the CPU route, the tests); a
    device future is not read, and the device computes without refusing.
    """
    if bool(((prev_checkpoint_rows < 0) | (prev_checkpoint_rows > prev_width - 1)).any()):
        raise MtpAsyncStepError(
            f"mtp_async_correct: a resume row lies outside 0 .. {prev_width - 1}, the rows "
            f"a step {prev_width} wide can keep: {prev_checkpoint_rows.tolist()}"
        )
    rejected = (prev_width - 1) - prev_checkpoint_rows.to(torch.int64)
    true_start = optimistic_starts.to(torch.int64) - rejected
    if bool((true_start < 0).any()):
        raise MtpAsyncStepError(
            f"mtp_async_correct: a corrected position falls below zero: optimistic "
            f"{optimistic_starts.tolist()} minus rejected {rejected.tolist()}"
        )
    if block_table_row is None:
        return
    shift = _page_shift(page_size)
    pages = int(block_table_row.shape[0])
    positions = true_start.reshape(-1, 1) + torch.arange(width, dtype=torch.int64)
    page_index = positions >> shift
    if bool((page_index >= pages).any()):
        raise MtpAsyncStepError(
            f"mtp_async_correct: a position of {positions.tolist()} falls in a page past the "
            f"table, which names {pages} page(s) per request; the row and the position come "
            f"from one step and a position outside the row has no slot"
        )
    batch = int(true_start.shape[0])
    columns = block_table_row.to(torch.int64).t()[:batch]
    if bool((torch.gather(columns, 1, page_index) < 0).any()):
        raise MtpAsyncStepError(
            f"mtp_async_correct: a position of {positions.tolist()} falls in a page its "
            f"request's block-table column does not name (-1); a position outside the row "
            f"has no slot"
        )


def mtp_async_correct_torch(
    prev_checkpoint_rows: Tensor, optimistic_starts: Tensor, block_table_row: Tensor | None, *,
    prev_width: int, width: int, page_size: int, padding: int,
) -> StepCorrection:
    """The correction as torch ops: the contract, and the CPU route.

    The real rows' positions are ``optimistic - (prev_width - 1 - prev_checkpoint_rows)``;
    row ``t`` of request ``r`` sits at ``start[r] + t`` with causal length one more and
    its slot is ``table[position // page, r] * page + position % page``. A padding row
    is one token at position 0 of its column's first (null) block, as
    ``_glm5next_latent_slot_mapping`` lays it out with ``reals = [1]``.
    """
    batch, rows, pages = _check_correct(
        prev_checkpoint_rows, optimistic_starts, block_table_row,
        prev_width=prev_width, width=width, page_size=page_size, padding=padding,
    )
    shift = _page_shift(page_size)
    page = 1 << shift
    device = optimistic_starts.device
    width = int(width)
    rejected = (int(prev_width) - 1) - prev_checkpoint_rows.to(torch.int64)
    true_start = optimistic_starts.to(torch.int64) - rejected
    pad = int(padding)
    start_position = torch.cat([true_start, torch.zeros(pad, dtype=torch.int64, device=device)])
    linear_start = torch.cat([true_start, torch.ones(pad, dtype=torch.int64, device=device)])
    steps = torch.arange(width, dtype=torch.int64, device=device)
    # A padding row's causal length still steps per row (``_glm5next_batch_row_seq_lens``
    # over ``[(width, 0)]``) while its slots are clamped to position 0 (``reals = [1]``).
    seq_positions = start_position.reshape(rows, 1) + steps
    offsets = torch.cat([
        steps.expand(batch, width),
        torch.zeros((pad, width), dtype=torch.int64, device=device),
    ])
    positions = start_position.reshape(rows, 1) + offsets
    latent_slots = None
    if block_table_row is not None:
        page_index = positions >> shift
        columns = block_table_row.to(torch.int64).t()
        blocks = torch.gather(columns, 1, page_index.clamp(min=0, max=pages - 1))
        latent_slots = (blocks * page + (positions & (page - 1))).reshape(-1)
    seq_lens = (seq_positions + 1).to(torch.int32).reshape(-1)
    checkpoint_rows = torch.cat([
        prev_checkpoint_rows, torch.zeros(pad, dtype=torch.int32, device=device)
    ])
    return StepCorrection(
        start_position.to(torch.int32), linear_start.to(torch.int32), seq_lens, latent_slots,
        checkpoint_rows,
    )


@nki.jit
def mtp_async_correct_kernel(prev_rows, starts, table, PREV_WIDTH: int, WIDTH: int,
                             PAGE_SHIFT: int, PADDING: int, WITH_SLOTS: bool):
    """``(start [R, 1], linear [R, 1], seq [R, W], slots [R, 2W], rows [R, 1])`` int32.

    ``WITH_SLOTS`` False (a stack with no latent bank; ``table`` is then a ``[1, R]``
    stand-in that is not read): the slot output is zeros and the table is not touched.

    ``prev_rows`` and ``starts`` are ``[B, 1]`` int32, ``table`` is ``[pages, R]`` int32 with
    ``R = B + PADDING``; one partition per row of the step. The real rows: position
    ``starts - (PREV_WIDTH - 1 - prev_rows)`` plus the row offset (an iota), causal length
    one more, the block picked from the request's block-table column (read transposed,
    one strided DMA) by an equality mask over the pages, the slot
    ``block << PAGE_SHIFT | offset`` written as the low word of an int32 pair whose high
    word is zero. The device does not refuse a position past the column's pages (the
    host's ``_torch`` route does): the scheduler allocates every page a step writes. The padding rows: position 0 (sparse), 1 (linear),
    resume row 0, causal lengths ``1 .. W`` and every slot at offset 0 of their column's
    first block.
    """
    batch = prev_rows.shape[0]
    pages, rows = table.shape
    kernel_assert(rows == batch + PADDING, "R = B + padding")
    kernel_assert(rows <= PARTITIONS, "one partition per row")
    kernel_assert(WIDTH >= 1 and PREV_WIDTH >= 1, "W >= 1")
    page = 1 << PAGE_SHIFT
    start_out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    linear_out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    seq_out = nl.ndarray((rows, WIDTH), dtype=nl.int32, buffer=nl.shared_hbm)
    slots_out = nl.ndarray((rows, 2 * WIDTH), dtype=nl.int32, buffer=nl.shared_hbm)
    rows_out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    # ---- the real requests: partitions 0 .. B - 1 ---------------------------------- #
    prev_i = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=prev_i, src=prev_rows)
    start_i = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=start_i, src=starts)
    prev_f = nl.ndarray((batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=prev_f, src=prev_i)
    start_f = nl.ndarray((batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=start_f, src=start_i)
    # rejected = (PREV_WIDTH - 1) - prev; true = start - rejected.
    rejected_f = nl.ndarray((batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rejected_f, data=prev_f, op0=nl.subtract,
                       operand0=float(PREV_WIDTH - 1), reverse0=True)
    true_f = nl.ndarray((batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=true_f, data1=start_f, data2=rejected_f, op=nl.subtract)
    true_i = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=true_i, src=true_f)
    nisa.dma_copy(dst=start_out[0:batch, :], src=true_i)
    nisa.dma_copy(dst=linear_out[0:batch, :], src=true_i)
    nisa.dma_copy(dst=rows_out[0:batch, :], src=prev_i)
    # pos[b, t] = true[b] + t; seq = pos + 1.
    offset_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=offset_f, pattern=[[1, WIDTH]], offset=0, channel_multiplier=0)
    pos_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pos_f, data=offset_f, op0=nl.add, operand0=true_f)
    seq_i = nl.ndarray((batch, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=seq_i, data=pos_f, op0=nl.add, operand0=1.0)
    nisa.dma_copy(dst=seq_out[0:batch, :], src=seq_i)
    pos_i = nl.ndarray((batch, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_i, src=pos_f)
    page_i = nl.ndarray((batch, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=page_i, data=pos_i, op0=nl.right_shift, operand0=PAGE_SHIFT)
    within_i = nl.ndarray((batch, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=within_i, data=pos_i, op0=nl.bitwise_and, operand0=page - 1)
    if not WITH_SLOTS:
        zero_slots = nl.ndarray((rows, 2 * WIDTH), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=zero_slots, value=0)
        nisa.dma_copy(dst=slots_out, src=zero_slots)
    # Each request's block-table column as a row: column_t[r, page] = table[page, r], one
    # strided read of the layer's own operand (partition step 1, free step R). The block
    # of row t is the column entry at its page, picked by an equality mask over the pages
    # and summed (the other terms, -1 past the row's pages included, are multiplied by 0).
    column_f = nl.ndarray((rows, pages), dtype=nl.float32, buffer=nl.sbuf)
    if WITH_SLOTS:
        column_i = nl.ndarray((rows, pages), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=column_i, src=table.ap(pattern=[[1, rows], [rows, pages]]))
        nisa.tensor_copy(dst=column_f, src=column_i)
    page_index_f = nl.ndarray((batch, pages), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=page_index_f, pattern=[[1, pages]], offset=0, channel_multiplier=0)
    page_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=page_f, src=page_i)
    block_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(WIDTH if WITH_SLOTS else 0):
        hit = nl.ndarray((batch, pages), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hit, data=page_index_f, op0=nl.equal, operand0=page_f[:, t:t + 1])
        picked = nl.ndarray((batch, pages), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=picked, data1=column_f[0:batch, :], data2=hit, op=nl.multiply)
        nisa.tensor_reduce(dst=block_f[:, t:t + 1], op=nl.add, data=picked, axis=(1,))
    scaled_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scaled_f, data=block_f, op0=nl.multiply, operand0=float(page))
    within_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=within_f, src=within_i)
    slot_f = nl.ndarray((batch, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=slot_f, data1=scaled_f, data2=within_f, op=nl.add)
    slot_i = nl.ndarray((batch, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slot_i, src=slot_f)
    zero_i = nl.ndarray((batch, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=zero_i, value=0)
    # Low words at even columns, zero high words at odd ones: an int64 bit for bit.
    if WITH_SLOTS:
        nisa.dma_copy(dst=slots_out.ap(pattern=[[2 * WIDTH, batch], [2, WIDTH]], offset=0),
                      src=slot_i)
        nisa.dma_copy(dst=slots_out.ap(pattern=[[2 * WIDTH, batch], [2, WIDTH]], offset=1),
                      src=zero_i)

    # ---- the padding rows: partitions B .. R - 1 ------------------------------------ #
    if PADDING > 0:
        pad_zero = nl.ndarray((PADDING, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=pad_zero, value=0)
        pad_one = nl.ndarray((PADDING, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=pad_one, value=1)
        nisa.dma_copy(dst=start_out[batch:rows, :], src=pad_zero)
        nisa.dma_copy(dst=linear_out[batch:rows, :], src=pad_one)
        nisa.dma_copy(dst=rows_out[batch:rows, :], src=pad_zero)
        # A padding row's causal length steps per row from position 0: t + 1.
        pad_seq = nl.ndarray((PADDING, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=pad_seq, pattern=[[1, WIDTH]], offset=1, channel_multiplier=0)
        nisa.dma_copy(dst=seq_out[batch:rows, :], src=pad_seq)
        # The column's first block (the null block), every row at offset 0 of it.
        pad_slot_i = nl.ndarray((PADDING, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
        pad_zero_w = nl.ndarray((PADDING, WIDTH), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=pad_zero_w, value=0)
        if WITH_SLOTS:
            pad_scaled_f = nl.ndarray((PADDING, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pad_scaled_f, data=column_f[batch:rows, 0:1],
                               op0=nl.multiply, operand0=float(page))
            pad_ones_f = nl.ndarray((PADDING, WIDTH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=pad_ones_f, value=1.0)
            nisa.tensor_scalar(dst=pad_slot_i, data=pad_ones_f, op0=nl.multiply,
                               operand0=pad_scaled_f)
        if WITH_SLOTS:
            nisa.dma_copy(dst=slots_out.ap(pattern=[[2 * WIDTH, PADDING], [2, WIDTH]],
                                           offset=batch * 2 * WIDTH), src=pad_slot_i)
            nisa.dma_copy(dst=slots_out.ap(pattern=[[2 * WIDTH, PADDING], [2, WIDTH]],
                                           offset=batch * 2 * WIDTH + 1), src=pad_zero_w)
    return start_out, linear_out, seq_out, slots_out, rows_out


def mtp_async_correct(
    prev_checkpoint_rows: Tensor, optimistic_starts: Tensor, block_table_row: Tensor | None, *,
    prev_width: int, width: int, page_size: int, padding: int,
) -> StepCorrection:
    """:class:`StepCorrection` for the step being built; the kernel where kernels run, else the torch route.

    Args:
        prev_checkpoint_rows: ``[B]`` int32, each request's resume row from the previous
            step's take (``kept - 1``; 0 when that step was a prefill or a one-row decode,
            with ``prev_width = 1``). A device future is read by the kernel only.
        optimistic_starts: ``[B]`` int32, the scheduler's cached length per request, which
            counts every row of the previous step as accepted.
        block_table_row: ``[pages, R]`` int32, the sparse carrier's column form: one column
            per request and per padding row (``R = B + padding``), ``-1`` past a row's pages,
            a padding column naming the null block first; ``None`` for a stack with no
            latent bank, which gets ``latent_slots`` None.
        prev_width: rows per request of the previous step (``1 + k`` after a verify step).
        width: rows per request of this step (``1 + k`` on a verify step, 1 on a one-row decode).
        page_size: the latent bank's page, a power of two.
        padding: the bucket's padding rows past the ``B`` requests.

    Raises:
        MtpAsyncStepError: another geometry or dtype, a page size that is not a power of
            two, or (when the values are on the host) a resume row outside its step, a
            corrected position below zero, or a position whose page the table does not name.
    """
    batch, rows, _pages = _check_correct(
        prev_checkpoint_rows, optimistic_starts, block_table_row,
        prev_width=prev_width, width=width, page_size=page_size, padding=padding,
    )
    shift = _page_shift(page_size)
    if not can_run_kernel(optimistic_starts):
        correction = mtp_async_correct_torch(
            prev_checkpoint_rows, optimistic_starts, block_table_row,
            prev_width=prev_width, width=width, page_size=page_size, padding=padding,
        )
        _count_torch_route()
        return correction
    with_slots = block_table_row is not None
    table = (
        block_table_row.contiguous() if with_slots
        else torch.zeros((1, rows), dtype=torch.int32, device=optimistic_starts.device)
    )
    start, linear, seq, slots, resume = _CORRECT_KERNEL(
        prev_rows=prev_checkpoint_rows.reshape(batch, 1).contiguous(),
        starts=optimistic_starts.reshape(batch, 1).contiguous(),
        table=table,
        PREV_WIDTH=int(prev_width), WIDTH=int(width), PAGE_SHIFT=shift, PADDING=int(padding),
        WITH_SLOTS=with_slots,
    )
    _count_kernel()
    return StepCorrection(
        start.reshape(rows), linear.reshape(rows), seq.reshape(-1),
        slots.view(torch.int64).reshape(-1) if with_slots else None, resume.reshape(rows),
    )


_TAKE_KERNEL = wrap_nki(mtp_async_take_kernel)
_CORRECT_KERNEL = wrap_nki(mtp_async_correct_kernel)
_COUNTERS = DispatchCounters()
_count_kernel = count_kernel(_COUNTERS)
_count_torch_route = count_torch_route(_COUNTERS)


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.reset()


def dispatch_counters() -> tuple[int, int]:
    """``(kernel launches that returned, torch-route calls)`` since the last reset."""
    return _COUNTERS.pair()
