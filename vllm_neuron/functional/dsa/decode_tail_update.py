# SPDX-License-Identifier: Apache-2.0
"""Decode-step tail update for the sparse-attention indexer's k-pool.

The indexer pools ``index_kpool`` consecutive tokens into one key. Prefill sees
whole pools and is served by :mod:`vllm_neuron.functional.dsa.kpool_hadamard`;
decode sees one token at a time, so this module keeps the raw tokens of a
request's incomplete trailing pool in a small ring -- keys in half 0, gate scores
in half 1 -- and advances it by one token per call. The completion read runs
before the stash, so a pool ending on this token is compressed from the prior
stashes plus the current token, and the stash is unconditional: gating it on
completion drops every intra-pool token and leaves stale ring rows in the next
pool. Ring row and pool slot are the same number, because a pool's first
position is always a multiple of ``pool_size``.

The completion rounds to bfloat16 twice, once on the pooled vector before the
Hadamard rotation and once on the rotated vector after it. That makes this path
deliberately not bit-identical to the prefill kernel, which carries fp32 into the
butterfly and casts once at the end.

The two entry points differ only in where the position lives.
:func:`dsa_decode_tail_update` takes a python int, so the kernel's ``slot`` is a
compile-time address and the completion branch resolves at trace time.
:func:`dsa_decode_tail_update_at` takes a tensor instead, and gets the same
constant address by rotating the ring so this token's row is the last one,
calling the kernel with ``slot = pool_size - 1`` and rotating the result back. At
the slot that ends a pool that rotation is the identity permutation, so the two
entry points agree bit for bit wherever the pooled value means anything.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    INDEX_HEAD_DIM,
    _fwht128_inplace,
    hadamard_matrix,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

DEFAULT_POOL_SIZE = 4
"""``index_kpool`` on the target checkpoint's config. Not a limit -- see :func:`_validate`."""

_SUPPORTED_DTYPES = (torch.bfloat16,)
"""Ring and token dtypes that take the NKI route. The reference asserts bf16 as well."""

TAIL_HALVES = 2
"""The ring holds two halves: keys at half 0 and gate scores at half 1."""


class DecodeTailUpdateError(ValueError):
    """A malformed call: wrong rank, mismatched shapes, a bad pool size, or a negative position."""


# ``_fwht128_inplace`` is private to ``kpool_hadamard`` and imported anyway: the prefill and the
# decode path must rotate with the same code, or a divergence between them would be invisible to
# both of their tests. ``HADAMARD_SCALE`` and ``INDEX_HEAD_DIM`` come along for the same reason.


@dataclass
class _DecodeTailDispatchCounters:
    """Per-process dispatch counters for this module. One step is one dispatch."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _DecodeTailDispatchCounters()


def reset_decode_tail_dispatch_counters() -> None:
    """Zero this module's dispatch counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def decode_tail_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def decode_tail_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator instead:
    ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose ``__module__`` is
    ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


def _row_pattern(rows: int, head_dim: int) -> list[list[int]]:
    """The access pattern for ``rows`` contiguous rows of a 2-D buffer of width ``head_dim``.

    One spelling for every read and every write in this kernel, so a stride mistake can only be
    made once. The row stride is the row width, because nothing here is strided over pools.
    """
    return [[head_dim, rows], [1, head_dim]]


def _load_rows(hbm, rows: int, head_dim: int, first_row: int):
    """``rows`` contiguous rows of a 2-D HBM buffer as one tile, in the source dtype.

    Used for the ring rows this step carries unchanged, so the carry is an exact byte copy with
    no widening and no rounding involved.
    """
    out = nl.ndarray((rows, head_dim), dtype=hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out, src=hbm.ap(pattern=_row_pattern(rows, head_dim), offset=first_row * head_dim))
    return out


def _load_rows_fp32(hbm, rows: int, head_dim: int, first_row: int):
    """``rows`` contiguous rows of a 2-D HBM buffer as one fp32 tile, starting at ``first_row``.

    Used for every row that enters the arithmetic. The DMA lands in a tile of the source dtype
    and a separate ``tensor_copy`` widens it; that staging is unconditional rather than guarded
    on ``hbm.dtype``, because comparing a NKI tensor's dtype against an ``nki.language`` dtype
    object is a trace-time equality this file would have to be right about.

    Every row is read at its own offset and never sliced out of a wider tile, because slicing a
    partition range out of an SBUF tile has no precedent in this tree -- sliced operands here
    slice the free axis. One row per read also means no operand ever needs a partition broadcast.
    """
    out = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=_load_rows(hbm, rows, head_dim, first_row))
    return out


def _cast_row(src, head_dim: int, dtype):
    """One ``(1, head_dim)`` tile, cast to ``dtype``."""
    out = nl.ndarray((1, head_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=src)
    return out


# ---------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------


@nki.jit
def _decode_tail_update_nki(tail_hbm, key_hbm, score_hbm, ape_hbm, pool_size, slot):
    """Advance the k-pool tail ring by one token, and compress a pool if this one ends it.

    Args:
        tail_hbm: ``[2 * pool_size, head_dim]`` -- the ring, flattened. Rows
            ``0 .. pool_size-1`` are the key half and rows ``pool_size .. 2*pool_size-1`` the
            gate-score half.
        key_hbm: ``[1, head_dim]`` -- this token's indexer key.
        score_hbm: ``[1, head_dim]`` -- this token's gate score.
        ape_hbm: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias, applied inside
            the softmax.
        pool_size: tokens per pool. A compile-time constant.
        slot: ``position % pool_size`` for this token. A compile-time constant, because it is
            an address.

    Returns:
        ``(pooled, tail_out)``. ``pooled`` is ``[1, head_dim]`` in the ring's dtype: the
        completed pool's compressed key when ``slot == pool_size - 1``, and zero otherwise --
        the host seam turns that into ``None`` rather than making a reader interpret zeros.
        ``tail_out`` is ``[2 * pool_size, head_dim]``, the ring with this token stashed at row
        ``slot`` of each half.

    The advanced ring is returned rather than the argument mutated, because an input HBM tensor
    is not an output at the NKI boundary. The step therefore copies the ``2 * pool_size - 2``
    rows it does not change -- 2 KiB in and 2 KiB out per token at the target geometry, small
    beside the attention step it sits inside.

    Because ``slot`` is a compile-time constant, the completion branch resolves at trace time:
    a non-completing step traces no softmax at all, and a completing step traces no select over
    "is this member the current token".
    """
    head_dim = tail_hbm.shape[1]
    n_rows = TAIL_HALVES * pool_size
    pooled_hbm = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    tail_out_hbm = nl.ndarray((n_rows, head_dim), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)

    key_sb = _load_rows_fp32(key_hbm, 1, head_dim, 0)
    score_sb = _load_rows_fp32(score_hbm, 1, head_dim, 0)

    # ---- 1. The completion read, first. ----------------------------------------------------- #
    if slot == pool_size - 1:
        # Pass 1: the per-channel max of `score + ape` over the pool's members, for softmax
        # stability. The sums are kept rather than recomputed in pass 2, because they are
        # already in SBUF.
        totals = []
        running_max = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        for member in range(pool_size):
            # The current token comes from the argument; every other member from the ring's
            # score half, at the row its own slot addresses.
            if member == slot:
                score_src = score_sb
            else:
                score_src = _load_rows_fp32(tail_hbm, 1, head_dim, pool_size + member)
            bias = _load_rows_fp32(ape_hbm, 1, head_dim, member)
            total = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=total, data1=score_src, data2=bias, op=nl.add)
            totals.append(total)
            if member == 0:
                nisa.tensor_copy(dst=running_max, src=total)
            else:
                nisa.tensor_tensor(dst=running_max, data1=running_max, data2=total, op=nl.maximum)

        # Pass 2: the softmax-weighted sum of the members' keys.
        acc = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        denom = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        nisa.memset(dst=denom, value=0.0)
        for member in range(pool_size):
            shifted = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=shifted, data1=totals[member], data2=running_max, op=nl.subtract)
            weight = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=weight, op=nl.exp, data=shifted)
            nisa.tensor_tensor(dst=denom, data1=denom, data2=weight, op=nl.add)
            if member == slot:
                key_src = key_sb
            else:
                key_src = _load_rows_fp32(tail_hbm, 1, head_dim, member)
            weighted = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=weighted, data1=weight, data2=key_src, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=weighted, op=nl.add)

        # Reciprocal-then-multiply, not a divide: the form the ISA exposes directly.
        inv = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=denom)
        pooled = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)

        # First bfloat16 round trip, before the rotation. Reproduced rather than skipped: it is
        # a real quantisation of the pooled vector, and dropping it would move this path away
        # from the reference by more than the rotation's own rounding.
        pooled_bf = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pooled_bf, src=pooled)
        pooled_rt = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pooled_rt, src=pooled_bf)

        scratch = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        rotated = _fwht128_inplace(pooled_rt, scratch, head_dim)
        scaled = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
        # Second bfloat16 round trip. Here it is also the output cast, so one copy serves both.
        result = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=scaled)
        nl.store(pooled_hbm, value=result)
    else:
        # No pool ends here. A zero row, so the returned tensor is defined for every trace and
        # the host decides what it means.
        empty = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.sbuf)
        nisa.memset(dst=empty, value=0.0)
        nl.store(pooled_hbm, value=empty)

    # ---- 2. The stash, second, and for this token whether or not a pool ended. -------------- #
    # Every output row is written exactly once: two rows come from this token and the rest are
    # carried across in their own dtype. Writing the prior ring wholesale and then overwriting
    # two of its rows would put two writes on the same addresses inside one kernel, and their
    # ordering is not something this file should have to assume.
    key_out = _cast_row(key_sb, head_dim, tail_hbm.dtype)
    score_out = _cast_row(score_sb, head_dim, tail_hbm.dtype)
    for row in range(n_rows):
        if row == slot:
            src = key_out
        elif row == pool_size + slot:
            src = score_out
        else:
            src = _load_rows(tail_hbm, 1, head_dim, row)
        nl.store(
            tail_out_hbm.ap(pattern=_row_pattern(1, head_dim), offset=row * head_dim), value=src
        )

    return pooled_hbm, tail_out_hbm


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(entry: str, pool_size: int, slot: int, head_dim: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    A folded helper takes ints, strings and dtypes only: Dynamo runs it at trace time and first
    converts every non-tensor argument into a python constant, and an ``@nki.jit`` kernel is a
    frozen dataclass it refuses to reconstruct. The kernel is therefore read as a module global,
    the same object the call site hands to ``wrap_nki`` on the next line.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_decode_tail_update_nki)
    logger.info(
        "[dsa-decode-tail-update] kernel=nki entry=%s pool_size=%d slot=%d head_dim=%d",
        entry,
        pool_size,
        slot,
        head_dim,
    )


@torch._dynamo.assume_constant_result
def _record_nki_dispatch_at(entry: str, pool_size: int, head_dim: int) -> None:
    """The same record for the tensor-position seam, without a slot.

    The slot is absent rather than logged, because on this seam it lives in a tensor and a
    folded helper would convert it to a python constant at trace time -- the host read this seam
    exists to remove. The kernel's own slot is the constant ``pool_size - 1`` on every call here.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_decode_tail_update_nki)
    logger.info(
        "[dsa-decode-tail-update] kernel=nki entry=%s pool_size=%d slot=last head_dim=%d",
        entry,
        pool_size,
        head_dim,
    )


def slot_of(position: int, pool_size: int) -> int:
    """The ring row this position writes: ``position % pool_size``.

    Public because the caller needs the same arithmetic to decide when a pool ends, and two
    spellings of one rule are how they drift apart.
    """
    if position < 0:
        raise DecodeTailUpdateError(f"position must not be negative; got {position}")
    if pool_size <= 0:
        raise DecodeTailUpdateError(f"pool_size must be positive; got {pool_size}")
    return position % pool_size


def completes_pool(position: int, pool_size: int) -> bool:
    """Whether the token at ``position`` ends a pool. A negative position raises rather than
    being treated as a padded entry."""
    return slot_of(position, pool_size) == pool_size - 1


def decode_pool_address(
    position: Tensor | int, pool_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Where this token's pooled key belongs, and whether it has one. Two 0-d tensors.

    Returns ``(pool_index, completes)``: ``position // pool_size``, the id of the pool this
    token would end, and whether it ends one at all, answered on device. A negative position is
    not refused here, because refusing it would read the value; :func:`slot_of` still refuses
    one where the position is a python int. This is the third spelling of one rule, kept in the
    same module as the other two so that they cannot drift apart unnoticed.
    """
    at = torch.as_tensor(position, device=device, dtype=torch.int64).reshape(())
    return (
        torch.div(at, pool_size, rounding_mode="floor"),
        torch.remainder(at, pool_size) == pool_size - 1,
    )


def _ring_permutations(
    slot: Tensor, pool_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """The permutation that moves ring row ``slot`` to the last row, and the one that undoes it.

    Row ``m`` of the rotated ring is row ``(slot + 1 + m) % pool_size`` of the original, so the
    last rotated row is ``slot`` itself, and the inverse sends rotated row
    ``(j - slot - 1) % pool_size`` back to row ``j``. At ``slot == pool_size - 1`` both are the
    identity, which is what makes the two entry points agree exactly.
    """
    rows = torch.arange(pool_size, device=device)
    return (
        torch.remainder(slot + 1 + rows, pool_size),
        torch.remainder(rows - slot - 1, pool_size),
    )


def _validate(tail: Tensor, key: Tensor, score: Tensor, ape: Tensor) -> tuple[int, int]:
    """Host-side shape and dtype validation. Returns ``(pool_size, head_dim)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor value, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace.
    """
    if tail.ndim != 3 or int(tail.shape[0]) != TAIL_HALVES:
        raise DecodeTailUpdateError(
            f"tail must be [{TAIL_HALVES}, pool_size, head_dim]; got shape {tuple(tail.shape)}"
        )
    pool_size, head_dim = int(tail.shape[1]), int(tail.shape[2])
    if pool_size <= 0:
        raise DecodeTailUpdateError(f"pool_size must be positive; got {pool_size}")
    if head_dim != INDEX_HEAD_DIM:
        raise DecodeTailUpdateError(
            f"the rotation is a {INDEX_HEAD_DIM}-point transform; got head_dim {head_dim}"
        )
    for name, tensor in (("key", key), ("score", score)):
        if tensor.ndim != 2 or tuple(tensor.shape) != (1, head_dim):
            raise DecodeTailUpdateError(
                f"{name} must be [1, head_dim] = {(1, head_dim)}; got {tuple(tensor.shape)}"
            )
    if ape.ndim != 2 or tuple(ape.shape) != (pool_size, head_dim):
        raise DecodeTailUpdateError(
            f"ape must be [pool_size, head_dim] = {(pool_size, head_dim)}; got {tuple(ape.shape)}"
        )
    return pool_size, head_dim


def can_run_dsa_decode_tail_update(tail: Tensor, key: Tensor, score: Tensor, ape: Tensor) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch fallback."""
    if not can_run_kernel():
        return False
    if tail.dtype not in _SUPPORTED_DTYPES or key.dtype not in _SUPPORTED_DTYPES:
        return False
    if score.dtype not in _SUPPORTED_DTYPES:
        return False
    if tail.ndim != 3 or int(tail.shape[0]) != TAIL_HALVES:
        return False
    if int(tail.shape[2]) != INDEX_HEAD_DIM:
        return False
    if tuple(key.shape) != (1, INDEX_HEAD_DIM) or tuple(score.shape) != tuple(key.shape):
        return False
    return ape.ndim == 2 and tuple(ape.shape) == (int(tail.shape[1]), INDEX_HEAD_DIM)


def dsa_decode_tail_update(
    tail: Tensor, key: Tensor, score: Tensor, ape: Tensor, position: int
) -> tuple[Tensor | None, Tensor]:
    """Advance the tail ring by one decode token, at a position known on the host.

    Args:
        tail: ``[2, pool_size, head_dim]`` bf16 -- the ring. Half 0 holds keys, half 1 holds
            gate scores.
        key: ``[1, head_dim]`` bf16 -- this token's indexer key.
        score: ``[1, head_dim]`` bf16 -- this token's gate score.
        ape: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        position: this token's absolute position in the request, as a python int. Use
            :func:`dsa_decode_tail_update_at` when the position lives in a tensor: the compiled
            graph specialises on ``slot``, so a graph captured at one position and replayed at
            another would write the ring row it was captured with.

    Returns:
        ``(pooled, new_tail)``. ``pooled`` is ``[1, head_dim]`` when this token completed a
        pool and ``None`` when it did not; ``new_tail`` has the same shape as ``tail``. The
        caller threads ``new_tail`` into the next step -- the ring is state, and this function
        does not mutate its argument.

    Raises:
        DecodeTailUpdateError: for a malformed call or a negative position.
    """
    pool_size, head_dim = _validate(tail, key, score, ape)
    slot = slot_of(position, pool_size)
    ends_pool = slot == pool_size - 1

    if not can_run_dsa_decode_tail_update(tail, key, score, ape):
        return _dsa_decode_tail_update_torch(tail, key, score, ape, position)

    flat_tail = tail.reshape(TAIL_HALVES * pool_size, head_dim).contiguous()

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter
    # store inside the trace becomes a value guard that fails on the first call after warmup.
    _record_nki_dispatch("step", pool_size, slot, head_dim)
    pooled, new_flat = wrap_nki(_decode_tail_update_nki)(
        flat_tail, key.contiguous(), score.contiguous(), ape.contiguous(), pool_size, slot
    )
    new_tail = new_flat.reshape(TAIL_HALVES, pool_size, head_dim)
    return (pooled if ends_pool else None), new_tail


def dsa_decode_tail_update_at(
    tail: Tensor, key: Tensor, score: Tensor, ape: Tensor, position: Tensor | int
) -> tuple[Tensor, Tensor]:
    """Advance the tail ring by one decode token, at a position this process never reads.

    Args:
        tail: ``[2, pool_size, head_dim]`` bf16 -- the ring, in the same slot-addressed layout
            :func:`dsa_decode_tail_update` reads and writes. The rotation below is internal to
            one call; no stored ring changes shape or meaning.
        key: ``[1, head_dim]`` bf16 -- this token's indexer key.
        score: ``[1, head_dim]`` bf16 -- this token's gate score.
        ape: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        position: this token's absolute position, as a tensor of any integer dtype (a python int
            is admitted and gives the same answer). It is never read on the host.

    Returns:
        ``(pooled, new_tail)``. ``pooled`` is ``[1, head_dim]`` always -- there is no ``None``
        here, because whether a pool ended is a value on device and a return type cannot depend
        on one. The row is this token's completed pool when :func:`decode_pool_address` says it
        completes one and is meaningless otherwise, so the caller addresses it accordingly, as
        the prefill leg's ``pool_window`` already does for its own non-completions.
        ``new_tail`` has ``tail``'s shape and layout, with this token stashed at its own slot.

    The rotation is the identity on the step whose value is used, so this entry point and
    :func:`dsa_decode_tail_update` agree bit for bit wherever the answer means anything. A step
    that ends no pool still runs one ``pool_size``-member softmax the caller discards, plus
    three permuted reads of at most ``2 * pool_size`` rows.
    """
    pool_size, head_dim = _validate(tail, key, score, ape)
    device = tail.device
    slot = torch.remainder(
        torch.as_tensor(position, device=device, dtype=torch.int64).reshape(()), pool_size
    )
    forward_perm, inverse_perm = _ring_permutations(slot, pool_size, device)
    tail_rotated = tail.index_select(1, forward_perm)
    ape_rotated = ape.index_select(0, forward_perm)
    last = pool_size - 1

    if not can_run_dsa_decode_tail_update(tail, key, score, ape):
        # The rotated ring at the last slot is exactly this fallback's own case, so it serves it
        # unchanged rather than a second single-step implementation appearing here. It counts,
        # as it must: this is still the torch route.
        pooled, new_rotated = _dsa_decode_tail_update_torch(
            tail_rotated, key, score, ape_rotated, last
        )
    else:
        flat_tail = tail_rotated.reshape(TAIL_HALVES * pool_size, head_dim).contiguous()
        _count_nki_dispatch()
        _record_nki_dispatch_at("step_at", pool_size, head_dim)
        pooled, new_flat = wrap_nki(_decode_tail_update_nki)(
            flat_tail,
            key.contiguous(),
            score.contiguous(),
            ape_rotated.contiguous(),
            pool_size,
            last,
        )
        new_rotated = new_flat.reshape(TAIL_HALVES, pool_size, head_dim)

    return pooled, new_rotated.index_select(1, inverse_perm)


# ---------------------------------------------------------------------------------------------
# Torch: the fallback, and the batch reference
# ---------------------------------------------------------------------------------------------
# Two functions rather than one, because the fallback counts a dispatch and the reference must
# not: a test that asserts zero fallbacks while comparing against a torch recompute needs the
# recompute to leave the counters alone.


def _compress_pool_torch(pool_key: Tensor, pool_score: Tensor, ape: Tensor) -> Tensor:
    """One complete pool, compressed the way the decode path compresses it.

    ``pool_key`` and ``pool_score`` are ``[pool_size, head_dim]`` and the result is
    ``[1, head_dim]`` in ``pool_key``'s dtype. Both bfloat16 round trips are here, which is why
    this is not the prefill reference with a different input: that one keeps fp32 all the way to
    the output cast. ``dim=0`` is the slot axis, which is what makes the softmax per
    ``(slot, channel)``; ``dim=-1`` would be a whole-vector softmax and a different kernel. The
    Hadamard matrix follows its operand's device, because the factory builds on the default one.
    """
    dtype = pool_key.dtype
    weights = torch.softmax(pool_score.float() + ape.float(), dim=0)
    pooled = (weights * pool_key.float()).sum(dim=0, keepdim=True)
    pooled = pooled.to(dtype).float()
    matrix = hadamard_matrix(int(pool_key.shape[1])).to(pool_key.device)
    rotated = pooled @ matrix.t()
    return (rotated * HADAMARD_SCALE).to(dtype)


def _dsa_decode_tail_update_torch(
    tail: Tensor, key: Tensor, score: Tensor, ape: Tensor, position: int
) -> tuple[Tensor | None, Tensor]:
    """The single-step fallback, in torch. Counted, because it is a route the seam can take."""
    _count_torch_fallback()
    pool_size = int(tail.shape[1])
    slot = slot_of(position, pool_size)

    pooled: Tensor | None = None
    if slot == pool_size - 1:
        pool_key = tail[0].clone()
        pool_score = tail[1].clone()
        pool_key[slot] = key[0].to(pool_key.dtype)
        pool_score[slot] = score[0].to(pool_score.dtype)
        pooled = _compress_pool_torch(pool_key, pool_score, ape)

    new_tail = tail.clone()
    new_tail[0, slot] = key[0].to(new_tail.dtype)
    new_tail[1, slot] = score[0].to(new_tail.dtype)
    return pooled, new_tail


def decode_tail_recompute(
    tail0: Tensor, keys: Tensor, scores: Tensor, ape: Tensor, start_position: int
) -> tuple[list[Tensor], Tensor]:
    """What ``k`` stepped updates must equal, computed without stepping.

    Args:
        tail0: ``[2, pool_size, head_dim]`` -- the ring before the first step, as a prefill
            would have seeded it.
        keys: ``[k, head_dim]`` -- the ``k`` decode tokens' keys, in position order.
        scores: ``[k, head_dim]`` -- their gate scores.
        ape: ``[pool_size, head_dim]``.
        start_position: the absolute position of ``keys[0]``.

    Returns:
        ``(pooled_list, tail_k)`` -- one compressed key per pool that ended during the ``k``
        steps, in order, and the ring as it stands after all ``k``.

    It works from the token stream rather than from a ring: the members of a pool that ends at
    position ``p`` are the tokens at ``p - pool_size + 1 .. p``, taken from ``keys``/``scores``
    where they fall inside the decode window and from ``tail0`` where they were seeded before
    it. No stepped ring is consulted, so an error in the ring's addressing cannot hide by
    appearing on both sides. It calls neither entry point and touches no counter.
    """
    pool_size, head_dim = int(tail0.shape[1]), int(tail0.shape[2])
    k = int(keys.shape[0])
    dtype = tail0.dtype

    def member(pos: int, half: int) -> Tensor:
        """The key (``half == 0``) or score (``half == 1``) of the token at absolute ``pos``."""
        if pos >= start_position:
            src = keys if half == 0 else scores
            return src[pos - start_position].to(dtype)
        return tail0[half, pos % pool_size].to(dtype)

    pooled_list: list[Tensor] = []
    for step in range(k):
        pos = start_position + step
        if pos % pool_size != pool_size - 1:
            continue
        first = pos - pool_size + 1
        pool_key = torch.stack([member(p, 0) for p in range(first, pos + 1)])
        pool_score = torch.stack([member(p, 1) for p in range(first, pos + 1)])
        pooled_list.append(_compress_pool_torch(pool_key, pool_score, ape))

    tail_k = tail0.clone()
    for step in range(k):
        pos = start_position + step
        tail_k[0, pos % pool_size] = keys[step].to(dtype)
        tail_k[1, pos % pool_size] = scores[step].to(dtype)
    _ = head_dim  # read for the shape contract above; not needed again
    return pooled_list, tail_k


# ---------------------------------------------------------------------------------------
# T tokens per request in one step: the verify step's ring advance
# ---------------------------------------------------------------------------------------

_PARTITIONS = 128


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    return nl.ndarray((parts, 1), dtype=dtype, buffer=nl.sbuf)


def _log2(value):
    shift = 0
    while (1 << shift) < value:
        shift += 1
    return shift


def ring_depth_for(pool_size: int, max_rows: int) -> int:
    """Ring rows a config needs: a power of two, at least ``pool_size``, holding one step's
    ``max_rows`` tokens and the ``pool_size - 2`` more a rollback can still need
    (:func:`dsa_decode_ring_rows`). ``max_rows = 1`` gives ``pool_size``."""
    depth = int(pool_size)
    while depth < max_rows + pool_size - 2:
        depth *= 2
    return depth


def max_rows_for(depth: int, pool_size: int) -> int:
    """The most tokens one step may hand a ring of ``depth`` rows: the inverse of
    :func:`ring_depth_for`."""
    return int(depth) - int(pool_size) + 2


def _ring_member(ring, half, member, pool_size, depth, group_masks):
    """Pool member ``member``'s key (``half`` 0) or gate score (``half`` 1) off the ring,
    in fp32: ring row ``member + pool_size * g`` with ``g`` the pool's index within the
    ring, which ``group_masks[g]`` marks per partition (one group: a plain copy)."""
    h, width = ring.shape
    head_dim = width // (TAIL_HALVES * depth)
    src = _sb((h, head_dim), nl.float32)
    base = half * depth + member
    if len(group_masks) == 0:
        row0 = base * head_dim
        nisa.tensor_copy(dst=src, src=ring[:, row0:row0 + head_dim])
        return src
    picked = _sb((h, head_dim), ring.dtype)
    for g, mask in enumerate(group_masks):
        row0 = (base + g * pool_size) * head_dim
        nisa.tensor_copy_predicated(dst=picked, src=ring[:, row0:row0 + head_dim],
                                    predicate=mask)
    nisa.tensor_copy(dst=src, src=picked)
    return src


@nki.jit
def dsa_decode_ring_rows_kernel(tail_hbm, slots_hbm, key_hbm, score_hbm, ape_hbm, pos_hbm,
                                pool_size, rows, source_digest):
    """Advance ``B`` rings by ``rows`` tokens each; compress the pool each row would close.

    Args:
        tail_hbm: ``[slots, 2 * depth * head_dim]`` the ring bank, one flattened ring per
            slot: keys in rows ``0 .. depth - 1``, gate scores after them. ``depth`` is a
            power-of-two multiple of ``pool_size``; a token at absolute position ``p``
            lives at ring row ``p % depth``.
        slots_hbm: ``[B, 1]`` int32, each request's slot.
        key_hbm / score_hbm: ``[B * rows, head_dim]`` this step's indexer keys and gate
            scores, request-major: request ``b``'s row ``t`` is row ``b * rows + t``.
        ape_hbm: ``[pool_size, head_dim]`` fp32, the per-slot additive bias.
        pos_hbm: ``[B, 1]`` int32, each request's position of row 0.
        pool_size: tokens per pool, a power of two.
        rows: tokens per request this step, a trace-time int, at most
            ``depth - pool_size + 2``.
        source_digest: :data:`ROWS_SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``(pooled, rings)``: ``[B * rows, head_dim]``, row ``b * rows + t`` the compressed
        pool that ends at position ``pos[b] + t`` -- meaningful exactly when that position
        is ``pool_size - 1`` mod ``pool_size`` (the caller writes the rest to the slot's
        trash row) -- and ``[B, 2 * depth * head_dim]`` the advanced rings, token ``t``
        stashed at row ``(pos[b] + t) % depth`` of each half.

    Row ``t``'s pool members at positions ``pos + t - pool_size + 1 .. pos + t`` are this
    step's own rows where they fall at or after ``pos`` (read off the operands, never off
    the ring) and ring rows otherwise; where the position closes a pool those ring rows are
    ``m + pool_size * g`` with ``g`` the pool's index within the ring, picked by predicated
    copies so no address depends on data. Every read of the ring precedes every stash, so a
    stash may land on a row a later pool of the same step no longer needs. The compression
    is :func:`decode_batch.dsa_decode_ring_step_kernel`'s instruction sequence per row, so
    at ``depth == pool_size`` the two agree bit for bit, row by row.
    """
    batch = slots_hbm.shape[0]
    head_dim = key_hbm.shape[1]
    width = tail_hbm.shape[1]
    depth = width // (TAIL_HALVES * head_dim)
    groups = depth // pool_size
    shift = _log2(pool_size)
    last = pool_size - 1
    pooled_hbm = nl.ndarray((batch * rows, head_dim), dtype=tail_hbm.dtype,
                            buffer=nl.shared_hbm)
    rings_hbm = nl.ndarray((batch, width), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    for r0 in range(0, batch, _PARTITIONS):
        h = min(_PARTITIONS, batch - r0)
        ring = _sb((h, width), tail_hbm.dtype)
        if tail_hbm.shape[0] == 1:
            # One slot: the only valid slot is 0, so the read is static. (Tracing in CPU
            # simulation fills int operands with ones, and slot 1 is past this bank.)
            nisa.dma_copy(dst=ring, src=tail_hbm.ap(pattern=[[0, h], [1, width]], offset=0))
        else:
            slot_t = _col(h, nl.int32)
            nisa.dma_copy(dst=slot_t, src=slots_hbm.ap(pattern=[[1, h], [1, 1]], offset=r0))
            nisa.dma_copy(dst=ring, src=tail_hbm.ap(pattern=[[width, h], [1, width]],
                                                    vector_offset=slot_t, indirect_dim=0))
        keys_bf, scores_bf, keys_f, scores_f = [], [], [], []
        for t in range(rows):
            key_bf = _sb((h, head_dim), key_hbm.dtype)
            nisa.dma_copy(dst=key_bf,
                          src=key_hbm.ap(pattern=[[rows * head_dim, h], [1, head_dim]],
                                         offset=(r0 * rows + t) * head_dim))
            score_bf = _sb((h, head_dim), score_hbm.dtype)
            nisa.dma_copy(dst=score_bf,
                          src=score_hbm.ap(pattern=[[rows * head_dim, h], [1, head_dim]],
                                           offset=(r0 * rows + t) * head_dim))
            key_f = _sb((h, head_dim), nl.float32)
            nisa.tensor_copy(dst=key_f, src=key_bf)
            score_f = _sb((h, head_dim), nl.float32)
            nisa.tensor_copy(dst=score_f, src=score_bf)
            keys_bf.append(key_bf)
            scores_bf.append(score_bf)
            keys_f.append(key_f)
            scores_f.append(score_f)
        pos_i = _col(h, nl.int32)
        nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[1, h], [1, 1]], offset=r0))
        ones = _sb((h, head_dim), nl.float32)
        nisa.memset(dst=ones, value=1.0)

        for t in range(rows):
            # ---- which of the ring's pools this row's position falls in -----------------
            at_i = _col(h, nl.int32)
            nisa.tensor_scalar(dst=at_i, data=pos_i, op0=nl.add, operand0=t)
            group_masks = []
            if groups > 1:
                pool_i = _col(h, nl.int32)
                nisa.tensor_scalar(dst=pool_i, data=at_i, op0=nl.right_shift, operand0=shift)
                grp_i = _col(h, nl.int32)
                nisa.tensor_scalar(dst=grp_i, data=pool_i, op0=nl.bitwise_and,
                                   operand0=groups - 1)
                grp_f = _col(h, nl.float32)
                nisa.tensor_copy(dst=grp_f, src=grp_i)
                for g in range(groups):
                    hit = _col(h, nl.float32)
                    nisa.tensor_scalar(dst=hit, data=grp_f, op0=nl.equal, operand0=float(g))
                    mask = _sb((h, head_dim), nl.uint8)
                    nisa.tensor_scalar(dst=mask, data=ones, op0=nl.multiply, operand0=hit)
                    group_masks.append(mask)

            # ---- the pool this row would close: members 0 .. last-1 from the ring ------
            totals = []
            running_max = _sb((h, head_dim), nl.float32)
            for member in range(pool_size):
                if member == last:
                    score_src = scores_f[t]
                elif t - last + member >= 0:
                    score_src = scores_f[t - last + member]
                else:
                    score_src = _ring_member(ring, 1, member, pool_size, depth, group_masks)
                bias = _sb((h, head_dim), nl.float32)
                nisa.dma_copy(dst=bias, src=ape_hbm.ap(pattern=[[0, h], [1, head_dim]],
                                                       offset=member * head_dim))
                total = _sb((h, head_dim), nl.float32)
                nisa.tensor_tensor(dst=total, data1=score_src, data2=bias, op=nl.add)
                totals.append(total)
                if member == 0:
                    nisa.tensor_copy(dst=running_max, src=total)
                else:
                    nisa.tensor_tensor(dst=running_max, data1=running_max, data2=total,
                                       op=nl.maximum)
            acc = _sb((h, head_dim), nl.float32)
            denom = _sb((h, head_dim), nl.float32)
            nisa.memset(dst=acc, value=0.0)
            nisa.memset(dst=denom, value=0.0)
            for member in range(pool_size):
                shifted = _sb((h, head_dim), nl.float32)
                nisa.tensor_tensor(dst=shifted, data1=totals[member], data2=running_max,
                                   op=nl.subtract)
                weight = _sb((h, head_dim), nl.float32)
                nisa.activation(dst=weight, op=nl.exp, data=shifted)
                nisa.tensor_tensor(dst=denom, data1=denom, data2=weight, op=nl.add)
                if member == last:
                    key_src = keys_f[t]
                elif t - last + member >= 0:
                    key_src = keys_f[t - last + member]
                else:
                    key_src = _ring_member(ring, 0, member, pool_size, depth, group_masks)
                weighted = _sb((h, head_dim), nl.float32)
                nisa.tensor_tensor(dst=weighted, data1=weight, data2=key_src, op=nl.multiply)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=weighted, op=nl.add)
            inv = _sb((h, head_dim), nl.float32)
            nisa.reciprocal(dst=inv, data=denom)
            pooled = _sb((h, head_dim), nl.float32)
            nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)
            pooled_bf = _sb((h, head_dim), tail_hbm.dtype)
            nisa.tensor_copy(dst=pooled_bf, src=pooled)
            pooled_rt = _sb((h, head_dim), nl.float32)
            nisa.tensor_copy(dst=pooled_rt, src=pooled_bf)
            scratch = _sb((h, head_dim), nl.float32)
            rotated = _fwht128_inplace(pooled_rt, scratch, head_dim)
            scaled = _sb((h, head_dim), nl.float32)
            nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply,
                               operand0=HADAMARD_SCALE)
            result = _sb((h, head_dim), tail_hbm.dtype)
            nisa.tensor_copy(dst=result, src=scaled)
            nisa.dma_copy(dst=pooled_hbm.ap(pattern=[[rows * head_dim, h], [1, head_dim]],
                                            offset=(r0 * rows + t) * head_dim),
                          src=result)

        # ---- the stashes: token t at row (pos + t) % depth of each half ---------------
        # Every ring read above is done, so a stash may take a row an earlier pool of this
        # step read: the next step's pools no longer need it (depth >= rows + pool - 2).
        for t in range(rows):
            pos_t = _col(h, nl.int32)
            nisa.tensor_scalar(dst=pos_t, data=pos_i, op0=nl.add, operand0=t)
            at_i = _col(h, nl.int32)
            nisa.tensor_scalar(dst=at_i, data=pos_t, op0=nl.bitwise_and, operand0=depth - 1)
            at_f = _col(h, nl.float32)
            nisa.tensor_copy(dst=at_f, src=at_i)
            for row in range(depth):
                hit = _col(h, nl.float32)
                nisa.tensor_scalar(dst=hit, data=at_f, op0=nl.equal, operand0=float(row))
                mask = _sb((h, head_dim), nl.uint8)
                nisa.tensor_scalar(dst=mask, data=ones, op0=nl.multiply, operand0=hit)
                for half in range(TAIL_HALVES):
                    row0 = (half * depth + row) * head_dim
                    nisa.tensor_copy_predicated(dst=ring[:, row0:row0 + head_dim],
                                                src=keys_bf[t] if half == 0 else scores_bf[t],
                                                predicate=mask)
        nisa.dma_copy(dst=rings_hbm.ap(pattern=[[width, h], [1, width]], offset=r0 * width),
                      src=ring)
    return pooled_hbm, rings_hbm


ROWS_SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)
"""A digest of this file, passed to the rows kernel so the kernel cache keys on its source."""


def _require_rows(tail_bank, slots, key, score, ape, position) -> tuple[int, int, int, int]:
    """Check a :func:`dsa_decode_ring_rows` call; return ``(batch, rows, pool, depth)``."""
    if tail_bank.ndim != 4 or int(tail_bank.shape[1]) != TAIL_HALVES:
        raise DecodeTailUpdateError(
            f"tail must be the ring bank [slots, {TAIL_HALVES}, depth, head_dim]; got "
            f"{tuple(tail_bank.shape)}")
    depth, head_dim = int(tail_bank.shape[2]), int(tail_bank.shape[3])
    if head_dim != INDEX_HEAD_DIM:
        raise DecodeTailUpdateError(f"the rotation is a {INDEX_HEAD_DIM}-point transform; got "
                                    f"head_dim {head_dim}")
    if ape.ndim != 2 or int(ape.shape[1]) != head_dim:
        raise DecodeTailUpdateError(f"ape must be [pool_size, {head_dim}]; got {tuple(ape.shape)}")
    pool = int(ape.shape[0])
    if pool < 2 or pool & (pool - 1):
        raise DecodeTailUpdateError(f"ape's pool_size must be a power of two >= 2; got {pool}")
    if depth < pool or depth & (depth - 1):
        raise DecodeTailUpdateError(
            f"the ring depth must be a power-of-two multiple of pool_size {pool}; got "
            f"{depth} rows")
    if not torch.is_tensor(slots) or slots.ndim != 1 or int(slots.shape[0]) < 1:
        raise DecodeTailUpdateError(f"slots must be a [B] tensor; got {slots!r}")
    batch = int(slots.shape[0])
    if tuple(position.shape) != (batch,):
        raise DecodeTailUpdateError(f"position must be [{batch}], one entry per request; got "
                                    f"{tuple(position.shape)}")
    for name, value in (("slots", slots), ("position", position)):
        if value.dtype not in (torch.int32, torch.int64):
            raise DecodeTailUpdateError(f"{name} must be int32 or int64; got {value.dtype}")
    if key.ndim != 2 or int(key.shape[1]) != head_dim or tuple(score.shape) != tuple(key.shape):
        raise DecodeTailUpdateError(f"key and score must both be [B * rows, {head_dim}]; got "
                                    f"{tuple(key.shape)} and {tuple(score.shape)}")
    total = int(key.shape[0])
    if total < batch or total % batch:
        raise DecodeTailUpdateError(
            f"key must hold a whole number of rows per request: {total} rows do not split "
            f"over {batch} requests")
    rows = total // batch
    if rows > max_rows_for(depth, pool):
        raise DecodeTailUpdateError(
            f"{rows} rows per request exceed what a ring of depth {depth} can take back on "
            f"rollback: at most {max_rows_for(depth, pool)} (depth - pool_size + 2), or a "
            f"rejected row would overwrite a token the next step's pools still need")
    return batch, rows, pool, depth


def dsa_decode_ring_rows(tail_bank: Tensor, slots: Tensor, key: Tensor, score: Tensor,
                         ape: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
    """Advance each request's ring by its ``rows`` tokens in one step. Nothing is written
    in place.

    Args:
        tail_bank: ``[slots, 2, depth, head_dim]`` bf16, the ring bank (keys in half 0,
            gate scores in half 1), read at ``slots`` only. ``depth`` is a power-of-two
            multiple of the pool size (:func:`ring_depth_for`).
        slots: ``[B]`` int, each request's slot; distinct.
        key / score: ``[B * rows, head_dim]`` bf16, request-major: request ``b``'s token
            ``t`` is row ``b * rows + t``. ``rows`` is read off the shapes.
        ape: ``[pool_size, head_dim]`` fp32.
        position: ``[B]`` int, each request's position of token 0.

    Returns:
        ``(pooled, rings)``: ``[B * rows, head_dim]``, row ``b * rows + t`` the pool that
        closes at ``position[b] + t`` where that position is ``pool_size - 1`` mod
        ``pool_size`` (meaningless otherwise), and ``[B, 2, depth, head_dim]`` the advanced
        rings, for the caller to write back at ``slots``.

    Refuses ``rows > depth - pool_size + 2``: a rejected row's stash would then alias a
    ring row the next step's pools still need.
    """
    batch, rows, pool, depth = _require_rows(tail_bank, slots, key, score, ape, position)
    usable = (can_run_kernel(key) and tail_bank.dtype == torch.bfloat16
              and key.dtype == torch.bfloat16 and score.dtype == torch.bfloat16)
    if not usable:
        _count_torch_fallback()
        return dsa_decode_ring_rows_torch_oracle(tail_bank, slots, key, score, ape, position)
    _count_nki_dispatch()
    _COUNTERS.last_kernel = _kernel_identity_of(dsa_decode_ring_rows_kernel)
    width = TAIL_HALVES * depth * int(tail_bank.shape[3])
    pooled, rings = wrap_nki(dsa_decode_ring_rows_kernel)(
        tail_bank.reshape(int(tail_bank.shape[0]), width).contiguous(),
        slots.reshape(batch, 1).to(torch.int32).contiguous(),
        key.contiguous(), score.contiguous(), ape.to(torch.float32).contiguous(),
        position.reshape(batch, 1).to(torch.int32).contiguous(), pool, rows,
        ROWS_SOURCE_DIGEST)
    return pooled, rings.reshape(batch, TAIL_HALVES, depth, int(tail_bank.shape[3]))


def dsa_decode_ring_rows_torch_oracle(tail_bank, slots, key, score, ape, position):
    """The rows step in torch, from the token stream: pool members come from this step's
    tokens where they fall at or after ``position`` and from the ring (row ``p % depth``)
    before it; no stepped ring is consulted. Rows that close no pool hold the compression
    of the same members anyway, so every row is defined."""
    batch, rows, pool, depth = _require_rows(tail_bank, slots, key, score, ape, position)
    rings = tail_bank[slots.to(torch.int64)].clone()
    dtype = rings.dtype
    pooled = torch.empty(batch * rows, int(rings.shape[3]), dtype=dtype)
    for b in range(batch):
        start = int(position[b])

        def member(pos, half):
            if pos >= start:
                src = key if half == 0 else score
                return src[b * rows + pos - start].to(dtype)
            return rings[b, half, pos % depth]

        for t in range(rows):
            pos = start + t
            members = range(pos - pool + 1, pos + 1)
            pool_key = torch.stack([member(p, 0) for p in members])
            pool_score = torch.stack([member(p, 1) for p in members])
            pooled[b * rows + t] = _compress_pool_torch(pool_key, pool_score, ape.float())[0]
        for t in range(rows):
            at = (start + t) % depth
            rings[b, 0, at] = key[b * rows + t].to(dtype)
            rings[b, 1, at] = score[b * rows + t].to(dtype)
    return pooled, rings
