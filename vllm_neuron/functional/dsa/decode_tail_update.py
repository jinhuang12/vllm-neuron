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

import logging
from dataclasses import dataclass

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
