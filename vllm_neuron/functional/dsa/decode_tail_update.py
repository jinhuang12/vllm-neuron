# SPDX-License-Identifier: Apache-2.0
"""Decode-step tail update for the sparse-attention indexer's k-pool.

``inc-glm53f-049``. WP4 pools ``index_kpool`` consecutive tokens into one indexer key. Prefill
sees whole pools and is served by :mod:`vllm_neuron.functional.dsa.kpool_hadamard` (``-047``).
Decode sees ONE token at a time, so a pool is completed across several steps: the tokens that do
not complete a pool are kept RAW in a small per-request ring, and the step that completes a pool
reads that ring plus the current token and compresses them.

This module owns that ring and that step. ``-047``'s own docstring names it: *"the raw tail cache
for a request's incomplete trailing pool (``kpool_compress.py:411``). ``inc-glm53f-049`` owns it."*

It is **kernel-class** under P13, and it is **SCRATCH**. The state is a device tensor that lives
between decode steps, and a decode loop runs one step per generated token; a torch-level step would
move that state off device and back on for every token. The substrate provides no k-pool tail step.
The torch code below is the CPU oracle and the batch reference -- one of the two roles the plan's
substrate register admits (``design/increment-plan.md`` section 4) -- and never the shipped
implementation.

THE PRECEDENT FOR THE SHAPE OF THIS MODULE IS LANDED, not invented here:
:mod:`vllm_neuron.functional.kda.decode_state` (``inc-glm53f-036``) is the same problem one layer
down -- take the state a prefill left, advance it by exactly one token, return the advanced state
and that token's output. Its rule about the one-token shape is this module's rule too: *"THE
ONE-TOKEN SHAPE IS NOT AN INEFFICIENCY TO BE BATCHED AWAY. It is the contract: the route predicate
reads the dispatch count as an equality against the caller's step count."*

What the kernel computes
------------------------
The semantics are TRANSCRIBED from the upstream reference, not invented here. The reference is
``kpool_decode_update_and_maybe_write_cache_batched`` (``:612``) and its kernel
``_kpool_decode_update_batched_kernel`` (``:440-609``) in
``vllm/models/glm5next/nvidia/ops/kpool_compress.py`` at ``vllm-project/vllm`` head ``878631b6``
(891 L, sha256 ``6e19b31c50de43f7dfa9c7d9b7a3d7b856939f49bd4a3c031d2b50d28fe04f68`` -- the same
bytes ``increments/probe-047-upstream-semantics.md`` records, re-fetched and re-digested for this
increment in ``increments/probe-049-upstream-semantics.md``).

Per decode token, with ``slot = position % pool_size``:

1. **The completion read runs FIRST.** If ``slot == pool_size - 1`` this token completes a pool.
   The pool's ``pool_size`` members are the ring's prior stashes for every slot except this one,
   and the CURRENT token for this one (``:526`` ``is_current``, ``:533`` and ``:564`` select on it).
   They are compressed exactly as ``-047`` compresses a prefill pool::

       w[s, d] = softmax_over_s( score[s, d] + ape[s, d] )
       pooled[d] = sum_s w[s, d] * key[s, d]
       out[d] = FWHT_128( pooled )[d] * (1 / sqrt(128))

2. **The stash runs SECOND**, for EVERY token and not only for completions. The current key goes to
   ring row ``slot`` of the key half and the current score to row ``slot`` of the score half.

THE ORDER IS LOAD-BEARING AND UPSTREAM SAYS SO IN ITS OWN COMMENT (``:595-598``): *"Stash the
current token AFTER any completion read so the completion uses prior stashes (and the current
token's own key/score via is_current), then leaves this token for future pools."* Stashing first
would overwrite the ring row this pool still needs when ``pool_size == 1``, and -- more to the
point -- it would make the completion read the ring for a value it already has as an argument.

THE STASH IS NOT GATED ON COMPLETION, and upstream records a MEASURED defect from getting that
wrong (``:502-507``): gating the stash on the pool-granular validity dropped every intra-pool
token, so *"a decode-built pool compressed 3 stale ring entries (the prefill-seeded prompt tail,
frozen forever) plus the current token"*. Every real token stashes.

The two bf16 round trips are the reference's, not this module's taste
--------------------------------------------------------------------
Upstream rounds to bf16 TWICE inside the completion: once on the pooled vector before the rotation
and once on the rotated vector after it (``:567-568`` -- ``x = (acc / denom).to(tl.bfloat16)`` then
``x = _hadamard128(x).to(tl.bfloat16)``). ``-047``'s prefill kernel does NOT do the first one: it
carries fp32 into the butterfly and casts once at the end. So this module's output is not
bit-identical to ``-047``'s on the same pool, and that is FAITHFUL rather than a defect. Both round
trips are reproduced, and the module's own torch reference reproduces them too, so the acceptance
compares like with like.

Why the ring index equals the pool slot
---------------------------------------
Upstream computes the ring row as ``phys = (pool_logical_start + pool_slot) % POOL_SIZE`` with
``pool_logical_start = position - slot`` (``:522``, ``:527``). Since ``slot == position %
pool_size``, ``position - slot`` is an exact multiple of ``pool_size``, so ``phys == pool_slot`` for
every member. The modulo is redundant given the reference's own definitions, and this module writes
``pool_slot`` directly. The test asserts the identity over the whole declared position range rather
than trusting this paragraph.

What this module deliberately does NOT do
-----------------------------------------
Each of these is owned elsewhere, and leaving it out is a recorded decision rather than an omission:

* **fp8 quantisation and the ue8m0 scale** (``:570-577``). This module returns bf16 and no scale.
  ``inc-glm53f-053``'s adapter owns that half, exactly as it does for ``-047``.
* **the indexer cache write at ``loc``** (``:579-593``). Nothing here touches a KV cache.
* **paging of the ring.** Upstream addresses the ring through ``tail_slot_mapping`` and a block
  base (``:498-500``). This module takes the ring for ONE request as a plain tensor; resolving a
  request to its block is the indexer integration's job (``inc-glm53f-051``).
* **batching several requests, or several verify tokens, per call.** Upstream's kernel is one
  program per request over ``next_n`` tokens in position order (``:467-477``). Here one call is one
  token, for the route-predicate reason ``-036`` records above. A batched call would also need a
  per-request slot, which is a data-dependent address, and ``slot`` here is a trace-time constant.

Why ``position`` is a python int
--------------------------------
``slot`` selects which ring row is read as the current token and which row is written, so it is an
ADDRESS. Passing it as a tensor would force a data-dependent trace, the same reason ``-047`` gives
for its ``n_pools`` and ``pool_size``. The compiled graph specialises on
``(pool_size, slot, head_dim, dtype)``, so a steady-state decode loop compiles at most ``pool_size``
traces -- four on the target checkpoint -- and then reuses them forever.
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
"""Ring and token dtypes that take the NKI route. Upstream asserts bf16 (``:651``, ``:658-659``)."""

TAIL_HALVES = 2
"""The ring holds two halves: keys at half 0 and gate scores at half 1 (``:637``, ``:606``)."""


class DecodeTailUpdateError(ValueError):
    """A malformed call: wrong rank, mismatched shapes, a bad pool size, or a negative position."""


# ---------------------------------------------------------------------------------------------
# Why the butterfly is IMPORTED rather than transcribed
# ---------------------------------------------------------------------------------------------
# ``_fwht128_inplace`` is private to ``kpool_hadamard`` and is imported anyway, deliberately. That
# module's own docstring states the reason for its shape: "a separate transcription here would let
# the two drift and would make the identity reading evidence about nothing". The same argument
# applies across modules -- the prefill path and the decode path must rotate with the SAME code, or
# a divergence between them would be invisible to both of their tests. ``HADAMARD_SCALE`` and
# ``INDEX_HEAD_DIM`` come along for the same reason: one definition, two callers.


@dataclass
class _DecodeTailDispatchCounters:
    """Route-predicate counters for this module, form R-1 (``design/increment-plan.md`` D13).

    ONE seam and therefore one ``nki_dispatch`` counter. The declared reading is an EQUALITY
    against the caller's step count: ``k`` steps must show ``k`` dispatches, per case, with the
    counters reset at the start of each case (section 4b's convention).
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _DecodeTailDispatchCounters()


def reset_decode_tail_dispatch_counters() -> None:
    """Zero the counters. Called at the START of each declared case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def decode_tail_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def decode_tail_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam LAST dispatched, or ``None``.

    Derived THROUGH the seam rather than from this module's import list, so it certifies what ran
    instead of what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    separates "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    MEASURED, not assumed, for the reason ``ragged_pack.py:170-181`` records at its own equivalent:
    ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose ``__module__`` is
    ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``, so reading those attributes
    off the decorated object would record the DECORATOR's identity.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


def _row_pattern(rows: int, head_dim: int) -> list[list[int]]:
    """The access pattern for ``rows`` CONTIGUOUS rows of a 2-D buffer of width ``head_dim``.

    One spelling, used by every read and every write in this kernel, so a stride mistake is one
    mistake in one place instead of six chances to make it. Same shape as ``-047``'s reads and
    stores, with the row stride equal to the row width because nothing here is strided over pools.
    """
    return [[head_dim, rows], [1, head_dim]]


def _load_rows(hbm, rows: int, head_dim: int, first_row: int):
    """``rows`` contiguous rows of a 2-D HBM buffer as one tile IN THE SOURCE DTYPE.

    Used for the ring rows this step CARRIES UNCHANGED. Carrying them in their own dtype makes the
    carry an exact byte copy, with no widening and no rounding to reason about -- a bf16 row that
    went out through fp32 and back would be exact anyway, but "exact because bf16 to fp32 to bf16
    round-trips" is an argument, and "exact because nothing converted it" is not.
    """
    out = nl.ndarray((rows, head_dim), dtype=hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out, src=hbm.ap(pattern=_row_pattern(rows, head_dim), offset=first_row * head_dim))
    return out


def _load_rows_fp32(hbm, rows: int, head_dim: int, first_row: int):
    """``rows`` contiguous rows of a 2-D HBM buffer as one fp32 tile, starting at ``first_row``.

    Used for every row that enters the ARITHMETIC. The DMA lands in a tile of the source dtype and
    a separate ``tensor_copy`` widens it. The staging is UNCONDITIONAL rather than guarded on
    ``hbm.dtype``, for the reason ``kpool_hadamard.py:258-264`` records: a dtype comparison between
    a NKI tensor's dtype and a ``nki.language`` dtype object is a trace-time equality this file
    would have to be right about, and an always-stage form is correct for every input dtype at the
    cost of one copy.

    ONE ROW PER READ, ALWAYS, AND NEVER A SLICE OF A WIDER TILE. Every row this kernel needs is
    read at its own offset. That is not a stylistic choice: slicing a PARTITION RANGE out of an
    SBUF tile has no precedent in this tree -- the landed sliced operands slice the FREE axis
    (``mla_sparse.py:350`` ``p_t[:, ck, :]``, ``permute_routed_tokens.py:591``
    ``deduped_free[:, 0:1]``), and the one landed partition-range operand is a ``dma_copy``
    DESTINATION in HBM (``permute_routed_tokens.py:507``). Reading each row separately uses only
    the access-pattern form ``-047`` already ships.

    NO BROADCAST IS NEEDED ANYWHERE IN THIS MODULE, which is the one simplification the one-token
    shape buys over ``-047``: that kernel replicates an ``ape`` row across up to 128 pool partitions
    with a zero partition stride, and here every tile has exactly one row, so an ``ape`` row is just
    a one-row read at its own offset.
    """
    out = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=_load_rows(hbm, rows, head_dim, first_row))
    return out


def _cast_row(src, head_dim: int, dtype):
    """One ``(1, head_dim)`` tile, cast to ``dtype``. A whole-tile ``tensor_copy``, as ``-047``."""
    out = nl.ndarray((1, head_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=src)
    return out


# ---------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------


@nki.jit
def _decode_tail_update_nki(tail_hbm, key_hbm, score_hbm, ape_hbm, pool_size, slot):
    """Advance the k-pool tail ring by exactly ONE token, and compress a pool if this one ends it.

    Args:
        tail_hbm: ``[2 * pool_size, head_dim]`` -- the ring, flattened. Rows ``0 .. pool_size-1``
            are the key half and rows ``pool_size .. 2*pool_size-1`` the gate-score half, which is
            upstream's own layout: it addresses the score half at ``block_base + KPOOL_HEAD +
            phys * HEAD_DIM`` where ``KPOOL_HEAD == pool_size * head_dim`` (``:606``, ``:696``).
        key_hbm: ``[1, head_dim]`` -- this token's indexer key.
        score_hbm: ``[1, head_dim]`` -- this token's gate score.
        ape_hbm: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias, applied INSIDE the
            softmax (``:534-538``).
        pool_size: tokens per pool. A compile-time constant.
        slot: ``position % pool_size`` for this token. A compile-time constant, because it is an
            address.

    Returns:
        ``(pooled, tail_out)``. ``pooled`` is ``[1, head_dim]`` in the ring's dtype: the completed
        pool's compressed key when ``slot == pool_size - 1``, and ZERO otherwise -- the host seam
        turns that into ``None`` rather than making a reader interpret zeros. ``tail_out`` is
        ``[2 * pool_size, head_dim]``, the ring with this token stashed at row ``slot`` of each half.

    Returning the new ring instead of mutating the old one is the landed contract for a decode state
    step (``kda/decode_state.py:227-232`` stores into ``o_hbm`` and ``state_out_hbm``), and it is
    also what the NKI boundary supports: an input HBM tensor is not an output.

    THE COST OF THAT, SAID OUT LOUD: the step copies the ``2 * pool_size - 2`` rows it does not
    change, so a decode token moves the whole ring rather than two rows of it. On the target
    checkpoint that is 8 rows of 128 bf16 elements -- 2 KiB in and 2 KiB out per token -- which is
    small beside the attention step it sits inside. Upstream avoids the copy by writing the ring in
    place through a paged mapping (``:600-607``); adopting that here needs the paged addressing this
    increment excludes, so the copy is the price of the narrower scope and ``inc-glm53f-051`` is
    where it can be revisited.

    THE COMPLETION BRANCH IS RESOLVED AT TRACE TIME, so a non-completing step traces no softmax at
    all and a completing step traces no ``where``. Upstream needs a data-dependent ``tl.where`` per
    member because its ``slot`` arrives in a tensor; here it is a python int, so "is this member the
    current token" is answered while the graph is built.
    """
    head_dim = tail_hbm.shape[1]
    n_rows = TAIL_HALVES * pool_size
    pooled_hbm = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    tail_out_hbm = nl.ndarray((n_rows, head_dim), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)

    key_sb = _load_rows_fp32(key_hbm, 1, head_dim, 0)
    score_sb = _load_rows_fp32(score_hbm, 1, head_dim, 0)

    # ---- 1. THE COMPLETION READ, FIRST (upstream's order, ``:595-598``). -------------------- #
    if slot == pool_size - 1:
        # Pass 1: the per-channel max of ``score + ape`` over the pool's members, for softmax
        # stability. The sums are KEPT rather than recomputed in pass 2 -- upstream recomputes
        # them (``:543-556``) because a Triton program holds one pool and re-reading is free
        # there; here they are already in SBUF.
        totals = []
        running_max = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        for member in range(pool_size):
            # ``phys == member``: see the module docstring's derivation. The current token comes
            # from the ARGUMENT, every other member from the ring's score half.
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

        # Reciprocal-then-multiply, not a divide: the form the ISA exposes directly (as ``-047``).
        inv = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=denom)
        pooled = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)

        # UPSTREAM'S FIRST bf16 ROUND TRIP (``:567``), before the rotation. Reproduced, not
        # skipped: it is a real quantisation of the pooled vector and dropping it would make this
        # path disagree with the reference by more than the rotation's own rounding.
        pooled_bf = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pooled_bf, src=pooled)
        pooled_rt = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pooled_rt, src=pooled_bf)

        scratch = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        rotated = _fwht128_inplace(pooled_rt, scratch, head_dim)
        scaled = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
        # UPSTREAM'S SECOND bf16 ROUND TRIP (``:568``). Here it is also the output cast, so one
        # copy serves both purposes.
        result = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=scaled)
        nl.store(pooled_hbm, value=result)
    else:
        # No pool ends here. A zero row, so the returned tensor is defined for every trace and the
        # host decides what it means.
        empty = nl.ndarray((1, head_dim), dtype=tail_hbm.dtype, buffer=nl.sbuf)
        nisa.memset(dst=empty, value=0.0)
        nl.store(pooled_hbm, value=empty)

    # ---- 2. THE STASH, SECOND, and for THIS token whether or not a pool ended. -------------- #
    # EVERY OUTPUT ROW IS WRITTEN EXACTLY ONCE, and each write is a whole-tile store through the
    # access-pattern form ``-047`` ships. Two rows come from this token and the rest are carried
    # across in their own dtype. Writing the prior ring wholesale and then overwriting two of its
    # rows would put two writes on the same addresses inside one kernel, and the ordering of those
    # two is not something this file should have to assume.
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
    """Record WHICH kernel the seam dispatched, and log it, OFF the compiled graph.

    ARGUMENT DISCIPLINE, WHICH IS THE WHOLE POINT: a folded helper takes ints, strings and dtypes
    ONLY, never an object. Dynamo runs a folded call at trace time and converts every non-tensor
    argument into a python constant first; an ``@nki.jit`` kernel is a FROZEN DATACLASS and Dynamo
    refuses to reconstruct one. So the kernel is read as a module global, the same object the call
    site hands to ``wrap_nki`` on the next line. The template is landed:
    ``kpool_hadamard.py:428-457``, itself following ``ragged_pack.py:471-517``.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_decode_tail_update_nki)
    logger.info(
        "[dsa-decode-tail-update] kernel=nki entry=%s pool_size=%d slot=%d head_dim=%d",
        entry,
        pool_size,
        slot,
        head_dim,
    )


def slot_of(position: int, pool_size: int) -> int:
    """The ring row this position writes: ``position % pool_size``.

    Public because the caller needs the same arithmetic to decide when a pool ends, and two
    spellings of one rule is how they drift apart.
    """
    if position < 0:
        raise DecodeTailUpdateError(f"position must not be negative; got {position}")
    if pool_size <= 0:
        raise DecodeTailUpdateError(f"pool_size must be positive; got {pool_size}")
    return position % pool_size


def completes_pool(position: int, pool_size: int) -> bool:
    """Whether the token at ``position`` ends a pool. Upstream's ``slot == POOL_SIZE - 1``
    (``:521``), with its ``pos_valid`` left to the caller: a negative position raises here rather
    than being silently treated as a padded entry."""
    return slot_of(position, pool_size) == pool_size - 1


def _validate(tail: Tensor, key: Tensor, score: Tensor, ape: Tensor) -> tuple[int, int]:
    """Host-side shape and dtype validation. Returns ``(pool_size, head_dim)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor VALUE, so nothing here forces a
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
    """THE COUNTED SEAM. Advance the tail ring by one decode token.

    Args:
        tail: ``[2, pool_size, head_dim]`` bf16 -- the ring. Half 0 holds keys, half 1 holds gate
            scores, which is upstream's layout (``:637``).
        key: ``[1, head_dim]`` bf16 -- this token's indexer key.
        score: ``[1, head_dim]`` bf16 -- this token's gate score.
        ape: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        position: this token's absolute position in the request. A python int; see the module
            docstring for why it is not a tensor.

    Returns:
        ``(pooled, new_tail)``. ``pooled`` is ``[1, head_dim]`` when this token completed a pool
        and ``None`` when it did not; ``new_tail`` has the same shape as ``tail``. The caller
        threads ``new_tail`` into the next step -- the ring is state, and this function does not
        mutate its argument.

    Raises:
        DecodeTailUpdateError: for a malformed call or a negative position.
    """
    pool_size, head_dim = _validate(tail, key, score, ape)
    slot = slot_of(position, pool_size)
    ends_pool = slot == pool_size - 1

    if not can_run_dsa_decode_tail_update(tail, key, score, ape):
        return _dsa_decode_tail_update_torch(tail, key, score, ape, position)

    flat_tail = tail.reshape(TAIL_HALVES * pool_size, head_dim).contiguous()

    _COUNTERS.nki_dispatch += 1
    # The log and the identity read are FOLDED off the traced graph; the counter increment stays,
    # because a plain int attribute store is a recorded side effect and not a host call.
    _record_nki_dispatch("step", pool_size, slot, head_dim)
    pooled, new_flat = wrap_nki(_decode_tail_update_nki)(
        flat_tail, key.contiguous(), score.contiguous(), ape.contiguous(), pool_size, slot
    )
    new_tail = new_flat.reshape(TAIL_HALVES, pool_size, head_dim)
    return (pooled if ends_pool else None), new_tail


# ---------------------------------------------------------------------------------------------
# Torch: the fallback, and the batch reference. TWO functions, and the split is deliberate.
# ---------------------------------------------------------------------------------------------
# The route predicate requires ``torch_fallback == 0`` on every declared case, AND those same cases
# compare against a torch recompute. One function serving both roles would make the predicate
# unreadable: the reference would raise the fallback counter on the very cases that must show zero.
# So the fallback counts and the reference does not, and the reference's docstring says so.


def _compress_pool_torch(pool_key: Tensor, pool_score: Tensor, ape: Tensor) -> Tensor:
    """One complete pool, compressed exactly as upstream's decode path does. THE ORACLE.

    ``pool_key`` and ``pool_score`` are ``[pool_size, head_dim]`` and the result is
    ``[1, head_dim]`` in ``pool_key``'s dtype.

    BOTH bf16 ROUND TRIPS ARE HERE (``:567-568``), and they are the reason this is not simply
    ``-047``'s oracle with a different input: that one keeps fp32 all the way to the output cast.
    ``dim=0`` is the SLOT axis, which is what makes the softmax per ``(slot, channel)``; a ``dim=-1``
    here would be a whole-vector softmax, which is a different kernel, and the test carries a case
    whose only job is to tell the two apart.
    """
    dtype = pool_key.dtype
    weights = torch.softmax(pool_score.float() + ape.float(), dim=0)
    pooled = (weights * pool_key.float()).sum(dim=0, keepdim=True)
    pooled = pooled.to(dtype).float()
    rotated = pooled @ hadamard_matrix(int(pool_key.shape[1])).t()
    return (rotated * HADAMARD_SCALE).to(dtype)


def _dsa_decode_tail_update_torch(
    tail: Tensor, key: Tensor, score: Tensor, ape: Tensor, position: int
) -> tuple[Tensor | None, Tensor]:
    """The single-step fallback, in torch. Counted, because it is a route the seam can take."""
    _COUNTERS.torch_fallback += 1
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
    """THE FULL RECOMPUTE. What ``k`` stepped updates must equal, computed WITHOUT stepping.

    Args:
        tail0: ``[2, pool_size, head_dim]`` -- the ring before the first step, as a prefill would
            have seeded it.
        keys: ``[k, head_dim]`` -- the ``k`` decode tokens' keys, in position order.
        scores: ``[k, head_dim]`` -- their gate scores.
        ape: ``[pool_size, head_dim]``.
        start_position: the absolute position of ``keys[0]``.

    Returns:
        ``(pooled_list, tail_k)`` -- one compressed key per pool that ended during the ``k`` steps,
        in order, and the ring as it stands after all ``k``.

    IT DOES NOT CALL THE SEAM AND IT DOES NOT TOUCH THE COUNTERS. That is what lets a case assert
    "exactly ``k`` dispatches" and "zero fallbacks" while comparing against this: every dispatch a
    case counts came from the stepped path.

    IT IS NOT A RE-IMPLEMENTATION OF THE STEP EITHER. It works from the token stream: the members
    of a pool that ends at position ``p`` are the tokens at ``p - pool_size + 1 .. p``, taken from
    ``keys``/``scores`` when they fall inside the decode window and from ``tail0`` when they were
    seeded before it. A stepped ring is never consulted, so an error in the ring's addressing
    cannot hide by appearing on both sides.
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
