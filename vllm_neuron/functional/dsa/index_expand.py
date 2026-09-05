# SPDX-License-Identifier: Apache-2.0
"""Expand the DSA indexer's selected POOL ids into the TOKEN indices the gather then reads.

WHAT THIS COMPUTES, in one line. The indexer selects pools, not tokens, so every selected pool id has
to become the ``pool_size`` token indices it covers, and the tokens after the last complete pool -- the
TAIL -- have to be appended because no pool covers them yet::

    history columns  ``g * pool_size + o``  ->  ``pool_ids[row, g] * pool_size + o``
    tail columns     ``topk + t``           ->  ``tail_start + t``  while ``t < seq_len % pool_size``
    everything else                         ->  ``-1``

with ``topk = n_groups * pool_size`` and ``tail_start`` the start of the incomplete final pool. The
output is ``[rows, topk + pool_size - 1]`` int32, which is upstream's own column order.

**A ``-1`` IS A VALUE, NOT AN OUT-OF-BOUNDS INDEX.** It is the sentinel meaning "this column selects no
token", and it appears for two independent reasons: the selector handed us a ``-1`` pool id (fewer pools
were selected than the width allows), or a tail column sits past the row's tail count. Every consumer
must mask on it. It is excluded BY NAME from the in-range population the test counts, because counting
it as an index would turn a correct sentinel into a false out-of-bounds reading.

THE CALLER PRECONDITION, which is upstream's and not an addition here::

    every non-negative pool id satisfies  0 <= pool_ids[row, g] < seq_len[row] // pool_size

Upstream's kernel gates ONLY on ``pool_ids >= 0`` (``kpool_compress.py:846-848``) and never compares an
expanded index against ``seq_len``, so a pool id past the row's last pool expands to token indices past
the end of that row's sequence, and the torch reference below reproduces that faithfully rather than
clamping it. The measured evidence is in ``probe-048-mechanism-host.out``
(``A6_NONSENTINEL_OUT_OF_BOUNDS=8`` on a fixture that violated the precondition, with the kernel still
bit-identical to the reference). The test states the precondition, checks it on every fixture, and
carries one labelled supplementary case that feeds an out-of-range pool id on purpose and asserts the
kernel and the reference agree on the out-of-range expansion.

**``pool_size`` MUST BE A POWER OF TWO, AND THE GATE ENFORCES IT.** The one piece of upstream's
arithmetic with no closed form in max and min is ``tail_start = seq_len // pool_size * pool_size``. This
kernel computes it as ``seq_len - (seq_len & (pool_size - 1))``, which is exact for a power-of-two
``pool_size`` and silently wrong for any other, so ``can_run_dsa_index_expand`` refuses a non-power-of-two
and the torch reference serves that call instead. The checkpoint's compress ratio is 4.

THE TWO CLOSED FORMS THAT REPLACE UPSTREAM'S COMPARES AND SELECTS. Both were measured exact against
upstream's own ``where`` form before a line of this file was written
(``probe-048-mechanism-r3-host.out``, ``probe-048-mechanism-r4-host.out``):

  * HISTORY. Upstream writes ``pid >= 0 ? pid * pool_size + o : -1``; this writes
    ``max(pid * pool_size + o, -1)``. For any ``pid <= -1`` the largest ``pid * pool_size + o`` can be
    is ``-pool_size + (pool_size - 1) = -1``, so the max pins every negative pool id to exactly ``-1``;
    for ``pid >= 0`` the value is ``>= 0 > -1``. No compare op and no select op.
  * TAIL. With ``d = seq_len - min(tail_start + t, seq_len)`` and ``mask = min(max(d, 0), 1)``, the
    value is ``mask * (tail_start + t + 1) - 1``: mask 1 gives ``tail_start + t``, mask 0 gives ``-1``.
    ``t`` runs over ``[0, pool_size - 1)``, which covers every possible tail count because
    ``seq_len % pool_size <= pool_size - 1``.

WHY THE HISTORY LOOP RUNS OVER OFFSETS AND NOT OVER COLUMNS. For a fixed offset ``o``, the values of
every group are one ``tensor_scalar`` over a full-width ``(rows, n_groups)`` tile, and they land in
output columns ``o``, ``o + pool_size``, ``o + 2 * pool_size``, ... -- a STRIDED column write. That is
``pool_size`` iterations regardless of how many pools were selected. Walking columns instead would be
``n_groups * pool_size`` iterations on tiles one element wide: 2048 columns and about 4096 instructions
at the production width, which round 4 measured at 8.0x the emit cost of this form for identical values.
The strided destination slice is measured, not assumed -- see the refusal census below.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.
  * No gather. Turning these indices into keys is the landed ``dsa_paged_gather`` (the ``-044`` ledger
    row), which this feeds.
  * No selection. The pool ids arrive from the landed ``dsa_topk_select`` (the ``-043`` row), which is
    pool-granular on this checkpoint (``sparse_attn_indexer_kpool.py:551-554``).
  * No clamping of an index to the row's sequence, and no masking of the sentinel. Both belong to the
    consumer, and inventing either here would silently diverge from upstream.
  * No slot arithmetic. ``decode_tail_update.slot_of`` already owns "which ring row does this position
    write", and the two are related by the identity ``seq_len % pool_size == slot_of(seq_len,
    pool_size)`` -- so this module's ``tail_count`` and that helper are one rule, stated here for the
    anti-drift reason that module's own docstring gives. It is NOT imported: the identity is worth a
    sentence, not a dependency.

MEASURED REFUSALS AND MEASURED WRONG ANSWERS THIS FILE IS AUTHORED AROUND, each read off a compiler or a
simulator rather than guessed. Transcripts: ``probe-048-mechanism-host.out`` (round 1),
``probe-048-mechanism-r2-host.out`` (round 2), ``probe-048-mechanism-r3-host.out`` (round 3),
``probe-048-mechanism-r4-host.out`` (round 4).

  * ``nl.mod`` COMPUTES CORRECTLY ON THE SIMULATOR AND DOES NOT COMPILE. Round 1 read the whole
    expansion bit-identical through an in-kernel ``nl.mod`` and the capture leg then refused it:
    ``invalid kwarg 'op0': unsupported operator 'mod'``, 0 graphs, verifier ON. Every round after that
    runs the capture leg BEFORE the values leg.
  * ``nl.divide`` COMPILES AND IS SILENTLY WRONG ON int32. It rounds to NEAREST, not toward the floor:
    on ``[-7, -1, 7, 9]`` with operand 4 it read ``[-2, 0, 2, 2]`` where floor division gives
    ``[-2, -1, 1, 2]``, and its expansion read max abs diff 11. A single sampled value of a rounding
    rule is not a reading of a rounding rule.
  * ``nl.right_shift`` COMPILES ALONE BUT NOT IN EVERY POSITION. Its own screen captured; composed as
    the SECOND op of a two-op ``tensor_scalar`` it refused with ``invalid kwarg 'op1': unsupported
    bi...``. That is why ``bitwise_and`` carries the tail arithmetic here.
  * ``nl.arange`` DOES NOT EXIST on this image (round 4 census: ``NL_HAS_ARANGE=False``, with
    ``mgrid``, ``nl.iota`` and ``nl.affine_select`` also absent), so the usual affine-index spelling of
    a strided write is unavailable and a step-sliced destination is the spelling that was screened.
  * THE STEP-SLICED SBUF DESTINATION ``acc[:, o:topk:pool_size]`` COMPILES. Round 4 screened four
    layouts at the production width and all four captured with ``hlo=1`` and the verifier ON, all four
    bit-identical over 3417 non-sentinel of 4102 entries; this one is chosen because it emits upstream's
    column order directly with no host reindex at all.
  * A ``@nki.jit`` KERNEL TAKES A TRACE-TIME int, and ``wrap_nki`` passes it through --
    ``decode_tail_update.py:544-546`` is the landed precedent. ``pool_size`` and its mask are handed
    over as two separate ints so the kernel body performs no arithmetic on an int argument in the one
    place a wrong value would corrupt quietly instead of raising.
  * A folded host helper may take ints, strings and dtypes ONLY; an ``@nki.jit`` object is a frozen
    dataclass Dynamo will not reconstruct (``kpool_hadamard.py:436-442``). ``_record_nki_dispatch``
    below takes ints and reads the kernel as a module global.

SUBSTRATE. Kernel-class, and every per-element operation is in NKI: multiply, add, maximum, minimum,
subtract and bitwise-and, with one strided copy per offset. Nothing is quantised, no MX primitive is
named or reached, and no NxDI import appears -- the prohibited package name is deliberately NOT spelled
out anywhere in this file, so P4's mechanical scan reads zero without needing a token-class
resolution for a docstring. The torch function at the bottom is the
REFERENCE the kernel is measured against and the server for a call the gate refuses -- it is not a
fallback for kernel-class work, and the counted ``torch_fallback`` reading is exactly 0 over the
declared cases.
"""

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

INDEX_KPOOL = 4
"""The target checkpoint's compress ratio -- how many tokens one pool covers. RECORDED, NOT A LIMIT.

``pool_size`` is an ordinary argument here, so any power of two runs. The value is read off the
checkpoint configuration and matches upstream's ``compress_ratio == index_kpool``
(``sparse_attn_indexer_kpool.py:551-554``).
"""

_SUPPORTED_DTYPES = (torch.int32,)
"""Index dtypes that take the NKI route. int32 is what the selector produces and what the gather reads;
int64 indices would double the SBUF traffic for a range no sequence length reaches."""


class IndexExpandError(ValueError):
    """A malformed call: wrong rank, a row-count mismatch, or a non-positive ``pool_size``."""


@dataclass
class _IndexExpandDispatchCounters:
    """Route-predicate counters for this module, form R-1 (``design/increment-plan.md`` D13).

    ``nki_dispatch`` counts dispatches THROUGH THIS MODULE'S SEAM, so the declared total over the
    declared case set is readable from one place. The declared set is the four cases the plan block
    names, one dispatch each, for a declared total of 4. The labelled supplementary case is counted in
    its own reset window and is excluded from that total.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _IndexExpandDispatchCounters()


def reset_index_expand_dispatch_counters() -> None:
    """Zero the counters. Called at the START of each declared case (section 4b's convention)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def index_expand_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def index_expand_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam LAST dispatched, or ``None``.

    Derived THROUGH the seam rather than from this module's import list, so it certifies what ran
    instead of what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    separates "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    Measured, not assumed, for the reason ``vllm_neuron/functional/dsa/ragged_pack.py:170-181``
    records: ``@nki.jit`` returns a kernel object whose ``__module__`` is ``"nki.framework.kernel"``
    and whose ``__qualname__`` is ``None``, so reading those attributes off the decorated object would
    record the DECORATOR's identity and certify nothing.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


def is_power_of_two(value: int) -> bool:
    """Whether ``value`` is a positive power of two.

    Public because the gate, the validator and the test all need the SAME rule, and two spellings of
    one rule is how they drift apart -- the reason ``decode_tail_update.slot_of`` gives for being
    public too.
    """
    return value > 0 and (value & (value - 1)) == 0


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _index_expand_nki(pool_ids_hbm, seq_lens_hbm, pool_size, pool_mask):
    """Selected pool ids expanded to token indices, with the incomplete final pool appended.

    Args:
        pool_ids_hbm: ``[rows, n_groups]`` int32 -- the selected pool ids, ``-1`` where no pool was
            selected.
        seq_lens_hbm: ``[rows, 1]`` int32 -- the per-row sequence length, ALREADY a column. A ``(1, N)``
            row is refused by the MLIR verifier for a ``tensor_scalar`` operand (the ``-045`` finding,
            recorded at ``score_gemm.py:84-88``), and every per-row value here is a scalar operand.
        pool_size: python int, a power of two -- how many tokens one pool covers.
        pool_mask: python int, ``pool_size - 1``. Handed over separately so this body performs no
            arithmetic on an int argument in the one place a wrong value would corrupt quietly.

    Returns:
        ``[rows, n_groups * pool_size + pool_size - 1]`` int32, in upstream's column order.

    THE HISTORY REGION IS WRITTEN ONE OFFSET AT A TIME, STRIDED. Each iteration computes every group's
    value for one offset on a full-width tile and copies it into the columns of stride ``pool_size``
    that own it, so the loop is ``pool_size`` long however many pools were selected.
    """
    rows = pool_ids_hbm.shape[0]
    n_groups = pool_ids_hbm.shape[1]
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1

    out = nl.ndarray((rows, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)

    pid = nl.ndarray((rows, n_groups), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pid, src=nl.load(pool_ids_hbm))
    seq = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq, src=nl.load(seq_lens_hbm))

    acc = nl.ndarray((rows, out_cols), dtype=nl.int32, buffer=nl.sbuf)

    # THE HISTORY REGION. `max(pid * pool_size + o, -1)` is upstream's `where(pid >= 0, ...)` with no
    # compare and no select: the largest value any negative pool id can reach is exactly -1.
    for o in range(pool_size):
        vals = nl.ndarray((rows, n_groups), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vals, data=pid,
                           op0=nl.multiply, operand0=pool_size, op1=nl.add, operand1=o)
        nisa.tensor_scalar(dst=vals, data=vals, op0=nl.maximum, operand0=-1)
        nisa.tensor_copy(dst=acc[:, o:topk:pool_size], src=vals)

    # `tail_start = seq_len - (seq_len & (pool_size - 1))`, exact for a power-of-two pool_size and the
    # reason the gate refuses any other. `nl.mod` would say this directly and does not compile.
    rem = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rem, data=seq, op0=nl.bitwise_and, operand0=pool_mask)
    tail_start = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tail_start, data1=seq, data2=rem, op=nl.subtract)

    # THE TAIL REGION, one column per possible tail token. `mask` is 1 exactly while
    # `tail_start + t < seq_len`, so the value is `tail_start + t` there and -1 elsewhere.
    for t in range(pool_size - 1):
        pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pos, data=tail_start, op0=nl.add, operand0=t)
        clipped = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=clipped, data1=pos, data2=seq, op=nl.minimum)
        room = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=room, data1=seq, data2=clipped, op=nl.subtract)
        mask = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mask, data=room, op0=nl.maximum, operand0=0, op1=nl.minimum, operand1=1)
        pos1 = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pos1, data=pos, op0=nl.add, operand0=1)
        prod = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=pos1, data2=mask, op=nl.multiply)
        col = topk + t
        nisa.tensor_scalar(dst=acc[:, col:col + 1], data=prod, op0=nl.subtract, operand0=1)

    nl.store(out, value=acc)
    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(rows: int, n_groups: int, pool_size: int, out_cols: int) -> None:
    """Record which kernel the seam dispatched, and log it, OFF the compiled graph.

    The template is landed and measured: ``kpool_hadamard.py:428-457`` by way of
    ``score_gemm.py:306-330``. A folded helper takes ints only -- Dynamo runs it at trace time and
    refuses to reconstruct an ``@nki.jit`` object -- so the kernel is read as a module global rather
    than passed in. D13.1 still holds: this body runs only when the dispatch branch runs, so the
    recorded identity is derived by TAKING the branch.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_index_expand_nki)
    logger.info(
        "[dsa-index-expand] kernel=nki rows=%d n_groups=%d pool_size=%d out_cols=%d",
        rows,
        n_groups,
        pool_size,
        out_cols,
    )


def _validate(pool_ids: Tensor, seq_lens: Tensor, pool_size: int) -> tuple[int, int]:
    """Host-side shape validation. Returns ``(rows, n_groups)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor VALUE, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace. The caller precondition on the pool ids
    THEMSELVES is not checked for that reason -- it is upstream's contract, stated in the module
    docstring and measured by the test, not enforced by a device round trip on every call.
    """
    if pool_ids.ndim != 2:
        raise IndexExpandError(
            f"pool_ids must be 2-D [rows, n_groups]; got shape {tuple(pool_ids.shape)}"
        )
    if seq_lens.ndim != 1:
        raise IndexExpandError(
            f"seq_lens must be 1-D [rows], which is upstream's shape; got "
            f"{tuple(seq_lens.shape)}"
        )
    rows, n_groups = (int(d) for d in pool_ids.shape)
    if int(seq_lens.shape[0]) != rows:
        raise IndexExpandError(
            f"seq_lens must carry one length per row of pool_ids ({rows}); got "
            f"{int(seq_lens.shape[0])}"
        )
    if rows <= 0 or n_groups <= 0:
        raise IndexExpandError(f"rows and n_groups must both be positive; got {(rows, n_groups)}")
    if pool_size <= 0:
        raise IndexExpandError(f"pool_size must be positive; got {pool_size}")
    return rows, n_groups


def can_run_dsa_index_expand(pool_ids: Tensor, seq_lens: Tensor, pool_size: int) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch reference.

    THE POWER-OF-TWO CHECK IS LOAD-BEARING, not defensive. The kernel derives ``tail_start`` as
    ``seq_len - (seq_len & (pool_size - 1))``, which is exact only for a power-of-two ``pool_size``
    and silently wrong -- not an error -- for any other, so refusing here is what keeps the wrong
    answer unreachable.
    """
    if not can_run_kernel():
        return False
    if not is_power_of_two(pool_size):
        return False
    if pool_ids.dtype not in _SUPPORTED_DTYPES or seq_lens.dtype not in _SUPPORTED_DTYPES:
        return False
    if pool_ids.ndim != 2 or seq_lens.ndim != 1:
        return False
    return int(seq_lens.shape[0]) == int(pool_ids.shape[0])


def dsa_index_expand(pool_ids: Tensor, seq_lens: Tensor, pool_size: int = INDEX_KPOOL) -> Tensor:
    """THE COUNTED SEAM. Selected pool ids expanded to token indices, with the tail appended.

    Args:
        pool_ids: ``[rows, n_groups]`` int32 -- the selected pool ids, ``-1`` where no pool was
            selected. Produced by the landed ``dsa_topk_select``, which is pool-granular on this
            checkpoint. Every non-negative id must satisfy the caller precondition in the module
            docstring.
        seq_lens: ``[rows]`` int32 -- the per-row sequence length, upstream's shape.
        pool_size: how many tokens one pool covers. Must be a power of two to take the NKI route;
            defaults to this checkpoint's compress ratio.

    Returns:
        ``[rows, n_groups * pool_size + pool_size - 1]`` int32 token indices in upstream's column
        order, where ``-1`` is the sentinel for "this column selects no token" and is a VALUE rather
        than an out-of-bounds index.

    Raises:
        IndexExpandError: for a malformed call -- a non-2-D ``pool_ids``, a ``seq_lens`` that is not
            1-D or does not match the row count, or a non-positive ``pool_size``.
    """
    rows, n_groups = _validate(pool_ids, seq_lens, pool_size)
    out_cols = n_groups * pool_size + pool_size - 1

    if not can_run_dsa_index_expand(pool_ids, seq_lens, pool_size):
        _COUNTERS.torch_fallback += 1
        return _dsa_index_expand_torch(pool_ids, seq_lens, pool_size)

    # The transport the device cannot pay for: every per-row value reaches a `tensor_scalar` as a
    # COLUMN operand, so the lengths are reshaped once here rather than per use on the device.
    seq_col = seq_lens.reshape(rows, 1).contiguous()

    _COUNTERS.nki_dispatch += 1
    # The log and the identity read are FOLDED off the traced graph; the counter increment stays,
    # because a plain int attribute store is a recorded side effect and not a host call.
    _record_nki_dispatch(rows, n_groups, pool_size, out_cols)
    return wrap_nki(_index_expand_nki)(
        pool_ids.contiguous(), seq_col, pool_size, pool_size - 1
    )


# ---------------------------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------------------------


def _dsa_index_expand_torch(pool_ids: Tensor, seq_lens: Tensor, pool_size: int) -> Tensor:
    """The reference the NKI route is measured against. NOT a fallback for kernel-class work (P13).

    This exists to be the oracle in the test and to serve a call the gate refuses, which is a
    malformed call, an unadmitted dtype or a non-power-of-two ``pool_size`` rather than kernel-class
    work taking a torch path. The counted ``torch_fallback`` reading is exactly 0 across the declared
    cases, and the test proves that zero can fire by handing the seam a non-power-of-two on purpose.

    Transcribed from ``kpool_compress.py:818-857`` and kept in UPSTREAM'S OWN ``where`` form rather
    than rewritten into the kernel's closed forms. That is the point: if the two spellings were the
    same spelling, agreeing with this would only prove the kernel agrees with itself.
    """
    rows, n_groups = (int(d) for d in pool_ids.shape)
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1

    cols = torch.arange(out_cols, dtype=torch.int64, device=pool_ids.device)
    cols = cols[None, :].expand(rows, out_cols)
    seq = seq_lens.to(torch.int64).reshape(rows, 1)

    tail_start = (seq // pool_size) * pool_size
    tail_count = seq - tail_start

    is_history = cols < topk
    group = torch.clamp(cols // pool_size, max=n_groups - 1)
    offset = cols % pool_size
    pid = torch.gather(pool_ids.to(torch.int64), 1, group)
    history = torch.where(pid >= 0, pid * pool_size + offset, torch.full_like(pid, -1))

    tail_offset = cols - topk
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail = torch.where(is_tail, tail_start + tail_offset, torch.full_like(cols, -1))

    return torch.where(is_history, history, tail).to(torch.int32)
