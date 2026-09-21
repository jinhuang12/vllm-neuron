# SPDX-License-Identifier: Apache-2.0
"""Expand selected pool ids into the token indices a gather can read.

The indexer selects pools rather than tokens, so every selected pool id becomes the
``pool_size`` token indices it covers, and the tokens after the last complete pool -- the
tail -- are appended because no pool covers them. A ``-1`` is the sentinel for "this column
selects no token": it is a value rather than an out-of-bounds index, and every consumer has
to mask on it.

The emitted width is the meaningful width rounded up to a whole number of ``KEY_CHUNK``
columns, padded with the same ``-1``. ``mla_sparse_attention`` refuses a selected-row count
that is not a positive multiple of ``KEY_CHUNK``, because that count rides the partition
axis in MM2 in chunks of exactly that size, and the meaningful width never is one. Ask
``index_expand_raw_width`` and ``index_expand_width`` for the two widths rather than
recomputing either.

``pool_size`` must be a power of two to take the NKI route: the kernel derives ``tail_start``
as ``seq_len - (seq_len & (pool_size - 1))``, which is exact for a power of two and silently
wrong for anything else. Callers must keep every non-negative pool id below
``seq_len[row] // pool_size``; nothing here clamps an expanded index against the sequence
length, and nothing masks the sentinel.
"""

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention.mla_sparse import KEY_CHUNK
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# KEY_CHUNK is imported rather than declared here: it is the sparse attention kernel's MM2
# chunk width and the hardware partition extent behind it, so the one definition stays with
# the constraint (``mla_sparse.py:111``). That module imports nothing from this package, so
# the import runs one way only.

INDEX_KPOOL = 4
"""The target checkpoint's compress ratio -- how many tokens one pool covers. Recorded, not a limit.

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
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _IndexExpandDispatchCounters()


def reset_index_expand_dispatch_counters() -> None:
    """Zero the counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def index_expand_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def index_expand_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator instead:
    ``@nki.jit`` returns a kernel object whose ``__module__`` is ``"nki.framework.kernel"``
    and whose ``__qualname__`` is ``None``.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


def is_power_of_two(value: int) -> bool:
    """Whether ``value`` is a positive power of two.

    Public because the gate, the validator and the tests all need the same rule.
    """
    return value > 0 and (value & (value - 1)) == 0


def index_expand_raw_width(n_groups: int, pool_size: int) -> int:
    """How many columns carry meaning: ``n_groups * pool_size + pool_size - 1``.

    The history region is ``n_groups * pool_size`` columns and the tail region is
    ``pool_size - 1``, one per token an incomplete final pool can hold. The emitted tensor is
    wider, so a consumer that needs to know where the meaningful columns stop asks this.
    """
    return n_groups * pool_size + pool_size - 1


PARTITION_MAX = 128
"""Query rows one SBUF tile can hold: the partition-axis bound, ``nl.tile_size.pmax``.

This bounds one row tile, not the call. The kernel walks the query-token axis in tiles of at
most this height, so a prefill with more selected rows than this is served rather than trapped
in the ``dma_copy`` assert. It is a module constant so the kernel and the tile arithmetic read
one number; ``dsa/causal_bound.py`` tiles its own row axis the same way.
"""


def _row_tiles_unchecked(rows: int) -> list[tuple[int, int]]:
    """The ``(start, height)`` query-row tiles, in order, with no refusal in the arithmetic.

    Only ``for`` over ``range``, ``append``, a tuple and an ``if``/``else``: the tracer refuses
    a list comprehension or a ``min`` where the kernel bodies reach, so the plain loop stands
    in for ``min(PARTITION_MAX, rows - start)``. One query row is one token here, so nothing
    can be split and the tile height is :data:`PARTITION_MAX` itself.
    """
    tiles = []
    for start in range(0, rows, PARTITION_MAX):
        remaining = rows - start
        if remaining < PARTITION_MAX:
            tiles.append((start, remaining))
        else:
            tiles.append((start, PARTITION_MAX))
    return tiles


def _row_tile_count_unchecked(rows: int) -> int:
    """How many tiles :func:`_row_tiles_unchecked` returns, by arithmetic.

    A kernel loop bound: the body counts with ``for idx in range(bound)`` because the compiler
    refuses a ``for`` whose loop variable is a tuple, and the bound is a plain name rather than
    a call.
    """
    return (rows + PARTITION_MAX - 1) // PARTITION_MAX


def row_tiles(rows: int) -> list[tuple[int, int]]:
    """The ``(start, height)`` query-row tiles the kernel walks, in order. The checked path.

    The kernel body calls :func:`_row_tiles_unchecked` instead, because this function raises
    and NKI refuses a traced ``raise``; the two return the same list for every admissible
    input.

    Raises:
        IndexExpandError: if ``rows`` is not positive. A call with no selected rows has no
            tiles.
    """
    if rows < 1:
        raise IndexExpandError(
            f"rows must be the positive number of query rows to tile; got rows={rows}"
        )
    return _row_tiles_unchecked(rows)


def row_tile_count(rows: int) -> int:
    """How many tiles :func:`row_tiles` returns. The checked path, and the same refusal."""
    if rows < 1:
        raise IndexExpandError(
            f"rows must be the positive number of query rows to tile; got rows={rows}"
        )
    return _row_tile_count_unchecked(rows)


def index_expand_width(n_groups: int, pool_size: int) -> int:
    """How many columns are emitted: the raw width rounded up to a whole ``KEY_CHUNK``.

    Derived rather than typed anywhere, because three places need the identical number: the
    kernel that allocates the output, the torch reference that must match it column for
    column, and the tests that assert the sparse kernel admits it.
    """
    raw = index_expand_raw_width(n_groups, pool_size)
    return ((raw + KEY_CHUNK - 1) // KEY_CHUNK) * KEY_CHUNK


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _index_expand_nki(pool_ids_hbm, seq_lens_hbm, pool_size, pool_mask):
    """Selected pool ids expanded to token indices, with the incomplete final pool appended.

    Args:
        pool_ids_hbm: ``[rows, n_groups]`` int32 -- the selected pool ids, ``-1`` where no
            pool was selected.
        seq_lens_hbm: ``[rows, 1]`` int32 -- the per-row sequence length, already a column.
            The MLIR verifier refuses a ``(1, N)`` row as a ``tensor_scalar`` operand, and
            every per-row value here is a scalar operand.
        pool_size: python int, a power of two -- how many tokens one pool covers.
        pool_mask: python int, ``pool_size - 1``. Handed over separately so this body performs
            no arithmetic on an int argument, where a wrong value would corrupt quietly
            instead of raising.

    Returns:
        ``[rows, index_expand_width(n_groups, pool_size)]`` int32, in upstream's column order,
        with every column past ``index_expand_raw_width(n_groups, pool_size)`` holding the
        ``-1`` sentinel.

    The three regions partition the output exactly, so nothing is written twice and nothing is
    left undefined::

        [0, topk)            history, the strided loop below
        [topk, raw_cols)     tail, one column per iteration, `pool_size - 1` of them
        [raw_cols, out_cols) padding, one memset, all `-1`

    The history region is written one offset at a time: each iteration computes every group's
    value for one offset on a full-width tile and copies it into the columns of stride
    ``pool_size`` that own it, so the loop is ``pool_size`` long however many pools were
    selected. Walking columns instead would be ``n_groups * pool_size`` iterations on tiles one
    element wide -- roughly 4096 instructions at the production width, measured at 8x the emit
    cost of this form.

    The query-row axis is walked in tiles of at most :data:`PARTITION_MAX` rows, each tile
    reading only its own rows of both inputs. Without that bound a taller call dies inside the
    vendor's assert at ``nki/isa/_copy.py:152``.
    """
    rows = pool_ids_hbm.shape[0]
    n_groups = pool_ids_hbm.shape[1]
    topk = n_groups * pool_size
    # Both widths are plain python ints resolved at trace time, so the traced graph sees two
    # constants rather than device arithmetic.
    raw_cols = index_expand_raw_width(n_groups, pool_size)
    out_cols = index_expand_width(n_groups, pool_size)

    out = nl.ndarray((rows, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)

    # The unchecked tile arithmetic, because the tracer follows these calls: the checked
    # `row_tiles` raises and NKI refuses a traced `raise`. The loop bound is a plain name and
    # the tile list is read by index, which are the forms the tracer accepts here.
    tiles = _row_tiles_unchecked(int(rows))
    tile_count = _row_tile_count_unchecked(int(rows))

    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]

        pid = nl.ndarray((height, n_groups), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=pid, src=nl.load(pool_ids_hbm[start:start + height, 0:n_groups])
        )
        # Re-loaded per tile: `seq_lens` is per-row, so hoisting this column out of the walk
        # would give every tile the first tile's lengths.
        seq = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=seq, src=nl.load(seq_lens_hbm[start:start + height, 0:1]))

        acc = nl.ndarray((height, out_cols), dtype=nl.int32, buffer=nl.sbuf)

        # The padding region first, so the tile's whole extent is accounted for before any
        # real value is written. The guard matters: an already admissible raw width makes this
        # an empty slice, which is exactly the `pool_size == 1` case.
        if out_cols > raw_cols:
            nisa.memset(acc[:, raw_cols:out_cols], -1)

        # `max(pid * pool_size + o, -1)` replaces a compare and a select: the largest value
        # any negative pool id can reach is exactly -1. This loop walks columns and is
        # `pool_size` long; it is not the row tiling.
        for o in range(pool_size):
            vals = nl.ndarray((height, n_groups), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=vals, data=pid,
                               op0=nl.multiply, operand0=pool_size, op1=nl.add, operand1=o)
            nisa.tensor_scalar(dst=vals, data=vals, op0=nl.maximum, operand0=-1)
            nisa.tensor_copy(dst=acc[:, o:topk:pool_size], src=vals)

        # `tail_start = seq_len - (seq_len & (pool_size - 1))`, exact for a power-of-two
        # pool_size and the reason the gate refuses any other. `nl.mod` would express it
        # directly but does not compile.
        rem = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rem, data=seq, op0=nl.bitwise_and, operand0=pool_mask)
        tail_start = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tail_start, data1=seq, data2=rem, op=nl.subtract)

        # The tail region, one column per possible tail token. `mask` is 1 exactly while
        # `tail_start + t < seq_len`, so the value is `tail_start + t` there and -1 elsewhere.
        # This loop walks columns too.
        for t in range(pool_size - 1):
            pos = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos, data=tail_start, op0=nl.add, operand0=t)
            clipped = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=clipped, data1=pos, data2=seq, op=nl.minimum)
            room = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=room, data1=seq, data2=clipped, op=nl.subtract)
            mask = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=room, op0=nl.maximum, operand0=0,
                               op1=nl.minimum, operand1=1)
            pos1 = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos1, data=pos, op0=nl.add, operand0=1)
            prod = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=pos1, data2=mask, op=nl.multiply)
            col = topk + t
            nisa.tensor_scalar(dst=acc[:, col:col + 1], data=prod, op0=nl.subtract, operand0=1)

        nl.store(out[start:start + height, 0:out_cols], value=acc)
    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(rows: int, n_groups: int, pool_size: int, out_cols: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    A folded helper may take ints only: Dynamo runs it at trace time and refuses to
    reconstruct an ``@nki.jit`` object, so the kernel is read as a module global rather than
    passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_index_expand_nki)
    # Both widths are logged: a line carrying only the emitted width cannot tell meaningful
    # columns from sentinel padding.
    logger.info(
        "[dsa-index-expand] kernel=nki rows=%d n_groups=%d pool_size=%d raw_cols=%d out_cols=%d",
        rows,
        n_groups,
        pool_size,
        index_expand_raw_width(n_groups, pool_size),
        out_cols,
    )


def _validate(pool_ids: Tensor, seq_lens: Tensor, pool_size: int) -> tuple[int, int]:
    """Host-side shape validation. Returns ``(rows, n_groups)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor value, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace. That is also why the caller
    precondition on the pool id values themselves is not checked.
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

    The power-of-two check is load-bearing rather than defensive: the kernel derives
    ``tail_start`` as ``seq_len - (seq_len & (pool_size - 1))``, which is silently wrong -- not
    an error -- for any other ``pool_size``.
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
    """Selected pool ids expanded to token indices, with the tail appended.

    Args:
        pool_ids: ``[rows, n_groups]`` int32 -- the selected pool ids, ``-1`` where no pool
            was selected. Every non-negative id must satisfy the caller precondition in the
            module docstring.
        seq_lens: ``[rows]`` int32 -- the per-row sequence length.
        pool_size: how many tokens one pool covers. Must be a power of two to take the NKI
            route; defaults to this checkpoint's compress ratio.

    Returns:
        ``[rows, index_expand_width(n_groups, pool_size)]`` int32 token indices in upstream's
        column order, where ``-1`` is the sentinel for "this column selects no token" and is a
        value rather than an out-of-bounds index. The first
        ``index_expand_raw_width(n_groups, pool_size)`` columns carry meaning and every column
        after them is ``-1``; the emitted width is a positive multiple of ``KEY_CHUNK`` so that
        ``mla_sparse_attention`` admits it. Ask the two width functions rather than recomputing
        either.

    Raises:
        IndexExpandError: for a malformed call -- a non-2-D ``pool_ids``, a ``seq_lens`` that
            is not 1-D or does not match the row count, or a non-positive ``pool_size``.
    """
    rows, n_groups = _validate(pool_ids, seq_lens, pool_size)
    out_cols = index_expand_width(n_groups, pool_size)

    if not can_run_dsa_index_expand(pool_ids, seq_lens, pool_size):
        _count_torch_fallback()
        return _dsa_index_expand_torch(pool_ids, seq_lens, pool_size)

    # Every per-row value reaches a `tensor_scalar` as a column operand, so the lengths are
    # reshaped once here rather than once per use on the device.
    seq_col = seq_lens.reshape(rows, 1).contiguous()

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter
    # store inside the trace becomes a value guard that fails on the first call after warmup.
    _record_nki_dispatch(rows, n_groups, pool_size, out_cols)
    return wrap_nki(_index_expand_nki)(
        pool_ids.contiguous(), seq_col, pool_size, pool_size - 1
    )


# ---------------------------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------------------------


def _dsa_index_expand_torch(pool_ids: Tensor, seq_lens: Tensor, pool_size: int) -> Tensor:
    """CPU reference for the kernel, and the server for a call the gate refuses.

    Kept in the reference implementation's ``where`` form rather than rewritten into the
    kernel's closed forms: if the two spellings were the same, agreeing with this would only
    prove the kernel agrees with itself.

    The padded columns come back ``-1`` from the predicate that already decides the tail --
    ``tail_offset`` for a padded column is at least ``pool_size - 1`` and ``tail_count`` is at
    most ``pool_size - 1``, so ``is_tail`` is False. The gather stays in bounds there without a
    guard because ``group`` is clamped to ``n_groups - 1``, producing a history value that
    ``is_history`` then discards.
    """
    rows, n_groups = (int(d) for d in pool_ids.shape)
    topk = n_groups * pool_size
    out_cols = index_expand_width(n_groups, pool_size)

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
