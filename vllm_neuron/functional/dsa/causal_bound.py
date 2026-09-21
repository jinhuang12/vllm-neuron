# SPDX-License-Identifier: Apache-2.0
"""Causal bound for the sparse-attention indexer's pool scores.

A query row must not select a key pool that finishes after the row's own
position. Pool ``p`` covers tokens ``p * pool_size .. (p + 1) * pool_size - 1``,
so it is complete for a row of length ``causal_len`` exactly when
``(p + 1) * pool_size <= causal_len``. :func:`dsa_causal_bound` pushes every
incomplete pool's score down to :data:`BOUND_FILL`; :func:`dsa_causal_sentinel`
then rewrites any selection that comes back holding that fill, or holding an
index past the real pool width, to :data:`SENTINEL`.

The fill is finite rather than ``-inf`` because the selector between those two
stages pads its own input with a finite constant and moves selected values
between partitions with a 0/1 permutation matmul, where ``0 * -inf`` is NaN. A
row shorter than one pool completes nothing, so all of its columns are filled;
what such a row means is the attention consumer's contract, not this module's.
Both kernels walk the query-token axis in tiles of at most
:data:`PARTITION_MAX` rows, the most one SBUF tile's partition axis holds.
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

SENTINEL = -1
"""What a selection that reaches no valid pool holds. The producer's value, written here and
masked there. Upstream writes the same ``-1`` for a candidate-less slot (``sampler.cu:405``)."""

BOUND_FILL = -1.0e30
"""The bounded-score value: finite, and far below any score the indexer can produce.

Upstream fills with ``-inf`` here (``rocm_aiter_mla_sparse.py:734``); this path cannot. The
selector this bound feeds moves selected values between partitions with a 0/1 permutation
matrix on the tensor engine, and ``0 * -inf`` is NaN under IEEE-754, so an ``-inf`` fill need
not come back as ``-inf`` and a marker keyed on the exact value would then mark nothing. A
finite fill crosses a permutation matmul unchanged: every output is one ``1 * value`` term
plus zeros.

The magnitude satisfies three constraints. It is below every legal indexer score, which are
softmax-scale logits of tens at most, so no real candidate can be mistaken for a fill. It is
far above ``FLOAT32_MIN`` (-3.4e38), so a matmul accumulation cannot overflow to an infinity.
And it survives a bfloat16 round-trip with room to spare, which :data:`BOUND_FILL_MARK`
needs. Finite mask fills are the existing convention where a kernel consumes them:
``functional/sampling.py:263`` masks with ``-3000.0`` and
``functional/attention/attention_cte.py:145`` with ``torch.finfo(dtype).min``."""

BOUND_FILL_MARK = -1.0e29
"""The threshold the marker fires at or below. Ten times closer to zero than :data:`BOUND_FILL`.

The gap is what makes the marker independent of dtype rounding: a ``BOUND_FILL`` that has been
rounded to bfloat16 and back is still orders of magnitude below this, while no real score comes near
it. A ``-inf`` that somehow survives is below it too, so this threshold also catches the value the
vendored selector itself writes when it strikes a taken candidate
(``rotational_topk_utils.py:1065``)."""

_SCORE_DTYPES = (torch.float32,)
"""Score dtypes that take the NKI route. The bound compares against a length and writes a float
sentinel; fp32 is what ``dsa_score_gemm`` hands the selector on this path."""

_INDEX_DTYPES = (torch.int32,)
"""Index dtypes that take the NKI route. ``dsa_index_expand`` admits int32, so the sentinel writer
keeps the selector's output in the dtype its consumer reads."""

PARTITION_MAX = 128
"""Query rows one SBUF tile can hold: the partition-axis bound, ``nl.tile_size.pmax``.

This bounds one row tile, not the call. The kernel walks the query-token axis in tiles of at
most this height, so a prefill with more tokens than this is served rather than trapped in the
assert the module docstring quotes. It is a module constant so both kernels and the tile
arithmetic read one number.
"""


class DsaCausalBoundError(ValueError):
    """A malformed call: a ``pool_size`` that is not a power of two, a ``causal_len`` of the wrong
    dtype, or a ``causal_len`` whose row count does not match ``scores``."""


@dataclass
class _DispatchCounters:
    """Per-process dispatch counters for one entry point.

    One instance per entry point, each with its own accessor and reset, so that a
    test can tell a bound call from a sentinel call instead of reading their sum.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_BOUND = _DispatchCounters()
_SENTINEL_COUNTERS = _DispatchCounters()


def reset_causal_bound_dispatch_counters() -> None:
    """Zero the bound entry point's counters."""
    _BOUND.nki_dispatch = 0
    _BOUND.torch_fallback = 0
    _BOUND.last_kernel = None


def causal_bound_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for :func:`dsa_causal_bound` since its last reset."""
    return (_BOUND.nki_dispatch, _BOUND.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_bound_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _BOUND.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_bound_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _BOUND.nki_dispatch += 1


def causal_bound_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the bound seam last dispatched, or ``None``."""
    return _BOUND.last_kernel


def reset_causal_sentinel_dispatch_counters() -> None:
    """Zero the sentinel entry point's counters."""
    _SENTINEL_COUNTERS.nki_dispatch = 0
    _SENTINEL_COUNTERS.torch_fallback = 0
    _SENTINEL_COUNTERS.last_kernel = None


def causal_sentinel_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for :func:`dsa_causal_sentinel` since its last reset."""
    return (_SENTINEL_COUNTERS.nki_dispatch, _SENTINEL_COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_sentinel_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _SENTINEL_COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_sentinel_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _SENTINEL_COUNTERS.nki_dispatch += 1


def causal_sentinel_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the sentinel seam last dispatched, or ``None``."""
    return _SENTINEL_COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator instead.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Query-token tiling, written once and read by both kernel bodies
# ---------------------------------------------------------------------------------------------


def _row_tiles_unchecked(rows: int) -> list[tuple[int, int]]:
    """The ``(start, height)`` query-token tiles, in order, with no refusal in the arithmetic.

    Written with only ``for``, ``range`` and ``append`` because the tracer has refused a list
    comprehension and a ``min`` where the kernel bodies reach this helper. One query token is
    one row, so no tile boundary can split anything and every full tile is
    :data:`PARTITION_MAX` high.
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
    """How many tiles :func:`_row_tiles_unchecked` returns. Ceiling division.

    The kernel bodies need the count as a plain loop bound and index the tile list, because
    the tracer refused a ``for`` whose loop variable is a tuple.
    """
    return (rows + PARTITION_MAX - 1) // PARTITION_MAX


def row_tiles(rows: int) -> list[tuple[int, int]]:
    """The ``(start, height)`` query-token tiles both kernels walk, in order. The checked path.

    The kernel bodies call :func:`_row_tiles_unchecked` instead, because this function raises
    and NKI refuses a traced ``raise``; the two return the same list for every admissible
    input. The last tile is short whenever ``rows`` is not a multiple of
    :data:`PARTITION_MAX`.

    Raises:
        DsaCausalBoundError: if ``rows`` is not positive.
    """
    if rows < 1:
        raise DsaCausalBoundError(
            f"rows must be the positive number of query rows to tile; got rows={rows}"
        )
    return _row_tiles_unchecked(rows)


def row_tile_count(rows: int) -> int:
    """How many tiles :func:`row_tiles` returns. The checked path, and the same refusal."""
    if rows < 1:
        raise DsaCausalBoundError(
            f"rows must be the positive number of query rows to tile; got rows={rows}"
        )
    return _row_tile_count_unchecked(rows)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _causal_bound_nki(scores_hbm, causal_len_hbm, pool_size):
    """:data:`BOUND_FILL` at every pool column the row's own length does not complete.

    Args:
        scores_hbm: ``[rows, width]`` float32 -- one score per candidate pool per query row.
        causal_len_hbm: ``[rows, 1]`` int32 -- each row's own causal length, already a column,
            because it reaches ``tensor_scalar`` as a per-row scalar operand.
        pool_size: python int, tokens per pool. A trace-time constant.

    Returns:
        ``[rows, width]`` float32. Column ``p`` of row ``i`` holds :data:`BOUND_FILL` when
        ``(p + 1) * pool_size > causal_len[i]`` and row ``i``'s original score, bit for bit,
        otherwise.

    The scores are copied into SBUF once and the bounded columns are overwritten in place by
    one predicated copy, so a kept column carries the loaded bits unchanged: no add of ``0.0``
    to turn ``-0.0`` into ``+0.0``, and no ``0 * -inf`` to produce NaN. The query-token axis is
    walked in tiles of at most :data:`PARTITION_MAX` rows, each tile reading only its own rows
    of both inputs.
    """
    rows = scores_hbm.shape[0]
    width = scores_hbm.shape[1]

    out = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.shared_hbm)

    # The unchecked tile arithmetic, because the tracer follows these calls: the checked
    # `row_tiles` raises, and NKI refuses a traced `raise`.
    tiles = _row_tiles_unchecked(int(rows))
    tile_count = _row_tile_count_unchecked(int(rows))

    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]

        # This tile's scores, loaded once. The mask writes into this tile, so it is also this
        # tile's slice of the result.
        scores_sb = nl.ndarray((height, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=scores_sb, src=nl.load(scores_hbm[start:start + height, 0:width])
        )

        # This tile's per-row lengths as a float32 column operand, re-loaded per tile because
        # `causal_len` is per row: a hoisted column would bound every tile by the first tile's
        # rows and would still read correct at `rows <= PARTITION_MAX`. float32 because the ISA
        # requires a `tensor_scalar` operand tile to be float32 and the MLIR verifier refuses
        # int32 there. Exact, since a causal length is a whole number far below 2**24.
        clen = nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=clen, src=nl.load(causal_len_hbm[start:start + height, 0:1])
        )

        # Negated here so the one tile-operand call below can be an add; `tensor_scalar` has no
        # precedent in this tree for a subtract against a tile operand.
        nclen = nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nclen, data=clen, op0=nl.multiply, operand0=-1.0)

        # The column ramp `p`, the same 0..width-1 on every partition of this tile.
        # `nisa.iota` with `channel_multiplier=0`, because this NKI image has no `nl.arange`,
        # `mgrid`, `nl.iota` or `nl.affine_select` to synthesise it.
        ramp = nl.ndarray((height, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=ramp, pattern=[[1, width]], offset=0, channel_multiplier=0)

        # `(p + 1) * pool_size`, the first token index past pool `p`, as one two-scalar chain.
        # The bound is written as this multiply-and-compare rather than
        # `p >= causal_len // pool_size` because `nl.divide` is silently wrong on int32 and
        # `nl.right_shift` refuses as the second op of a chain. Over integers the two agree.
        end = nl.ndarray((height, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=end, data=ramp,
                           op0=nl.multiply, operand0=float(pool_size),
                           op1=nl.add, operand1=float(pool_size))

        # `(p + 1) * pool_size - causal_len[i]`, one tile operand broadcast along the free axis.
        room = nl.ndarray((height, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=room, data=end, op0=nl.add, operand0=nclen)

        # 1 exactly where the pool is incomplete for this row, which is where the bound applies.
        # `greater` into an integer destination is the compare form this tree already uses.
        bounded = nl.ndarray((height, width), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=bounded, data=room, op0=nl.greater, operand0=0.0)

        fill = nl.ndarray((height, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fill, value=BOUND_FILL)

        # `scores_sb` is left alone wherever `bounded` is 0, which is what keeps the untouched
        # columns bit-identical rather than merely arithmetically unchanged.
        nisa.tensor_copy_predicated(src=fill, predicate=bounded, dst=scores_sb)

        nl.store(out[start:start + height, 0:width], value=scores_sb)
    return out


@nki.jit
def _causal_sentinel_nki(values_hbm, indices_hbm, width):
    """``-1`` at every selection that reaches no pool the row may see -- by value or by index.

    Args:
        values_hbm: ``[rows, k]`` float32 -- the selector's returned values.
        indices_hbm: ``[rows, k]`` int32 -- the selector's returned pool ids.
        width: python int, the number of real pool columns the selector was given. A trace-time
            constant, the way ``pool_size`` is in :func:`_causal_bound_nki`.

    Returns:
        ``[rows, k]`` int32. :data:`SENTINEL` at every slot whose value is at or below
        :data:`BOUND_FILL_MARK` and at every slot whose index is ``>= width``; the selector's
        own index, bit for bit, everywhere else.

    Two arms are needed because the selector's own padding is finite. The value arm catches a
    column the bound filled: those are real columns with legal indices, so only the value tells
    them apart. The index arm catches a pad column the selector invented, which carries an
    ordinary finite value that outranks every bound-filled column -- so it wins slots on the
    very rows this bound acts on, and only its position past ``width`` gives it away.

    The result tile starts as all :data:`SENTINEL` and an index is copied in only where the
    value is a real score, so marking is the default and keeping is the exception. A NaN fails
    that ``greater`` comparison and is marked too, which keeps this kernel independent of
    whether the tensor engine's ``0 * x`` follows IEEE-754.
    """
    rows = values_hbm.shape[0]
    k = values_hbm.shape[1]

    out = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)

    # The unchecked tile arithmetic, for the reason `_causal_bound_nki` gives.
    tiles = _row_tiles_unchecked(int(rows))
    tile_count = _row_tile_count_unchecked(int(rows))

    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]

        # This tile's indices, loaded once. The pad screen writes into this tile.
        idx_sb = nl.ndarray((height, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=idx_sb, src=nl.load(indices_hbm[start:start + height, 0:k])
        )

        vals = nl.ndarray((height, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=vals, src=nl.load(values_hbm[start:start + height, 0:k])
        )

        # The sentinel source, and the result that starts out entirely sentinel.
        fill = nl.ndarray((height, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=fill, value=SENTINEL)
        marked = nl.ndarray((height, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=marked, value=SENTINEL)

        # The index arm. `index >= width` is written as `index > width - 1`, so that `greater` is
        # the only comparison this kernel needs.
        idxf = nl.ndarray((height, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idxf, src=idx_sb)
        pad = nl.ndarray((height, k), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=pad, data=idxf, op0=nl.greater, operand0=float(width) - 1.0
        )
        nisa.tensor_copy_predicated(src=fill, predicate=pad, dst=idx_sb)

        # The value arm, as a keep mask: 1 where the value is a real score. A fill, an `-inf`
        # and a NaN all fail this compare and stay at the sentinel the result already holds.
        keep = nl.ndarray((height, k), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=vals, op0=nl.greater, operand0=BOUND_FILL_MARK)
        nisa.tensor_copy_predicated(src=idx_sb, predicate=keep, dst=marked)

        nl.store(out[start:start + height, 0:k], value=marked)
    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_bound_dispatch(rows: int, width: int) -> None:
    """Record which kernel the bound seam dispatched, and log it, off the compiled graph.

    A folded helper takes ints only: Dynamo runs it at trace time and refuses to reconstruct an
    ``@nki.jit`` object, so the kernel is read as a module global instead of passed in.
    """
    _BOUND.last_kernel = _kernel_identity_of(_causal_bound_nki)
    logger.info("[dsa-causal-bound] kernel=nki rows=%d width=%d", rows, width)


@torch._dynamo.assume_constant_result
def _record_sentinel_dispatch(rows: int, k: int, width: int) -> None:
    """Record which kernel the sentinel seam dispatched, and log it, off the compiled graph."""
    _SENTINEL_COUNTERS.last_kernel = _kernel_identity_of(_causal_sentinel_nki)
    logger.info(
        "[dsa-causal-sentinel] kernel=nki rows=%d k=%d width=%d", rows, k, width
    )


def _validate_bound(scores: Tensor, causal_len: Tensor, pool_size: int) -> int:
    """Host-side validation for the bound. Returns ``rows``.

    Reads only ``.shape`` and ``.dtype``, never a value, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace. All three refusals raise rather
    than declining to the torch path, because each is a caller bug that a correct-looking
    answer would hide.
    """
    if not isinstance(pool_size, int) or isinstance(pool_size, bool):
        raise DsaCausalBoundError(
            f"pool_size must be a python int, because it is a trace-time constant; got "
            f"{type(pool_size).__name__} {pool_size!r}"
        )
    if pool_size < 1 or (pool_size & (pool_size - 1)) != 0:
        raise DsaCausalBoundError(
            f"pool_size must be a power of two; got pool_size={pool_size}. The pooling geometry "
            f"this bound shares with dsa_index_expand and dsa_kpool_hadamard is a power of two "
            f"throughout, so any other value is a caller error rather than a shape this kernel "
            f"declines to serve"
        )
    if scores.ndim != 2:
        raise DsaCausalBoundError(
            f"scores must be 2-D [rows, width], one score per candidate pool per query row; got "
            f"shape {tuple(scores.shape)}"
        )
    if causal_len.dtype not in _INDEX_DTYPES:
        raise DsaCausalBoundError(
            f"causal_len must be int32, the dtype the indexer carries and the consumer reads; got "
            f"{causal_len.dtype}"
        )
    rows = int(scores.shape[0])
    if int(causal_len.shape[0]) != rows:
        raise DsaCausalBoundError(
            f"causal_len must carry one length per score row; got {int(causal_len.shape[0])} "
            f"lengths for {rows} score rows"
        )
    return rows


def can_run_dsa_causal_bound(scores: Tensor, causal_len: Tensor, pool_size: int) -> bool:
    """Whether the NKI kernel serves this bound call. ``False`` sends it to the torch path.

    Narrow on purpose: every malformed call has already raised in :func:`_validate_bound`, so
    the only question left is whether NKI is available at all.
    """
    if not can_run_kernel():
        return False
    return scores.dtype in _SCORE_DTYPES and causal_len.dtype in _INDEX_DTYPES


def dsa_causal_bound(scores: Tensor, causal_len: Tensor, pool_size: int) -> Tensor:
    """:data:`BOUND_FILL` at every pool a query row's own length does not complete.

    Args:
        scores: ``[rows, width]`` float32, one score per candidate pool per query row, as
            ``dsa_score_gemm`` returns them.
        causal_len: ``[rows, 1]`` or ``[rows]`` int32 -- each row's own causal length. This is
            the same per-row length column ``dsa_index_expand``'s call already receives; it is
            not recomputed here.
        pool_size: tokens per pool, a power of two.

    Returns:
        ``[rows, width]`` float32, bounded as :func:`_causal_bound_nki` describes.

    Raises:
        DsaCausalBoundError: for a non-int or non-power-of-two ``pool_size``, a ``scores`` that
            is not 2-D, a ``causal_len`` that is not int32, or a row-count mismatch.
    """
    rows = _validate_bound(scores, causal_len, pool_size)

    if not can_run_dsa_causal_bound(scores, causal_len, pool_size):
        _count_bound_torch_fallback()
        return dsa_causal_bound_torch_oracle(scores, causal_len, pool_size)

    # The per-row length reaches `tensor_scalar` as a column operand, so the reshape is done
    # once here rather than per use on the device.
    clen_col = causal_len.reshape(rows, 1).contiguous()

    _count_bound_nki_dispatch()
    _record_bound_dispatch(rows, int(scores.shape[1]))
    return wrap_nki(_causal_bound_nki)(scores.contiguous(), clen_col, pool_size)


def _validate_sentinel(values: Tensor, indices: Tensor, width: int) -> int:
    """Host-side validation for the sentinel writer. Returns ``rows``.

    ``width`` is checked the way ``pool_size`` is in :func:`_validate_bound` and for the same
    reason: it is a trace-time constant, so a tensor or a bool arriving here would be baked
    into the graph as something the caller did not mean. The index arm has no meaning without
    it, so a bad ``width`` raises rather than declining to the torch path.
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise DsaCausalBoundError(
            f"width must be a python int, because it is a trace-time constant; got "
            f"{type(width).__name__} {width!r}"
        )
    if width < 1:
        raise DsaCausalBoundError(
            f"width must be the positive number of real pool columns the selector was given; got "
            f"width={width}"
        )
    if values.ndim != 2 or indices.ndim != 2:
        raise DsaCausalBoundError(
            f"values and indices must both be 2-D [rows, k]; got {tuple(values.shape)} and "
            f"{tuple(indices.shape)}"
        )
    if tuple(values.shape) != tuple(indices.shape):
        raise DsaCausalBoundError(
            f"values and indices must be the same shape, one value per selected index; got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}"
        )
    if indices.dtype not in _INDEX_DTYPES:
        raise DsaCausalBoundError(
            f"indices must be int32, the dtype dsa_index_expand admits; got {indices.dtype}"
        )
    return int(values.shape[0])


def can_run_dsa_causal_sentinel(values: Tensor, indices: Tensor, width: int) -> bool:
    """Whether the NKI kernel serves this sentinel call. ``False`` sends it to the torch path.

    ``width`` is accepted so the gate has the same signature as the call it is about, but is
    not read here: a malformed ``width`` has already raised in :func:`_validate_sentinel`.
    """
    if not can_run_kernel():
        return False
    return values.dtype in _SCORE_DTYPES and indices.dtype in _INDEX_DTYPES


def dsa_causal_sentinel(values: Tensor, indices: Tensor, width: int) -> Tensor:
    """:data:`SENTINEL` at every selection that reaches no pool the row may see.

    Args:
        values: ``[rows, k]`` float32, the selector's returned values.
        indices: ``[rows, k]`` int32, the selector's returned pool ids.
        width: the number of real pool columns the selector was given, which is
            ``bounded.shape[1]`` at the dispatch site. Anything the selector returns at or
            above it is a pad it invented, not a pool -- see :func:`_causal_sentinel_nki`.

    Returns:
        ``[rows, k]`` int32, sentinelised as :func:`_causal_sentinel_nki` describes.

    Raises:
        DsaCausalBoundError: for a non-int or non-positive ``width``, a rank or shape mismatch,
            or a non-int32 ``indices``.
    """
    rows = _validate_sentinel(values, indices, width)

    if not can_run_dsa_causal_sentinel(values, indices, width):
        _count_sentinel_torch_fallback()
        return dsa_causal_sentinel_torch_oracle(values, indices, width)

    _count_sentinel_nki_dispatch()
    _record_sentinel_dispatch(rows, int(values.shape[1]), width)
    return wrap_nki(_causal_sentinel_nki)(values.contiguous(), indices.contiguous(), width)


# ---------------------------------------------------------------------------------------------
# Torch references, also taken as the fallback when the NKI route is unavailable
# ---------------------------------------------------------------------------------------------


def dsa_causal_bound_torch_oracle(
    scores: Tensor, causal_len: Tensor, pool_size: int
) -> Tensor:
    """CPU reference for the bound, and the route taken when NKI is unavailable.

    Deliberately kept in the floor-division spelling ``p >= causal_len // pool_size`` rather
    than the kernel's rearranged ``(p + 1) * pool_size > causal_len``: if the two spellings
    matched, agreeing with this would only show that the module agrees with itself.
    """
    rows, width = int(scores.shape[0]), int(scores.shape[1])
    complete = (causal_len.reshape(rows, 1).to(torch.int64) // pool_size)
    cols = torch.arange(width, device=scores.device, dtype=torch.int64)[None, :]
    return scores.masked_fill(cols >= complete, BOUND_FILL)


def dsa_causal_sentinel_torch_oracle(
    values: Tensor, indices: Tensor, width: int
) -> Tensor:
    """CPU reference for the sentinel writer, and the route taken when NKI is unavailable.

    Written in the opposite polarity to the kernel on purpose. This builds the mark mask
    directly out of ``le`` and ``ge`` and names NaN with ``torch.isnan``, where the kernel
    starts from an all-``-1`` tile and gets NaN for free from a failed ``greater``.
    """
    filled = torch.le(values, BOUND_FILL_MARK) | torch.isnan(values)
    pad = torch.ge(indices, width)
    return torch.where(filled | pad, torch.full_like(indices, SENTINEL), indices)
