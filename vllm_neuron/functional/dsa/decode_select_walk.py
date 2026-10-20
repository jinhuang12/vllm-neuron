# SPDX-License-Identifier: Apache-2.0
"""The DSA decode pools selection on a rank axis wider than one compaction row.

Under decode context parallelism (DCP) each rank selects its local top-``k`` over its own
pools, and the group merges the ranks' lists (``dcp_candidate_merge``), which reads each
entry's score beside its pool id. :func:`decode_select.dsa_decode_select_pools` gives both on
the whole-row path only: a rank axis of up to :data:`decode_select.WHOLE_ROW_COLUMNS` pools
(``max_model_len <= 65536 x DCP`` at ``index_kpool`` 4). :func:`dsa_decode_select_pools_walk`
gives the same two lists on a wider axis, up to :data:`MAX_DECODE_POOLS_WALK_CANDIDATES`, with
the one-rank chunk walk (``decode_select``'s module docstring, "The compaction") and the
whole-row path's score gather, per chunk:

1. **The selection mask**: ``decode_select._select_rows`` over the mask span, whole chunks
   of ``WHOLE_ROW_COLUMNS`` (the scores past the axis are ``BOUND_FILL``): the same
   threshold search, tie rule (lowest candidate index first) and causal sentinel.
2. **Per (request, chunk) row**, eight rows per ``nonzero_with_count``, one per GpSimd
   core: the chunk's mask in columns ``1 .. WHOLE_ROW_COLUMNS`` of a tile whose column 0
   holds ``POOL_PAD_VALUE``; its ids to the second half of the row's id list, behind ``k``
   columns of ``_LIST_PAD``, and its count beside them. Then the chunk's scores over the
   mask, in the same tile, and one ``nc_n_gather`` at column ``max(id, -1) + 1`` (a pad
   reads column 0): the values to the second half of the row's value list, behind ``k``
   columns of ``POOL_PAD_VALUE``. A real id is below its chunk's columns inside the axis,
   so no gather reads past the scores.
3. **The windows**: ``k`` columns of chunk ``c``'s lists from column ``k - p``, ``p`` the
   ids of the request's lower chunks (an offset each DMA reads from SBUF when it runs):
   the chunk's entries land at their places in the joined list, pads everywhere else.
4. **The join**, an element-wise maximum over the chunks' windows: the ids shifted by
   their chunk's first candidate (a pad stays negative), then ``max(id, -1)``; the values
   as they are. At each place at most one window holds a real entry and every other one a
   pad, which is below every id and every real score (``POOL_PAD_VALUE`` is
   ``BOUND_FILL``; a real score is above ``BOUND_FILL_MARK``). So the join returns each
   entry bit for bit, and a place no chunk fills returns the pad.

The lists are ``dsa_decode_select_pools``' second and third outputs: ``values [R, k]`` fp32
and ``pool_ids [R, k]`` int32, the ids ascending, then the pads. There are no tokens: the
DCP caller expands the merged ids (``Glm5NextDSAIndexer.forward_requests``).

The one-rank chunk walk has run on trn2 at four chunks and one request per step
(``decode_select.DEVICE_CHECKED_ROW_CHUNKS``). This walk, with its value lists and value
join, has not run on the device. A decode step that batches more than one request keeps
the whole-row bound (:func:`check_decode_pools_walk_context`), as the one-rank rule does.
Each launch counts as ``select`` in ``decode_batch``'s family and in this module's count
(:func:`walk_dispatch_count`). Both cores of an LNC2 core split the rows from two rows on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa import decode_select
from vllm_neuron.functional.dsa.decode_batch import (
    PARTITIONS,
    _count_dispatch,
    _count_torch_fallback,
)
from vllm_neuron.functional.dsa.decode_select import (
    _LIST_PAD,
    COMPACT_ROWS,
    GPSIMD_STRIDE,
    MAX_DECODE_POOLS_CANDIDATES,
    POOL_PAD_ID,
    POOL_PAD_VALUE,
    WHOLE_ROW_COLUMNS,
    DecodeSelectError,
    _ceil_div,
    _check_context,
    _gpsimd_copy,
    _gpsimd_dst,
    _program_share,
    _sb,
    _select_rows,
    _selected_pools_torch,
    decode_select_programs,
    mask_span,
    row_chunks,
    select_fold_cap,
)
from vllm_neuron.source_digest import source_digest
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The most whole-row chunks the walk serves: the chunk count the one-rank walk has run on
#: trn2 with (``decode_select.DEVICE_CHECKED_ROW_CHUNKS``). This walk's own first device run
#: is still to come; raise this bound only after a device check of the walk at the new count.
POOLS_WALK_ROW_CHUNKS = 4
#: The widest rank candidate axis the walk serves at one request per decode step; the runner
#: refuses a longer per-rank context (:func:`check_decode_pools_walk_context`).
MAX_DECODE_POOLS_WALK_CANDIDATES = POOLS_WALK_ROW_CHUNKS * WHOLE_ROW_COLUMNS

#: This file's and :mod:`decode_select`'s content, handed to the kernel as a trace-time int:
#: the compiled-kernel cache keys on a kernel's own source and its arguments, and this
#: kernel's body reuses that module's helpers.
SOURCE_DIGEST = source_digest(__file__, decode_select.__file__)

_FILE = "decode_select_walk.py"


@dataclass
class _WalkCounters:
    dispatch: int = 0


_COUNTERS = _WalkCounters()


def reset_walk_dispatch_count() -> None:
    """Zero the walk's launch count."""
    _COUNTERS.dispatch = 0


def walk_dispatch_count() -> int:
    """Walk kernel launches since the last reset."""
    return _COUNTERS.dispatch


@torch._dynamo.assume_constant_result
def _count_walk_dispatch() -> None:
    """Count one walk launch, off the traced graph."""
    _COUNTERS.dispatch += 1


# ---------------------------------------------------------------------------------------
# Device helpers. Positional arguments only: the NKI front end drops keyword defaults.
# ---------------------------------------------------------------------------------------


def _chunks_past_pad(tile, hbm, row0, m, chunks):
    """Chunk rows ``row0 .. row0 + m - 1`` of a ``[requests, width]`` HBM tensor into
    columns ``1 ..`` of partitions ``0, 16, .., 16 (m - 1)`` of ``tile``, past its column 0.

    Row ``q`` is chunk ``q % chunks`` of request ``q // chunks``: its columns inside the
    width, at most :data:`WHOLE_ROW_COLUMNS`. One DMA per row, as ``_gpsimd_copy``.
    """
    width = hbm.shape[1]
    for i in range(m):
        q = row0 + i
        first = (q % chunks) * WHOLE_ROW_COLUMNS
        cols = min(WHOLE_ROW_COLUMNS, width - first)
        part = GPSIMD_STRIDE * i
        nisa.dma_copy(dst=tile[part:part + 1, 1:cols + 1],
                      src=hbm.ap(pattern=[[width, 1], [1, cols]],
                                 offset=(q // chunks) * width + first))


def _walk_pool_rows(mask_hbm, bounded_hbm, ids_hbm, values_hbm, lists_hbm, vlists_hbm,
                    counts_hbm, r0, n, chunks, select_k):
    """Requests ``r0 .. r0 + n - 1``: their mask rows compacted in ``chunks`` chunks, each
    id's score gathered beside it, and the chunk lists joined: the ids to ``ids_hbm``
    (ascending, :data:`POOL_PAD_ID` after them) and their scores to ``values_hbm``
    (:data:`POOL_PAD_VALUE` after them). The module docstring's steps 2 to 4.

    Row ``q = r * chunks + c`` (``r`` counted from 0) is chunk ``c`` of request ``r`` in
    each scratch: ``mask_hbm`` read as rows of :data:`WHOLE_ROW_COLUMNS`; ``lists_hbm``
    (int32) and ``vlists_hbm`` (fp32), ``[N, 1]`` so that a window offset counts single
    entries, as rows of ``2 k``; ``counts_hbm``, ``[N, 1]`` int32.

    The pad column and the score gather are ``decode_select._compact_pool_rows``': one tile
    for the mask rows and then the score rows, its pad column written before the mask rows
    and read by the gather after the score rows, so the score rows are a later write to
    the same tile, ordered after the compaction's read of the mask rows. The window starts,
    the windows and the id join are ``decode_select._compact_row_chunks``'; the value
    windows take the id windows' offsets.
    """
    chunk_cols = WHOLE_ROW_COLUMNS
    list_cols = 2 * select_k
    rows_n = n * chunks
    row0 = r0 * chunks

    # ---- 1. chunk lists behind pad prefixes, and their counts -------------------------
    prefix = _sb((rows_n, select_k), nl.int32)
    nisa.memset(dst=prefix, value=_LIST_PAD)
    nisa.dma_copy(dst=lists_hbm.ap(pattern=[[list_cols, rows_n], [1, select_k]],
                                   offset=row0 * list_cols), src=prefix)
    value_prefix = _sb((rows_n, select_k), nl.float32)
    nisa.memset(dst=value_prefix, value=POOL_PAD_VALUE)
    nisa.dma_copy(dst=vlists_hbm.ap(pattern=[[list_cols, rows_n], [1, select_k]],
                                    offset=row0 * list_cols), src=value_prefix)
    for g in range(_ceil_div(rows_n, COMPACT_ROWS)):
        m = min(COMPACT_ROWS, rows_n - g * COMPACT_ROWS)
        group_row = row0 + g * COMPACT_ROWS
        # Partitions 0 .. 16 (m - 1): every compute instruction here starts on partition 0.
        parts = GPSIMD_STRIDE * (m - 1) + 1
        rows = _sb((PARTITIONS, chunk_cols + 1), nl.float32)
        nisa.memset(dst=rows[:, 0:1], value=POOL_PAD_VALUE)
        found = _gpsimd_dst((PARTITIONS, chunk_cols + 1), nl.int32)
        # The partitions between the GpSimd ones gather too: a pad there reads column 0.
        nisa.memset(dst=found[0:parts, 0:select_k], value=_LIST_PAD)
        _chunks_past_pad(rows, mask_hbm, group_row, m, chunks)
        nisa.nonzero_with_count(dst=found, src=rows[:, 1:chunk_cols + 1], index_offset=0,
                                padding_val=_LIST_PAD)
        _gpsimd_copy(found, lists_hbm, group_row, m, select_k, list_cols, select_k, False)
        for i in range(m):
            part = GPSIMD_STRIDE * i
            nisa.dma_copy(dst=counts_hbm.ap(pattern=[[1, 1], [1, 1]], offset=group_row + i),
                          src=found[part:part + 1, chunk_cols:chunk_cols + 1])
        at = _sb((PARTITIONS, select_k), nl.int32)  # max(id, -1) + 1: the column of its score
        nisa.tensor_scalar(dst=at[0:parts, 0:select_k], data=found[0:parts, 0:select_k],
                           op0=nl.maximum, operand0=-1, op1=nl.add, operand1=1)
        # ---- the score rows over the mask rows: every real id is inside its chunk ------
        _chunks_past_pad(rows, bounded_hbm, group_row, m, chunks)
        values = _gpsimd_dst((PARTITIONS, select_k), nl.float32)
        nisa.nc_n_gather(dst=values[0:parts, 0:select_k], data=rows[0:parts, 0:chunk_cols + 1],
                         indices=at[0:parts, 0:select_k].view(nl.uint32))
        _gpsimd_copy(values, vlists_hbm, group_row, m, select_k, list_cols, select_k, False)

    # ---- 2. each window's start: k minus the ids of the request's lower chunks ---------
    counts = _sb((1, rows_n), nl.int32)
    nisa.dma_copy(dst=counts, src=counts_hbm.ap(pattern=[[rows_n, 1], [1, rows_n]],
                                                offset=row0))
    held = _sb((1, rows_n), nl.float32)  # a count of at most k: exact
    nisa.tensor_copy(dst=held, src=counts)
    start_f = _sb((1, rows_n), nl.float32)
    nisa.memset(dst=start_f, value=float(select_k))
    held_rc = held.reshape((1, n, chunks))
    start_rc = start_f.reshape((1, n, chunks))
    for c in range(1, chunks):
        nisa.tensor_tensor(dst=start_rc[0:1, 0:n, c:c + 1],
                           data1=start_rc[0:1, 0:n, c - 1:c],
                           data2=held_rc[0:1, 0:n, c - 1:c], op=nl.subtract)
    start = _sb((1, rows_n), nl.int32)
    nisa.tensor_copy(dst=start, src=start_f)

    # ---- 3. the windows: each chunk's entries at their places in the joined list -------
    windows = _sb((n, chunks * select_k), nl.int32)
    value_windows = _sb((n, chunks * select_k), nl.float32)
    for r in range(n):
        for c in range(chunks):
            q = r * chunks + c
            nisa.dma_copy(dst=windows[r:r + 1, c * select_k:(c + 1) * select_k],
                          src=lists_hbm.ap(pattern=[[list_cols, 1], [1, select_k]],
                                           offset=(row0 + q) * list_cols,
                                           scalar_offset=start[0:1, q:q + 1],
                                           indirect_dim=0))
            nisa.dma_copy(dst=value_windows[r:r + 1, c * select_k:(c + 1) * select_k],
                          src=vlists_hbm.ap(pattern=[[list_cols, 1], [1, select_k]],
                                            offset=(row0 + q) * list_cols,
                                            scalar_offset=start[0:1, q:q + 1],
                                            indirect_dim=0))

    # ---- 4. the join: the largest of the windows at each place -------------------------
    ids = _sb((n, select_k), nl.int32)
    nisa.tensor_copy(dst=ids, src=windows[0:n, 0:select_k])
    joined = _sb((n, select_k), nl.float32)
    nisa.tensor_copy(dst=joined, src=value_windows[0:n, 0:select_k])
    for c in range(1, chunks):
        nisa.scalar_tensor_tensor(dst=ids, data=windows[0:n, c * select_k:(c + 1) * select_k],
                                  op0=nl.add, operand0=float(c * chunk_cols), op1=nl.maximum,
                                  operand1=ids)
        nisa.tensor_tensor(dst=joined, data1=value_windows[0:n, c * select_k:(c + 1) * select_k],
                           data2=joined, op=nl.maximum)
    nisa.tensor_scalar(dst=ids, data=ids, op0=nl.maximum, operand0=-1)
    nisa.dma_copy(dst=ids_hbm.ap(pattern=[[select_k, n], [1, select_k]], offset=r0 * select_k),
                  src=ids)
    nisa.dma_copy(dst=values_hbm.ap(pattern=[[select_k, n], [1, select_k]],
                                    offset=r0 * select_k), src=joined)


# ---------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------


@nki.jit
def dsa_decode_select_pools_walk_kernel(bounded_hbm, select_k, fold_cap, source_digest):
    """Each row's local top-``k`` before the expansion, on a candidate axis of more than
    one compaction row: ``dsa_decode_select_pools_kernel``'s values and pool ids.

    Args:
        bounded_hbm: ``[R, C]`` fp32, each row's bounded candidate scores,
            ``WHOLE_ROW_COLUMNS < C <= MAX_DECODE_POOLS_WALK_CANDIDATES``.
        select_k: python int, pools selected per row, at most ``WHOLE_ROW_COLUMNS``.
        fold_cap: python int, ``select_fold_cap`` of the mask span.
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``(values, pool_ids)``: ``[R, select_k]`` fp32, the selected scores, then
        :data:`POOL_PAD_VALUE`; ``[R, select_k]`` int32, their candidate-axis indices in
        ascending order, then :data:`POOL_PAD_ID`. Entry ``j`` of ``values`` is the score of
        entry ``j`` of ``pool_ids``.

    The rows split evenly over the programs of the launch grid, each program's share in
    tiles of up to ``tile_requests`` of the mask span.
    """
    batch = bounded_hbm.shape[0]
    span = mask_span(bounded_hbm.shape[1])
    chunks = row_chunks(span)
    values_hbm = nl.ndarray((batch, select_k), dtype=nl.float32, buffer=nl.shared_hbm)
    pool_ids_hbm = nl.ndarray((batch, select_k), dtype=nl.int32, buffer=nl.shared_hbm)
    mask_hbm = nl.ndarray((batch, span), dtype=nl.float32, buffer=nl.private_hbm)
    # A list row of 2 k entries and a count per (row, chunk), one column each so that an
    # offset counts single entries.
    lists_hbm = nl.ndarray((batch * chunks * 2 * select_k, 1), dtype=nl.int32,
                           buffer=nl.private_hbm)
    vlists_hbm = nl.ndarray((batch * chunks * 2 * select_k, 1), dtype=nl.float32,
                            buffer=nl.private_hbm)
    counts_hbm = nl.ndarray((batch * chunks, 1), dtype=nl.int32, buffer=nl.private_hbm)
    lo, hi, per_tile, n_tiles = _program_share(batch, span)
    for t in range(n_tiles):
        r0 = lo + t * per_tile
        n = min(per_tile, hi - r0)
        _select_rows(bounded_hbm, mask_hbm, r0, n, select_k, fold_cap, span)
        _walk_pool_rows(mask_hbm, bounded_hbm, pool_ids_hbm, values_hbm, lists_hbm,
                        vlists_hbm, counts_hbm, r0, n, chunks, select_k)
    return values_hbm, pool_ids_hbm


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def check_decode_pools_walk_context(rank_context: int, pool_size: int | None,
                                    max_num_seqs: int) -> None:
    """Refuse a decode context parallel rank whose candidate axis is past the width its
    decode step serves.

    Args:
        rank_context: the tokens one rank's decode indexer covers.
        pool_size: the model's tokens per pool (``index_kpool``); ``None`` for a model with
            no DSA indexer, which is not checked.
        max_num_seqs: the most requests one decode step batches.

    Raises:
        DecodeSelectError: ``rank_context // pool_size`` is above
            :data:`MAX_DECODE_POOLS_WALK_CANDIDATES`, or, at ``max_num_seqs > 1``, above
            ``decode_select.MAX_DECODE_POOLS_CANDIDATES``: the walk serves one request per
            step, as the one-rank walk does.
    """
    if int(max_num_seqs) > 1:
        _check_context(rank_context, pool_size, MAX_DECODE_POOLS_CANDIDATES,
                       "MAX_DECODE_POOLS_CANDIDATES",
                       f"the decode context parallel selection at max_num_seqs={max_num_seqs}",
                       f"The whole-row bound is in decode_select.py; the walk of {_FILE} "
                       f"serves one request per step: lower max_model_len or set "
                       f"max_num_seqs=1.")
        return
    _check_context(rank_context, pool_size, MAX_DECODE_POOLS_WALK_CANDIDATES,
                   "MAX_DECODE_POOLS_WALK_CANDIDATES",
                   "the decode context parallel pools walk",
                   f"The bound is in {_FILE}: lower max_model_len or raise "
                   f"decode_context_parallel_size.")


def _validate_walk(bounded: Tensor, select_k) -> tuple[int, int]:
    """Check the operands of :func:`dsa_decode_select_pools_walk`; ``(R, C)`` of
    ``bounded``.

    Raises:
        DecodeSelectError: naming the first operand outside the walk's contract, its bound
            and this file.
    """
    if not torch.is_tensor(bounded) or bounded.ndim != 2:
        shape = tuple(bounded.shape) if torch.is_tensor(bounded) else repr(bounded)
        raise DecodeSelectError(f"bounded must be [R, C]; got {shape} ({_FILE})")
    rows, width = int(bounded.shape[0]), int(bounded.shape[1])
    if rows < 1:
        raise DecodeSelectError(f"bounded must carry at least one row; got "
                                f"{tuple(bounded.shape)} ({_FILE})")
    if bounded.dtype != torch.float32:
        raise DecodeSelectError(f"bounded must be float32, the score kernel's dtype; got "
                                f"{bounded.dtype} ({_FILE})")
    if (not isinstance(select_k, int) or isinstance(select_k, bool)
            or not 0 < select_k <= WHOLE_ROW_COLUMNS):
        raise DecodeSelectError(
            f"select_k must be a python int in [1, WHOLE_ROW_COLUMNS={WHOLE_ROW_COLUMNS}] "
            f"(a chunk lists at most its own columns); got {select_k!r} ({_FILE})")
    if width <= WHOLE_ROW_COLUMNS:
        raise DecodeSelectError(
            f"a {width}-wide candidate axis is one compaction row (<= WHOLE_ROW_COLUMNS="
            f"{WHOLE_ROW_COLUMNS}): decode_select.dsa_decode_select_pools serves it, not "
            f"the walk ({_FILE})")
    if width > MAX_DECODE_POOLS_WALK_CANDIDATES:
        raise DecodeSelectError(
            f"a {width}-wide candidate axis is past MAX_DECODE_POOLS_WALK_CANDIDATES="
            f"{MAX_DECODE_POOLS_WALK_CANDIDATES} ({POOLS_WALK_ROW_CHUNKS} chunks of "
            f"{WHOLE_ROW_COLUMNS}), the widest the decode context parallel pools walk "
            f"serves ({_FILE})")
    return rows, width


def dsa_decode_select_pools_walk(bounded: Tensor, *, select_k: int) -> tuple[Tensor, Tensor]:
    """A rank's local top-``k`` on a candidate axis wider than one compaction row.

    ``dsa_decode_select_pools``' ``(values, pool_ids)`` for
    ``WHOLE_ROW_COLUMNS < C <= MAX_DECODE_POOLS_WALK_CANDIDATES``: the input of the decode
    context parallel merge. No tokens: the caller expands the merged ids.

    Args:
        bounded: ``[R, C]`` fp32, a rank's bounded candidate scores over its local pool
            axis (``BOUND_FILL`` at every pool that is not complete for the row).
        select_k: pools selected per row, a python int in ``[1, WHOLE_ROW_COLUMNS]``.

    Returns:
        ``(values, pool_ids)``:

        * ``values``: ``[R, select_k]`` fp32, the score of each entry of ``pool_ids`` (bit
          for bit ``bounded[r, pool_ids[r, j]]``), :data:`POOL_PAD_VALUE` at a pad;
        * ``pool_ids``: ``[R, select_k]`` int32, the selected local candidate-axis indices
          in ascending order (the ``select_k`` largest scores, ties at the ``k``-th value to
          the lowest index), :data:`POOL_PAD_ID` after them in a row with fewer than
          ``select_k`` complete pools.

    Raises:
        DecodeSelectError: before any launch, for an operand outside that contract
            (:func:`_validate_walk`).

    Without a kernel (no device and no simulator) the torch oracle answers.
    """
    rows, width = _validate_walk(bounded, select_k)
    if not can_run_kernel(bounded):
        _count_torch_fallback()
        return dsa_decode_select_pools_walk_torch_oracle(bounded, select_k=select_k)
    programs = decode_select_programs(rows)
    _count_dispatch("select", programs, rows)
    _count_walk_dispatch()
    call = wrap_nki(dsa_decode_select_pools_walk_kernel)
    if programs == 2:
        call = call[2]
    return call(bounded.contiguous(), int(select_k), int(select_fold_cap(mask_span(width))),
                SOURCE_DIGEST)


# ---------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------


def dsa_decode_select_pools_walk_torch_oracle(bounded: Tensor, *, select_k: int
                                              ) -> tuple[Tensor, Tensor]:
    """CPU reference and fallback of :func:`dsa_decode_select_pools_walk`:
    ``decode_select._selected_pools_torch``'s pools (a stable descending sort, its first
    ``select_k`` that are complete pools, in ascending index order), each with its score,
    the pads after them."""
    width = int(bounded.shape[1])
    ids = _selected_pools_torch(bounded, select_k)
    real = ids < width
    scores = bounded.to(torch.float32)
    values = torch.gather(scores, 1, torch.where(real, ids, torch.zeros_like(ids)))
    values = torch.where(real, values, torch.full_like(values, POOL_PAD_VALUE))
    pool_ids = torch.where(real, ids, torch.full_like(ids, POOL_PAD_ID)).to(torch.int32)
    return values, pool_ids
