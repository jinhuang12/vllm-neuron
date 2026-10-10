# SPDX-License-Identifier: Apache-2.0
"""The DSA decode selection for ``B`` requests in one kernel.

``Glm5NextDSAIndexer.forward_requests`` turns each request's bounded candidate scores
(``[B, C]`` fp32, ``BOUND_FILL`` where a pool is not complete for the request) into the
token indices the sparse attention reads. The earlier route took four kernels for it
(the rotational top-k, the causal sentinel, the sentinel order and the index expand).
Here it is one kernel, :func:`dsa_decode_select`, which needs no sort:

1. **The ``k``-th largest score, by its bits.** Request ``b``'s threshold is found one bit
   at a time, most significant first, on the order-preserving integer key of an fp32 value
   (sign bit flipped for a positive value, every bit flipped for a negative one): bit ``i``
   is set when at least ``k`` scores are ``>=`` the value whose key is the bits found so
   far with bit ``i`` set. Each step is one count over the request's scores, compared as
   fp32 values, so the result is the exact ``k``-th largest value ``v`` after 32 steps.
2. **The selection.** Every score ``> v`` is selected, and of the scores ``== v`` the
   lowest-indexed ``k - #{> v}``: the **tie rule**, lowest candidate index first. A score
   ``<= BOUND_FILL_MARK`` is never selected, which is the causal sentinel: a row with fewer
   than ``k`` complete pools selects all of them and nothing else.
3. **The compaction.** The selected candidates are listed in ascending pool order
   (``nisa.nonzero_with_count``), ``-1`` after them.
4. **The expansion.** Pool ``p`` becomes tokens ``p * pool .. p * pool + pool - 1`` and the
   tail tokens of the incomplete final pool follow, in ``dsa_index_expand``'s layout:
   ``[B, index_expand_width(k, pool)]`` int32.

The selected set is the four-kernel route's: the same ``k`` largest scores and the same
rule for the masked pools. Two things differ, and neither is in the contract the attention
reads:

* The order of the real tokens: ascending pool order, where the four-kernel route had
  descending score order. ``mla_decode`` masks ``-1`` wherever it is and sums over the
  selected rows.
* The tie rule. The four-kernel route pinned nothing about equal values; this kernel pins
  the lowest index. The sets can differ only where values tie at the ``k``-th place.

Values compare as fp32 values: ``-0.0 == +0.0``. Scores must be finite (or the finite
``BOUND_FILL``): the bit search tests NaN keys on the way down to ``-inf``, so a ``k``-th
value of ``-inf`` is not found. The score kernel produces finite values.

Layout. A request's ``C`` scores fold over ``f`` partitions in chunks of ``F = ceil(C / f)``
columns, the rest padded with ``BOUND_FILL``. A tile carries ``n`` requests,
``n * f <= 128``: at most :data:`TILE_REQUESTS`, and few enough that a chunk stays within
:data:`FOLD_COLUMNS` (``n <= 128 // ceil(C / FOLD_COLUMNS)``); ``f = min(128 // n, cap)``
with the cap :func:`select_fold_cap` sets for ``C``. The partitions run chunk-major
(partition ``j * n + r`` holds chunk ``j`` of request ``r``, one DMA per chunk) or
request-major (partition ``r * f + j``, one DMA per request and one for its partial last
chunk), whichever needs fewer DMAs; every DMA is two-dimensional with the partitions
outermost, the form the device compiler aligns (a three-dimensional HBM access pattern
onto a partition range is refused there, though the simulator takes it). A count is then one
vector instruction over ``F`` columns and a cross-partition sum per request, one matmul by
the ``[P, P]`` same-request matrix; the tie rank adds the lower chunks' ties by the
strictly-lower same-request matrix. Neither depends on the layout.

The compaction needs a row on one partition of a GpSimd core (partitions
``0, 16, ..., 112``), so the mask goes through an HBM scratch on its way there. A row of
up to :data:`WHOLE_ROW_COLUMNS` candidates is compacted whole. A wider row is cut into ``S`` segments of at most :data:`SEGMENT_COLUMNS`: each
segment is compacted on its own (its ids shifted by the segment's first candidate) and
truncated to ``k`` columns, which loses nothing because the whole row selects at most
``k``. The segment lists stay on the request's GpSimd partition, side by side in segment
order, and merge :data:`MERGE_LISTS` at a time: ``nonzero_with_count`` over "is an id"
gives the positions of the ids in order, ``nc_n_gather`` reads them, and the result is
again one list of at most ``k`` ids in ascending order, the first list of the next merge.
Every shape is static; no step depends on how many pools a segment selected.

The decode indexer serves the whole-row path only (:data:`MAX_DECODE_INDEX_CANDIDATES`):
the segment merge returned wrong rows on trn2 in ways that depended on the graph around
the kernel and that the compiled-program ordering check does not see, so a model whose
``max_model_len`` needs a wider candidate axis is refused when the runner starts
(:func:`check_decode_index_context`). The merge stays for the work that lifts the ceiling.

Both physical cores of an LNC2 core split the requests (``[2]`` grid at ``B >= 2``).

Dispatches count in ``decode_batch``'s family (``decode_batch_dispatch_counters`` and the
fourth entry of ``decode_batch_route_counts``): this is the batched decode indexer's third
stage, after that module's ring step and scores.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, BOUND_FILL_MARK, SENTINEL
from vllm_neuron.functional.dsa.decode_batch import (
    PARTITIONS,
    _count_dispatch,
    _count_torch_fallback,
)
from vllm_neuron.functional.dsa.index_expand import (
    _dsa_index_expand_torch,
    index_expand_width,
    is_power_of_two,
)
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: ``nonzero_with_count`` reads and writes partition ``16 i`` of each 16-partition group.
GPSIMD_STRIDE = 16
#: Rows compacted by one ``nonzero_with_count``: one per GpSimd core.
COMPACT_ROWS = PARTITIONS // GPSIMD_STRIDE
#: Doublings from one partition to all of them (log2 of ``PARTITIONS``). A module
#: constant, because the NKI front end resolves names, not method calls on them.
PARTITION_DOUBLINGS = PARTITIONS.bit_length() - 1
#: Requests one fold tile carries at the most: four partitions each.
TILE_REQUESTS = 32
#: The widest chunk a tile aims for: a fold tile is ``[P, F]`` fp32 and the selection keeps
#: seven of them; fewer requests share a tile when the candidate axis would need more.
FOLD_COLUMNS = 2048
#: The widest chunk served at all, where a single request's row already folds over every
#: partition (``C > 128 * FOLD_COLUMNS``): the SBUF a partition offers (224 KiB on trn2)
#: holds seven fold tiles and the compaction tiles up to this width.
FOLD_COLUMNS_LIMIT = 4096
#: The widest row compacted whole (``[P, W]`` fp32 mask plus ``[P, W + 1]`` int32 ids). Up
#: to it one ``nonzero_with_count`` per eight requests is cheaper on the device than
#: segments and a merge (a second compaction and a gather per eight requests).
WHOLE_ROW_COLUMNS = 16384
#: The widest segment of a wider row, which is compacted in segments and merged: room for
#: the fold tiles of the widest axis (:data:`FOLD_COLUMNS_LIMIT`) beside it.
SEGMENT_COLUMNS = 8192
#: Up to this many candidates a request folds over at most :data:`SMALL_AXIS_FOLD`
#: partitions, past it over every one: the faster fold on the device. On a short axis the
#: 32 bit steps cost the same cross-partition sum whatever the fold, and a 16-partition
#: fold keeps the per-step tiles smallest; on a longer axis the per-step column work
#: dominates and every partition shortens it.
SMALL_AXIS_COLUMNS = 2048
SMALL_AXIS_FOLD = 16
#: Lists merged at once: ``MERGE_LISTS * k`` ids side by side on one partition, the first
#: list the merge of the segments before them.
MERGE_LISTS = 8
#: The widest candidate axis served: a lone request folds over 128 partitions of
#: :data:`FOLD_COLUMNS_LIMIT` columns. A physical (SBUF) bound, not a policy.
MAX_SELECT_CANDIDATES = PARTITIONS * FOLD_COLUMNS_LIMIT
#: The widest candidate axis the decode indexer serves: the whole-row path, 16384 pools
#: (a 64k context at four tokens a pool). Wider rows go through the segment merge, whose
#: device results are not yet trusted (module docstring); the runner refuses a
#: ``max_model_len`` past it (:func:`check_decode_index_context`).
MAX_DECODE_INDEX_CANDIDATES = WHOLE_ROW_COLUMNS
#: fp32 columns of a ``[P, 1]`` scratch tile: one whole 32-byte line.
_LINE = 8
#: The sign bit as an int32. Every immediate here is a power of two or ``-1``, so it is
#: exact whatever width the engine carries an immediate in.
_SIGN_BIT = -2147483648
_SUPPORTED_DTYPES = (torch.float32,)

#: This file's content digest, handed to the kernel as a trace-time int: the compiled
#: kernel cache keys on a kernel's own source and its arguments, and the kernel calls
#: helpers whose edits it would otherwise not see.
SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)


class DecodeSelectError(ValueError):
    """A malformed selection call."""


# ---------------------------------------------------------------------------------------
# Trace-time geometry, shared with the host side and the tests
# ---------------------------------------------------------------------------------------


def _ceil_div(a, b):
    """``ceil(a / b)`` for ints ``a >= 0`` and ``b > 0``, in integer arithmetic.

    Plain Python, so the host geometry helpers and the kernel bodies (trace-time ints)
    share it.
    """
    return (a + b - 1) // b


def tile_requests(width: int) -> int:
    """Requests a fold tile carries at the most for a ``width``-wide candidate axis."""
    fold_min = _ceil_div(int(width), FOLD_COLUMNS)
    return max(1, min(TILE_REQUESTS, PARTITIONS // fold_min))


def select_fold_cap(width: int) -> int:
    """The most partitions one request's ``width`` scores fold over (the kernel's
    ``fold_cap``): :data:`SMALL_AXIS_FOLD` up to :data:`SMALL_AXIS_COLUMNS` candidates,
    every partition past them."""
    return SMALL_AXIS_FOLD if int(width) <= SMALL_AXIS_COLUMNS else PARTITIONS


def fold_layout(n: int, width: int, fold_cap: int) -> tuple[int, int, bool]:
    """``(fold, chunk, request_major)`` of a tile of ``n`` requests.

    ``fold = min(128 // n, fold_cap)`` partitions per request and
    ``chunk = ceil(width / fold)`` columns per partition. The partitions run request-major
    when that takes fewer DMAs: one per request plus one per partial last chunk, against
    one per non-empty chunk.
    """
    fold = min(PARTITIONS // int(n), int(fold_cap))
    chunk = _ceil_div(int(width), fold)
    chunks_used = _ceil_div(int(width), chunk)
    partial = int(width) % chunk != 0
    per_request = n * (1 + int(partial))
    return fold, chunk, per_request < chunks_used


def segments(width: int) -> tuple[int, int]:
    """``(S, W)``: a ``width``-wide mask row compacts in ``S`` segments of ``W`` columns
    (the last one ``width - (S - 1) * W``); one segment, the whole row, up to
    :data:`WHOLE_ROW_COLUMNS`."""
    if int(width) <= WHOLE_ROW_COLUMNS:
        return 1, int(width)
    count = _ceil_div(int(width), SEGMENT_COLUMNS)
    return count, _ceil_div(int(width), count)


# ---------------------------------------------------------------------------------------
# Device helpers. Positional arguments only: the NKI front end drops keyword defaults.
# ---------------------------------------------------------------------------------------


def _sb(shape, dtype):
    """A new kernel-local SBUF tile, ``shape`` (partitions first) of ``dtype``."""
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    """A ``[parts, 1]`` view of a tile one 32-byte line wide."""
    return nl.ndarray((parts, _LINE), dtype=dtype, buffer=nl.sbuf)[:, 0:1]


def _request_matrices(n, f, request_major):
    """``(same, below)``, ``[P, P]`` fp32 with ``P = n * f``.

    ``same[p, q]`` is 1 where partitions ``p`` and ``q`` carry the same request and
    ``below[p, q]`` where, in addition, ``p < q`` (a lower chunk of ``q``'s request, in
    either layout). As a matmul's stationary operand, ``same`` sums a per-partition column
    over each request's partitions and ``below`` over the chunks below each partition.
    """
    parts = n * f
    diff = _sb((n, parts), nl.float32)
    # diff[r, q] = owner(q) - r: zero exactly where column q carries request r.
    if request_major:
        nisa.iota(dst=diff, pattern=[[1, n], [0, f]], offset=0, channel_multiplier=-1)
    else:
        nisa.iota(dst=diff, pattern=[[0, f], [1, n]], offset=0, channel_multiplier=-1)
    owner = _sb((n, parts), nl.float32)
    nisa.tensor_scalar(dst=owner, data=diff, op0=nl.equal, operand0=0.0)
    same_ps = nl.ndarray((parts, parts), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=same_ps, stationary=owner, moving=owner)
    same = _sb((parts, parts), nl.float32)
    nisa.tensor_copy(dst=same, src=same_ps)
    order = _sb((parts, parts), nl.float32)
    nisa.iota(dst=order, pattern=[[1, parts]], offset=0, channel_multiplier=-1)
    below = _sb((parts, parts), nl.float32)
    nisa.scalar_tensor_tensor(dst=below, data=order, op0=nl.greater, operand0=0.0,
                              op1=nl.multiply, operand1=same)
    return same, below


def _request_sum(matrix, column, parts):
    """``matrix^T @ column``: a per-partition count summed over request partitions."""
    out = nl.ndarray((parts, _LINE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out[:, 0:1], stationary=matrix, moving=column)
    return out[:, 0:1]


def _dma(on_tile, on_hbm, into_tile):
    """One DMA between a tile view and an HBM view, into the tile or out of it."""
    if into_tile:
        nisa.dma_copy(dst=on_tile, src=on_hbm)
    else:
        nisa.dma_copy(dst=on_hbm, src=on_tile)


def _gpsimd_dst(shape, dtype):
    """A destination tile for a GpSimd library instruction (``nonzero_with_count``,
    ``nc_n_gather``), written once here so that its life starts before the instruction's
    sources end theirs: the SBUF allocator then never places it over them. These
    instructions run on the device in pieces that re-read their sources, so an in-place
    placement corrupts them (the simulator runs them in one step and does not show it).
    """
    tile = _sb(shape, dtype)
    nisa.memset(dst=tile[0:1, 0:1], value=0)
    return tile


def _gpsimd_copy(tile, hbm, row0, m, cols, row_cols, first, into_tile):
    """Move columns ``[first, first + cols)`` of rows ``row0 .. row0 + m - 1`` of a
    ``[rows, row_cols]`` HBM tensor to or from partitions ``0, 16, .., 16 (m - 1)`` of
    ``tile``, columns ``[0, cols)``: the first partition of each of ``m`` GpSimd cores.
    Every parameter is positional: the NKI front end has no keyword-only parameters.

    One DMA per row. A stepped partition slice (``tile[0:16 m:16]``) would move the rows in
    one DMA, but the device compiler does not order a later write to the same SBUF bytes
    after such a read (a write-after-read race at ``B = 64``), and refuses the slice over
    fewer than three partitions.
    """
    for i in range(m):
        part = GPSIMD_STRIDE * i
        _dma(tile[part:part + 1, 0:cols],
             hbm.ap(pattern=[[row_cols, 1], [1, cols]], offset=(row0 + i) * row_cols + first),
             into_tile)


def _fold_copy(hbm, tile, r0, n, width, chunk, fold, request_major, into_tile):
    """Move requests ``r0 .. r0 + n - 1`` of a ``[rows, width]`` HBM tensor between HBM
    and the fold layout. The pad columns of a partial chunk, and chunks past the width,
    are left as they are.

    Chunk-major: chunk ``j`` of request ``r`` on partition ``j * n + r``, one DMA per
    chunk, the ``n`` partitions outermost. Request-major: partition ``r * fold + j``, one
    DMA per request over its whole chunks and one over its partial last chunk.
    """
    whole = width // chunk
    tail = width - whole * chunk
    if request_major:
        for i in range(n):
            base = (r0 + i) * width
            part = i * fold
            if whole > 0:
                hap = hbm.ap(pattern=[[chunk, whole], [1, chunk]], offset=base)
                _dma(tile[part:part + whole, 0:chunk], hap, into_tile)
            if tail > 0:
                hap = hbm.ap(pattern=[[tail, 1], [1, tail]], offset=base + whole * chunk)
                _dma(tile[part + whole:part + whole + 1, 0:tail], hap, into_tile)
        return
    for j in range(fold):
        first = j * chunk
        cols = min(chunk, width - first)
        if cols > 0:
            hap = hbm.ap(pattern=[[width, n], [1, cols]], offset=r0 * width + first)
            _dma(tile[j * n:(j + 1) * n, 0:cols], hap, into_tile)


def _kth_largest(scores, same, parts, chunk, select_k):
    """Each partition's request's ``select_k``-th largest score, ``[parts, 1]`` fp32.

    The MSB-first bit search the module docstring describes. Step 0 tests ``+0.0``,
    whose key is the sign bit alone; it fixes the key-to-bits map of every later trial:
    XOR with the sign bit for a non-negative result, with ``-1`` for a negative one.
    """
    junk = _sb((parts, chunk), nl.float32)
    count = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=junk, data=scores, op0=nl.greater_equal, operand0=0.0,
                              reduce_op=nl.add, reduce_res=count)
    total = _request_sum(same, count, parts)
    taken = _col(parts, nl.int32)  # -1 where the bit is taken, else 0
    nisa.tensor_scalar(dst=taken, data=total, op0=nl.greater_equal,
                       operand0=float(select_k), op1=nl.multiply, operand1=-1.0)
    key = _col(parts, nl.int32)
    nisa.tensor_scalar(dst=key, data=taken, op0=nl.bitwise_and, operand0=_SIGN_BIT)
    flip = _col(parts, nl.int32)
    nisa.tensor_scalar(dst=flip, data=taken, op0=nl.bitwise_xor, operand0=-1,
                       op1=nl.bitwise_or, operand1=_SIGN_BIT)
    for step in range(31):
        bit = 1 << (30 - step)
        trial = _col(parts, nl.int32)
        nisa.tensor_scalar(dst=trial, data=key, op0=nl.bitwise_or, operand0=bit,
                           op1=nl.bitwise_xor, operand1=flip)
        count = _col(parts, nl.float32)
        nisa.tensor_scalar_reduce(dst=junk, data=scores, op0=nl.greater_equal,
                                  operand0=trial.view(nl.float32), reduce_op=nl.add,
                                  reduce_res=count)
        total = _request_sum(same, count, parts)
        taken = _col(parts, nl.int32)
        nisa.tensor_scalar(dst=taken, data=total, op0=nl.greater_equal,
                           operand0=float(select_k), op1=nl.multiply, operand1=-1.0)
        grown = _col(parts, nl.int32)
        nisa.tensor_scalar(dst=grown, data=taken, op0=nl.bitwise_and, operand0=bit,
                           op1=nl.bitwise_or, operand1=key)
        key = grown
    bits = _col(parts, nl.int32)
    nisa.tensor_scalar(dst=bits, data=key, op0=nl.bitwise_xor, operand0=flip)
    return bits.view(nl.float32), junk


def _select_mask(scores, same, below, parts, chunk, select_k):
    """``[parts, chunk]`` fp32, 1.0 where a candidate is selected, else 0.0."""
    kth, chosen = _kth_largest(scores, same, parts, chunk, select_k)
    # Above the cut, and never a masked pool: the larger of v and the mark.
    cut = _col(parts, nl.float32)
    nisa.tensor_scalar(dst=cut, data=kth, op0=nl.maximum, operand0=BOUND_FILL_MARK)
    above = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=chosen, data=scores, op0=nl.greater, operand0=cut,
                              reduce_op=nl.add, reduce_res=above)
    above_total = _request_sum(same, above, parts)
    room = _col(parts, nl.float32)  # k - #{> v}: the ties the request still takes
    nisa.tensor_scalar(dst=room, data=above_total, op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=float(select_k))
    real = _col(parts, nl.float32)  # 0 when v is a masked pool: no tie is taken then
    nisa.tensor_scalar(dst=real, data=kth, op0=nl.greater, operand0=BOUND_FILL_MARK)
    ties = _sb((parts, chunk), nl.float32)
    tie_count = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=ties, data=scores, op0=nl.equal, operand0=kth,
                              reduce_op=nl.add, reduce_res=tie_count)
    ties_below = _request_sum(below, tie_count, parts)
    quota = _col(parts, nl.float32)
    nisa.tensor_tensor(dst=quota, data1=room, data2=ties_below, op=nl.subtract)
    nisa.tensor_tensor(dst=quota, data1=quota, data2=real, op=nl.multiply)
    ones = _sb((parts, chunk), nl.float32)
    nisa.memset(dst=ones, value=1.0)
    rank = _sb((parts, chunk), nl.float32)  # 1-based rank of a tie within its chunk
    nisa.tensor_tensor_scan(dst=rank, data0=ones, data1=ties, initial=0.0,
                            op0=nl.multiply, op1=nl.add)
    picked = _sb((parts, chunk), nl.float32)
    nisa.scalar_tensor_tensor(dst=picked, data=rank, op0=nl.less_equal, operand0=quota,
                              op1=nl.multiply, operand1=ties)
    mask = _sb((parts, chunk), nl.float32)
    nisa.tensor_tensor(dst=mask, data1=picked, data2=chosen, op=nl.add)
    return mask


def _compact_whole_rows(mask_hbm, ids_hbm, r0, n, width, select_k):
    """Each request's whole mask row compacted on one GpSimd partition; its ids to row
    ``r`` of ``ids_hbm``."""
    n_groups = _ceil_div(n, COMPACT_ROWS)
    for g in range(n_groups):
        m = min(COMPACT_ROWS, n - g * COMPACT_ROWS)
        row0 = r0 + g * COMPACT_ROWS
        found = _gpsimd_dst((PARTITIONS, width + 1), nl.int32)
        rows = _sb((PARTITIONS, width), nl.float32)
        _gpsimd_copy(rows, mask_hbm, row0, m, width, width, 0, True)
        nisa.nonzero_with_count(dst=found, src=rows, index_offset=0, padding_val=SENTINEL)
        _gpsimd_copy(found, ids_hbm, row0, m, select_k, select_k, 0, False)


def _merge_slots(slots, span, select_k):
    """The first ``k`` ids of ``slots`` in order: ``[P, k]`` int32, ``-1`` past the ids,
    rows ``[0, span)`` written.

    ``slots`` is ``[P, MERGE_LISTS * k + 1]`` int32: on partition ``16 i`` up to
    :data:`MERGE_LISTS` lists side by side, each its ids (``>= 0``) first and a negative
    pad after them, and a last column of ``-1``. ``nonzero_with_count`` over "is an id"
    gives the positions of the ids in order and ``nc_n_gather`` reads them.
    """
    wide = MERGE_LISTS * select_k
    out = _gpsimd_dst((PARTITIONS, select_k), nl.int32)
    valid = _sb((PARTITIONS, wide), nl.float32)
    nisa.tensor_scalar(dst=valid, data=slots[:, 0:wide], op0=nl.greater_equal, operand0=0.0)
    # Positions of the ids in order; past them, the always-(-1) last column.
    # nonzero_with_count writes partitions 16 i only and a compute instruction may not
    # start on one of them, so one gather spans them all, the partitions between
    # gathering column 0.
    found = _gpsimd_dst((PARTITIONS, wide + 1), nl.int32)
    nisa.memset(dst=found[0:span, 0:select_k], value=0)
    nisa.nonzero_with_count(dst=found, src=valid, index_offset=0, padding_val=wide)
    nisa.nc_n_gather(dst=out[0:span, 0:select_k], data=slots[0:span, 0:wide + 1],
                     indices=found[0:span, 0:select_k].view(nl.uint32))
    return out


def _compact_segments(mask_hbm, ids_hbm, zeros_hbm, r0, n, width, select_k):
    """Each request's mask row in ``S`` segments, compacted and merged on the request's
    own GpSimd partition; its ids to row ``r`` of ``ids_hbm``.

    One ``nonzero_with_count`` compacts segment ``s`` of eight requests, so its
    ``index_offset`` (the segment's first candidate) turns positions into candidate ids;
    its first ``k`` columns go to the next free slot of the requests' slot tile
    (:func:`_merge_slots`) by a copy on the same partitions. A full tile merges into its
    first slot and the next segment takes the second. The lists never go through HBM:
    written to an HBM scratch and read back for the merge, a request's lists past the
    first came back as zeros on the device (``B = 16``, 32768 candidates; the simulator
    was exact). A short last segment's pad columns are zeroed
    by a DMA from ``zeros_hbm``: a compute instruction may not start on partition
    ``16 i``.
    """
    count, seg = segments(width)
    wide = MERGE_LISTS * select_k
    for g in range(_ceil_div(n, COMPACT_ROWS)):
        m = min(COMPACT_ROWS, n - g * COMPACT_ROWS)
        row0 = r0 + g * COMPACT_ROWS
        # Partitions 0 .. 16 (m - 1): every compute instruction here starts on partition 0.
        span = GPSIMD_STRIDE * (m - 1) + 1
        slots = _sb((PARTITIONS, wide + 1), nl.int32)
        nisa.memset(dst=slots, value=-1)
        used = 0
        for s in range(count):
            if used == MERGE_LISTS:
                merged = _merge_slots(slots, span, select_k)
                slots = _sb((PARTITIONS, wide + 1), nl.int32)
                nisa.memset(dst=slots, value=-1)
                nisa.tensor_copy(dst=slots[0:span, 0:select_k],
                                 src=merged[0:span, 0:select_k])
                used = 1
            first = s * seg
            cols = min(seg, width - first)
            found = _gpsimd_dst((PARTITIONS, seg + 1), nl.int32)
            rows = _sb((PARTITIONS, seg), nl.float32)
            _gpsimd_copy(rows, mask_hbm, row0, m, cols, width, first, True)
            for i in range(m if cols < seg else 0):
                part = GPSIMD_STRIDE * i
                nisa.dma_copy(dst=rows[part:part + 1, cols:seg],
                              src=zeros_hbm.ap(pattern=[[seg - cols, 1], [1, seg - cols]],
                                               offset=0))
            nisa.nonzero_with_count(dst=found, src=rows, index_offset=first,
                                    padding_val=SENTINEL)
            nisa.tensor_copy(dst=slots[0:span, used * select_k:(used + 1) * select_k],
                             src=found[0:span, 0:select_k])
            used += 1
        _gpsimd_copy(_merge_slots(slots, span, select_k), ids_hbm, row0, m, select_k, select_k,
                     0, False)


def _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, ids_hbm, zeros_hbm, tok_hbm,
                 r0, n, select_k, pool_size, out_cols, fold_cap):
    """Requests ``r0 .. r0 + n - 1``: select, compact, expand."""
    width = bounded_hbm.shape[1]
    fold, chunk, request_major = fold_layout(n, width, fold_cap)
    parts = fold * n

    # ---- the scores in the fold layout, pads at BOUND_FILL ------------------------------
    scores = _sb((parts, chunk), nl.float32)
    if width < fold * chunk:
        nisa.memset(dst=scores, value=BOUND_FILL)
    _fold_copy(bounded_hbm, scores, r0, n, width, chunk, fold, request_major, True)
    same, below = _request_matrices(n, fold, request_major)

    # ---- the selection, out to the scratch row by row -----------------------------------
    mask = _select_mask(scores, same, below, parts, chunk, select_k)
    _fold_copy(mask_hbm, mask, r0, n, width, chunk, fold, request_major, False)

    # ---- compaction: rows on the GpSimd cores' first partitions -------------------------
    count, _seg = segments(width)
    if count == 1:
        _compact_whole_rows(mask_hbm, ids_hbm, r0, n, width, select_k)
    else:
        _compact_segments(mask_hbm, ids_hbm, zeros_hbm, r0, n, width, select_k)

    # ---- expansion: pool p -> tokens p * pool + t; -1 stays -1 --------------------------
    spread = 1
    for _ in range(PARTITION_DOUBLINGS):
        if n * spread * 2 <= PARTITIONS and select_k % (spread * 2) == 0:
            spread = spread * 2
    lanes = n * spread
    per = select_k // spread
    ids = _sb((lanes, per), nl.int32)
    nisa.dma_copy(dst=ids, src=ids_hbm.ap(pattern=[[per, lanes], [1, per]],
                                          offset=r0 * select_k))
    first = _sb((lanes, per), nl.int32)
    nisa.tensor_scalar(dst=first, data=ids, op0=nl.multiply, operand0=pool_size)
    offset = _sb((lanes, per, pool_size), nl.int32)
    nisa.iota(dst=offset, pattern=[[0, per], [1, pool_size]], offset=0, channel_multiplier=0)
    tokens = _sb((lanes, per, pool_size), nl.int32)
    nisa.tensor_tensor(dst=tokens,
                       data1=first.ap(pattern=[[per, lanes], [1, per], [0, pool_size]]),
                       data2=offset, op=nl.add)
    # A -1 id gives -pool .. -1 here; the max takes all of them to -1.
    nisa.tensor_scalar(dst=tokens, data=tokens, op0=nl.maximum, operand0=-1)
    # Lane r * spread + q is the contiguous run q of request r's history, so the lanes
    # are one contiguous block of the scratch, which the output rows then copy whole.
    history = select_k * pool_size
    nisa.dma_copy(dst=tok_hbm.ap(pattern=[[per * pool_size, lanes], [1, per * pool_size]],
                                 offset=r0 * history),
                  src=tokens.reshape((lanes, per * pool_size)))
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[out_cols, n], [1, history]], offset=r0 * out_cols),
                  src=tok_hbm.ap(pattern=[[history, n], [1, history]], offset=r0 * history))

    # ---- the tail: tokens of the incomplete final pool, then -1 to the width ------------
    tail_w = out_cols - select_k * pool_size
    lens = _col(n, nl.int32)
    nisa.dma_copy(dst=lens, src=lens_hbm.ap(pattern=[[1, n], [1, 1]], offset=r0))
    open_i = _col(n, nl.int32)
    nisa.tensor_scalar(dst=open_i, data=lens, op0=nl.bitwise_and, operand0=pool_size - 1)
    open_f = _col(n, nl.float32)
    nisa.tensor_copy(dst=open_f, src=open_i)
    start = _col(n, nl.int32)
    nisa.tensor_scalar(dst=start, data=lens, op0=nl.bitwise_and, operand0=-pool_size)
    # An arithmetic op takes an fp32 per-partition operand (the device refuses int32);
    # a token index is far below 2**24, so the widening is exact.
    start_f = _col(n, nl.float32)
    nisa.tensor_copy(dst=start_f, src=start)
    lane = _sb((n, tail_w), nl.int32)
    nisa.iota(dst=lane, pattern=[[1, tail_w]], offset=0, channel_multiplier=0)
    hit = _sb((n, tail_w), nl.float32)
    nisa.tensor_scalar(dst=hit, data=lane, op0=nl.less, operand0=open_f)
    tail = _sb((n, tail_w), nl.int32)  # start + t + 1 where t is open, else 0; then -1
    nisa.tensor_scalar(dst=tail, data=lane, op0=nl.add, operand0=start_f, op1=nl.add,
                       operand1=1)
    nisa.tensor_tensor(dst=tail, data1=tail, data2=hit, op=nl.multiply)
    nisa.tensor_scalar(dst=tail, data=tail, op0=nl.add, operand0=-1)
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[out_cols, n], [1, tail_w]],
                                 offset=r0 * out_cols + select_k * pool_size),
                  src=tail)


# ---------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------


@nki.jit
def dsa_decode_select_kernel(bounded_hbm, lens_hbm, select_k, pool_size, out_cols,
                             fold_cap, source_digest):
    """Selected token indices for ``B`` decode requests.

    Args:
        bounded_hbm: ``[B, C]`` fp32, each request's bounded candidate scores.
        lens_hbm: ``[B, 1]`` int32, each request's length (this step's token included).
        select_k: python int, pools selected per request, ``< C``.
        pool_size: python int, a power of two.
        out_cols: python int, ``index_expand_width(select_k, pool_size)``.
        fold_cap: python int, the most partitions a request folds over
            (:func:`select_fold_cap`).
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``[B, out_cols]`` int32, the module docstring's layout.

    The requests split evenly over the programs of the launch grid, each program's share
    in tiles of up to :func:`tile_requests` requests.
    """
    batch = bounded_hbm.shape[0]
    width = bounded_hbm.shape[1]
    out_hbm = nl.ndarray((batch, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)
    mask_hbm = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.private_hbm)
    ids_hbm = nl.ndarray((batch, select_k), dtype=nl.int32, buffer=nl.private_hbm)
    count, _seg = segments(width)
    tok_hbm = nl.ndarray((batch, select_k * pool_size), dtype=nl.int32,
                         buffer=nl.private_hbm)
    # Zeros for a short last segment's pad columns (one row of SEGMENT_COLUMNS).
    zeros_hbm = nl.ndarray((1, SEGMENT_COLUMNS), dtype=nl.float32, buffer=nl.private_hbm)
    if count > 1 and width % count != 0:
        zero_cols = SEGMENT_COLUMNS // PARTITIONS
        zero_sb = _sb((PARTITIONS, zero_cols), nl.float32)
        nisa.memset(dst=zero_sb, value=0.0)
        nisa.dma_copy(dst=zeros_hbm.ap(pattern=[[zero_cols, PARTITIONS], [1, zero_cols]],
                                       offset=0), src=zero_sb)
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    share = (batch + n_prgs - 1) // n_prgs
    lo = prg * share
    hi = min(batch, lo + share)
    per_tile = tile_requests(width)
    n_tiles = (hi - lo + per_tile - 1) // per_tile
    for t in range(n_tiles):
        r0 = lo + t * per_tile
        n = min(per_tile, hi - r0)
        _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, ids_hbm, zeros_hbm, tok_hbm,
                     r0, n, select_k, pool_size, out_cols, fold_cap)
    return out_hbm


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def decode_select_programs(batch: int) -> int:
    """Programs the selection launches: both cores of an LNC2 core from two requests on."""
    if lnc_pair() and int(batch) >= 2:
        return 2
    return 1


def check_decode_index_context(max_model_len: int, pool_size: int | None) -> None:
    """Refuse a ``max_model_len`` whose decode candidate axis is past the served width.

    Args:
        max_model_len: the longest context the runner serves.
        pool_size: the model's tokens per pool (``index_kpool``); ``None`` for a model
            with no DSA indexer, which is not checked.

    Raises:
        DecodeSelectError: ``max_model_len // pool_size`` (the candidate axis of the
            ``max_model_len`` decode graph) is above :data:`MAX_DECODE_INDEX_CANDIDATES`.
    """
    if pool_size is None:
        return
    candidates = int(max_model_len) // int(pool_size)
    if candidates > MAX_DECODE_INDEX_CANDIDATES:
        longest = (MAX_DECODE_INDEX_CANDIDATES + 1) * int(pool_size) - 1
        raise DecodeSelectError(
            f"max_model_len={max_model_len} needs {candidates} decode index candidates "
            f"(index_kpool={pool_size}); the DSA decode indexer serves up to "
            f"MAX_DECODE_INDEX_CANDIDATES={MAX_DECODE_INDEX_CANDIDATES} (max_model_len <= "
            f"{longest}). Past that width the selection merges row segments, which returned "
            f"wrong rows on trn2 in ways that depend on the surrounding graph (an open "
            f"defect, indexer report section 4.3). Lower max_model_len.")


def _validate(bounded: Tensor, seq_lens: Tensor, select_k, pool_size) -> tuple[int, int]:
    """Check the operands of :func:`dsa_decode_select`; ``(B, C)`` of ``bounded``.

    Args:
        bounded: ``[B, C]`` candidate scores, ``B >= 1``, ``C <= MAX_SELECT_CANDIDATES``.
        seq_lens: ``[B]`` request lengths.
        select_k: a Python int, ``0 < select_k < C``.
        pool_size: a Python int, a power of two.

    Raises:
        DecodeSelectError: naming the first operand outside that contract.
    """
    if bounded.ndim != 2:
        raise DecodeSelectError(f"bounded must be [B, C]; got {tuple(bounded.shape)}")
    batch, width = int(bounded.shape[0]), int(bounded.shape[1])
    if batch < 1:
        raise DecodeSelectError("bounded must carry at least one request")
    if not isinstance(select_k, int) or not 0 < select_k < width:
        raise DecodeSelectError(
            f"select_k must be a python int in (0, C) = (0, {width}); at C <= k there is "
            f"nothing to select and the caller bypasses; got {select_k!r}")
    if not isinstance(pool_size, int) or not is_power_of_two(pool_size):
        raise DecodeSelectError(f"pool_size must be a power-of-two python int; got "
                                f"{pool_size!r}")
    if width > MAX_SELECT_CANDIDATES:
        raise DecodeSelectError(
            f"a {width}-wide candidate axis folds into more than {FOLD_COLUMNS_LIMIT} "
            f"columns per partition, past what one partition's SBUF holds; the widest "
            f"served is {MAX_SELECT_CANDIDATES} pools")
    if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != batch:
        raise DecodeSelectError(f"seq_lens must be [B] = [{batch}]; got "
                                f"{tuple(seq_lens.shape)}")
    return batch, width


def can_run_dsa_decode_select(bounded: Tensor, select_k: int, pool_size: int) -> bool:
    """Whether the kernel serves this call; ``False`` sends it to the torch oracle."""
    if not can_run_kernel(bounded):
        return False
    if bounded.dtype not in _SUPPORTED_DTYPES:
        return False
    return select_k <= int(bounded.shape[1])


def dsa_decode_select(bounded: Tensor, seq_lens: Tensor, *, select_k: int,
                      pool_size: int) -> Tensor:
    """Each request's selected token indices. ``[B, index_expand_width(k, pool)]`` int32.

    Args:
        bounded: ``[B, C]`` fp32, the bounded candidate scores (``dsa_decode_scores``).
        seq_lens: ``[B]`` int, each request's length with this step's token.
        select_k: pools selected per request, ``0 < select_k < C``.
        pool_size: tokens per pool, a power of two.

    Returns:
        The selected pools' tokens in ascending order, ``-1`` after them, then the tail
        tokens at columns ``[k * pool, k * pool + pool - 1)`` and ``-1`` to the width.
        The token set is the four-kernel route's (its rotational top-k, the causal
        sentinel and the index expand) but for ties at the ``k``-th value, where the
        lowest candidate index wins.
    """
    batch, width = _validate(bounded, seq_lens, select_k, pool_size)
    out_cols = int(index_expand_width(select_k, pool_size))
    if not can_run_dsa_decode_select(bounded, select_k, pool_size):
        _count_torch_fallback()
        return dsa_decode_select_torch_oracle(bounded, seq_lens, select_k=select_k,
                                              pool_size=pool_size)
    programs = decode_select_programs(batch)
    _count_dispatch("select", programs, batch)
    call = wrap_nki(dsa_decode_select_kernel)
    if programs == 2:
        call = call[2]
    return call(bounded.contiguous(), seq_lens.reshape(batch, 1).to(torch.int32).contiguous(),
                int(select_k), int(pool_size), out_cols, int(select_fold_cap(width)),
                SOURCE_DIGEST)


# ---------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------


def dsa_decode_select_torch_oracle(bounded: Tensor, seq_lens: Tensor, *, select_k: int,
                                   pool_size: int) -> Tensor:
    """CPU reference and fallback: a stable descending sort spells the tie rule.

    A stable sort keeps equal values in index order, so the first ``select_k`` of it are
    the ``select_k`` largest with ties to the lowest index. The masked pools among them
    become ``-1`` (the ``dsa_causal_sentinel`` rule), the real ones go to ascending
    order, and ``dsa_index_expand``'s torch reference lays out the tokens.
    """
    width = int(bounded.shape[1])
    top = torch.sort(bounded.to(torch.float32), dim=1, descending=True, stable=True)
    values, ids = top.values[:, :select_k], top.indices[:, :select_k]
    keep = values > BOUND_FILL_MARK
    ids = torch.where(keep, ids, torch.full_like(ids, width)).sort(dim=1).values
    pool_ids = torch.where(ids < width, ids, torch.full_like(ids, SENTINEL)).to(torch.int32)
    return _dsa_index_expand_torch(pool_ids, seq_lens.to(torch.int32), pool_size)
