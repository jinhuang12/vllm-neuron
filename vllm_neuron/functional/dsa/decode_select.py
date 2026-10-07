# SPDX-License-Identifier: Apache-2.0
"""The DSA decode selection for ``B`` requests in one kernel.

``Glm5NextDSAIndexer.forward_requests`` turns each request's bounded candidate scores
(``[B, C]`` fp32, ``BOUND_FILL`` where a pool is not complete for the request) into the
token indices the sparse attention reads. 0a08ff4 did that with four kernels: the
rotational top-k, the causal sentinel, the sentinel order and the index expand. Here it is
one kernel, :func:`dsa_decode_select`, which needs no sort:

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

The selected set is 0a08ff4's: the same ``k`` largest scores and the same rule for the
masked pools. Two things differ, and neither is in the contract the attention reads:

* The order of the real tokens: ascending pool order, where 0a08ff4 had descending score
  order. ``mla_decode`` masks ``-1`` wherever it is and sums over the selected rows.
* The tie rule. 0a08ff4's selector pinned nothing about equal values; this kernel pins the
  lowest index. The sets can differ only where values tie at the ``k``-th place.

Values compare as fp32 values: ``-0.0 == +0.0``. Scores must be finite (or the finite
``BOUND_FILL``): the bit search tests NaN keys on the way down to ``-inf``, so a ``k``-th
value of ``-inf`` is not found. The score kernel produces finite values.

Layout. A request's ``C`` scores fold over ``f = min(128 // n, MAX_FOLD)`` partitions
(``n`` requests a tile, at most :data:`TILE_REQUESTS`): partition ``j * n + r`` holds
candidates ``[j * F, (j + 1) * F)`` of request ``r``, ``F = ceil(C / f)``, the rest padded
with ``BOUND_FILL``. Every DMA is two-dimensional with the partitions outermost, one per
chunk ``j``, which is the form the device compiler aligns (a three-dimensional HBM access
pattern onto a partition range is refused there, though the simulator takes it). A count
is then one vector instruction over ``F`` columns and a cross-partition sum per request,
one matmul by the ``[P, P]`` same-request matrix; the tie
rank adds the lower chunks' ties by the strictly-lower same-request matrix. The
compaction needs each request's mask on one partition of a GpSimd core (partitions
``0, 16, ..., 112``), so the mask goes through an HBM scratch on its way there.

Both physical cores of an LNC2 core split the requests (``[2]`` grid at ``B >= 2``).

Dispatches count in ``decode_batch``'s family (``decode_batch_dispatch_counters`` and the
fourth entry of ``decode_batch_route_counts``): this is the batched decode indexer's third
stage, after that module's ring step and scores.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, BOUND_FILL_MARK, SENTINEL
from vllm_neuron.functional.dsa.decode_batch import _count_dispatch, _count_torch_fallback
from vllm_neuron.functional.dsa.index_expand import (
    _dsa_index_expand_torch,
    index_expand_width,
    is_power_of_two,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

PARTITIONS = 128
#: ``nonzero_with_count`` reads and writes partition ``16 i`` of each 16-partition group.
GPSIMD_STRIDE = 16
#: Requests compacted by one ``nonzero_with_count``: one per GpSimd core.
COMPACT_ROWS = PARTITIONS // GPSIMD_STRIDE
#: Requests one fold tile carries: four partitions each at the most.
TILE_REQUESTS = 32
#: Chunks a request folds into at the most: one load and one store DMA per chunk.
MAX_FOLD = 16
#: The widest candidate axis served, the score kernel's own bound
#: (``decode_batch.MAX_CANDIDATES``).
MAX_SELECT_CANDIDATES = PARTITIONS * PARTITIONS
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
# Device helpers. Positional arguments only: the NKI front end drops keyword defaults.
# ---------------------------------------------------------------------------------------


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    """A ``[parts, 1]`` view of a tile one 32-byte line wide."""
    return nl.ndarray((parts, _LINE), dtype=dtype, buffer=nl.sbuf)[:, 0:1]


def _request_matrices(n, f):
    """``(same, below)``, ``[P, P]`` fp32 with ``P = n * f``, partition ``j * n + r``.

    ``same[p, q]`` is 1 where partitions ``p`` and ``q`` carry the same request and
    ``below[p, q]`` where, in addition, ``p < q`` (a lower chunk of ``q``'s request). As a
    matmul's stationary operand, ``same`` sums a per-partition column over each request's
    partitions and ``below`` over the chunks below each partition.
    """
    parts = n * f
    diff = _sb((n, parts), nl.float32)
    # diff[r, j * n + r'] = r' - r: zero exactly where the column carries request r.
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


def _gpsimd_copy(tile, hbm, row0, m, cols, into_tile):
    """Move rows ``row0 .. row0 + m - 1`` of a ``[rows, cols]`` HBM tensor to or from
    partitions ``0, 16, .., 16 (m - 1)`` of ``tile``, columns ``[0, cols)``: the first
    partition of each of ``m`` GpSimd cores.

    All eight rows move in one DMA with a stepped partition slice. The device compiler
    refuses that slice over fewer partitions (one and two were refused: "illegal partition
    step"; the simulator takes them), so fewer rows move one DMA each.
    """
    if m == COMPACT_ROWS:
        _dma(tile[0:GPSIMD_STRIDE * COMPACT_ROWS:GPSIMD_STRIDE, 0:cols],
             hbm.ap(pattern=[[cols, m], [1, cols]], offset=row0 * cols), into_tile)
    else:
        for i in range(m):
            part = GPSIMD_STRIDE * i
            _dma(tile[part:part + 1, 0:cols],
                 hbm.ap(pattern=[[cols, 1], [1, cols]], offset=(row0 + i) * cols), into_tile)


def _fold_copy(hbm, tile, r0, n, width, chunk, fold, into_tile):
    """Move requests ``r0 .. r0 + n - 1`` of a ``[rows, width]`` HBM tensor between HBM
    and the fold layout: chunk ``j`` of request ``r`` on partition ``j * n + r``.

    One two-dimensional DMA per chunk, the ``n`` partitions outermost; a chunk past the
    width moves nothing, and the pad columns of a partial chunk are left as they are.
    """
    for j in range(fold):
        first = j * chunk
        cols = min(chunk, width - first)
        if cols > 0:
            hap = hbm.ap(pattern=[[width, n], [1, cols]], offset=r0 * width + first)
            if into_tile:
                nisa.dma_copy(dst=tile[j * n:(j + 1) * n, 0:cols], src=hap)
            else:
                nisa.dma_copy(dst=hap, src=tile[j * n:(j + 1) * n, 0:cols])


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


def _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, ids_hbm, tok_hbm, r0, n,
                 select_k, pool_size, out_cols):
    """Requests ``r0 .. r0 + n - 1``, ``n <= TILE_REQUESTS``: select, compact, expand."""
    width = bounded_hbm.shape[1]
    fold = min(PARTITIONS // n, MAX_FOLD)
    parts = fold * n
    chunk = (width + fold - 1) // fold

    # ---- the scores in the fold layout, pads at BOUND_FILL ------------------------------
    scores = _sb((parts, chunk), nl.float32)
    if width < fold * chunk:
        nisa.memset(dst=scores, value=BOUND_FILL)
    _fold_copy(bounded_hbm, scores, r0, n, width, chunk, fold, True)
    same, below = _request_matrices(n, fold)

    # ---- the selection, out to the scratch row by row -----------------------------------
    mask = _select_mask(scores, same, below, parts, chunk, select_k)
    _fold_copy(mask_hbm, mask, r0, n, width, chunk, fold, False)

    # ---- compaction: one request on each GpSimd core's first partition ------------------
    n_groups = (n + COMPACT_ROWS - 1) // COMPACT_ROWS
    for g in range(n_groups):
        m = min(COMPACT_ROWS, n - g * COMPACT_ROWS)
        row0 = r0 + g * COMPACT_ROWS
        rows = _sb((PARTITIONS, width), nl.float32)
        _gpsimd_copy(rows, mask_hbm, row0, m, width, True)
        found = _sb((PARTITIONS, width + 1), nl.int32)
        nisa.nonzero_with_count(dst=found, src=rows, index_offset=0, padding_val=SENTINEL)
        _gpsimd_copy(found, ids_hbm, row0, m, select_k, False)

    # ---- expansion: pool p -> tokens p * pool + t; -1 stays -1 --------------------------
    spread = 1
    for _ in range(7):
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
                             source_digest):
    """Selected token indices for ``B`` decode requests.

    Args:
        bounded_hbm: ``[B, C]`` fp32, each request's bounded candidate scores.
        lens_hbm: ``[B, 1]`` int32, each request's length (this step's token included).
        select_k: python int, pools selected per request, ``< C``.
        pool_size: python int, a power of two.
        out_cols: python int, ``index_expand_width(select_k, pool_size)``.
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``[B, out_cols]`` int32, the module docstring's layout.

    The requests split evenly over the programs of the launch grid, each program's share
    in tiles of up to :data:`TILE_REQUESTS`.
    """
    batch = bounded_hbm.shape[0]
    width = bounded_hbm.shape[1]
    out_hbm = nl.ndarray((batch, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)
    mask_hbm = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.private_hbm)
    ids_hbm = nl.ndarray((batch, select_k), dtype=nl.int32, buffer=nl.private_hbm)
    tok_hbm = nl.ndarray((batch, select_k * pool_size), dtype=nl.int32,
                         buffer=nl.private_hbm)
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    share = (batch + n_prgs - 1) // n_prgs
    lo = prg * share
    hi = min(batch, lo + share)
    n_tiles = (hi - lo + TILE_REQUESTS - 1) // TILE_REQUESTS
    for t in range(n_tiles):
        r0 = lo + t * TILE_REQUESTS
        n = min(TILE_REQUESTS, hi - r0)
        _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, ids_hbm, tok_hbm, r0, n,
                     select_k, pool_size, out_cols)
    return out_hbm


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def decode_select_programs(batch: int) -> int:
    """Programs the selection launches: both cores of an LNC2 core from two requests on."""
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and int(batch) >= 2:
        return 2
    return 1


def _validate(bounded: Tensor, seq_lens: Tensor, select_k, pool_size) -> tuple[int, int]:
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
    width = int(bounded.shape[1])
    return width <= MAX_SELECT_CANDIDATES and select_k <= width


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
        The token set is 0a08ff4's ``expand_indices(_select_bounded(bounded), seq_lens)``
        but for ties at the ``k``-th value, where the lowest candidate index wins.
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
                int(select_k), int(pool_size), out_cols, SOURCE_DIGEST)


# ---------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------


def dsa_decode_select_torch_oracle(bounded: Tensor, seq_lens: Tensor, *, select_k: int,
                                   pool_size: int) -> Tensor:
    """CPU reference and fallback: a stable descending sort spells the tie rule.

    A stable sort keeps equal values in index order, so the first ``select_k`` of it are
    the ``select_k`` largest with ties to the lowest index. The masked pools among them
    become ``-1`` (0a08ff4's ``dsa_causal_sentinel`` rule), the real ones go to ascending
    order, and ``dsa_index_expand``'s torch reference lays out the tokens.
    """
    width = int(bounded.shape[1])
    top = torch.sort(bounded.to(torch.float32), dim=1, descending=True, stable=True)
    values, ids = top.values[:, :select_k], top.indices[:, :select_k]
    keep = values > BOUND_FILL_MARK
    ids = torch.where(keep, ids, torch.full_like(ids, width)).sort(dim=1).values
    pool_ids = torch.where(ids < width, ids, torch.full_like(ids, SENTINEL)).to(torch.int32)
    return _dsa_index_expand_torch(pool_ids, seq_lens.to(torch.int32), pool_size)
