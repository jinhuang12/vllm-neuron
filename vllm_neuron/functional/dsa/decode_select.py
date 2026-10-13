# SPDX-License-Identifier: Apache-2.0
"""The DSA decode selection for ``B`` requests in one kernel.

``Glm5NextDSAIndexer.forward_requests`` turns each request's bounded candidate scores
(``[B, C]`` fp32, ``BOUND_FILL`` where a pool is not complete for the request) into the
token indices the sparse attention reads. The earlier route took four kernels for it
(the rotational top-k, the causal sentinel, the sentinel order and the index expand).
Here it is one kernel, :func:`dsa_decode_select`, which needs no sort:

1. **The ``k``-th largest score, by its bits.** Request ``b``'s threshold is found on the
   order-preserving integer key of an fp32 value (sign bit flipped for a positive value,
   every bit flipped for a negative one), most significant bits first. The sign bit is set
   when at least ``k`` scores are ``>= +0.0``. Below it, a chunk of up to
   :data:`RADIX_COLUMNS` columns (see Layout) takes the 31 bits as seven four-bit digits
   and a three-bit one: a digit is the number of its nonzero values ``t`` for which at
   least ``k`` scores are ``>=`` the value whose key is the bits found so far with the
   digit at ``t``, its trials counted together. A longer chunk takes the bits one at a
   time: bit ``i`` is set when at least ``k`` scores are ``>=`` the value whose key is the
   bits found so far with bit ``i`` set. Each trial is one count over the request's
   scores, compared as fp32 values, so the result is the exact ``k``-th largest value
   ``v`` either way, and the two searches select the same candidates. A step ends in a
   cross-partition sum whatever its trials, so a short chunk, whose counts cost little,
   takes 8 digit steps rather than 31 bit steps. The second digit ends at bit 23, the
   exponent's lowest: the keys below ``-FLT_MAX``'s are negative NaNs, which no score
   reaches, so a digit with a trial among them would count short of its value; with that
   cut, the first two digits' trials are finite, and every later trial carries a nonzero
   bit above 23 when ``v`` is negative and finite.
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
onto a partition range is refused there, though the simulator takes it). A bit step's
count is then one vector instruction over ``F`` columns and a cross-partition sum per
request, one matmul by the ``[P, P]`` same-request matrix; a digit step counts its trials
in two vector instructions over ``15 F`` columns and sums them in one matmul. The tie
rank adds the lower chunks' ties by the strictly-lower same-request matrix. None of it
depends on the layout.

The compaction needs a row on one partition of a GpSimd core (partitions
``0, 16, ..., 112``), so the mask goes through an HBM scratch on its way there. A row of
up to :data:`WHOLE_ROW_COLUMNS` candidates is compacted whole. A wider row is walked in
``S = ceil(C / WHOLE_ROW_COLUMNS)`` chunks of that width (:func:`row_chunks`), its mask
padded to ``S`` whole chunks (:func:`mask_span`; the pad scores are ``BOUND_FILL``, never
selected). One ``nonzero_with_count`` compacts eight (request, chunk) rows, so the chunks
of one request run on ``S`` GpSimd cores at once. A chunk's list needs only ``k`` columns:
the whole row selects at most ``k``. The lists are joined without a gather. Chunk ``c``'s
list goes to an HBM row behind ``k`` pad columns, and one DMA reads ``k`` columns of that
row from column ``k - p``, where ``p`` is the number of ids in the request's lower chunks
(an offset the DMA reads from SBUF when it runs). That window holds the chunk's ids at
their places in the joined list and pads at every other place, so the ``S`` windows, each
id shifted by its chunk's first candidate, join by an element-wise maximum (a pad is below
every id). Every shape is static and no control flow depends on a value: only the window
offsets do.

The decode indexer serves candidate axes up to :data:`MAX_DECODE_INDEX_CANDIDATES` and
refuses a longer ``max_model_len`` when the runner starts
(:func:`check_decode_index_context`). A runner that batches more than one decode request
is served up to :data:`MAX_BATCHED_DECODE_INDEX_CANDIDATES`, one whole row. Decode context
parallelism selects with :func:`dsa_decode_select_pools`, on the whole-row path only, up to
:data:`MAX_DECODE_POOLS_CANDIDATES` per rank (:func:`check_decode_pools_context`).

Both physical cores of an LNC2 core split the requests (``[2]`` grid at ``B >= 2``).

Decode context parallelism (DCP): each rank of a group selects over its own pools, and the
group merges the ranks' lists (trn2-2's item 3). :func:`dsa_decode_select_pools` is this
selection with, beside the tokens, the local top-``k`` before the expansion: the selected
pool ids in ascending order and their scores, the pads (:data:`POOL_PAD_ID`,
:data:`POOL_PAD_VALUE`) after them. The ids are local; the DCP ownership map
(``vllm_neuron.utils.dcp_ownership``) maps them to the global pool axis and the merged ids
back to the rank's own bank rows.

Dispatches count in ``decode_batch``'s family (``decode_batch_dispatch_counters`` and the
fourth entry of ``decode_batch_route_counts``): this is the batched decode indexer's third
stage, after that module's ring step and scores.
"""

from __future__ import annotations

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
from vllm_neuron.source_digest import source_digest
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
#: The widest row one ``nonzero_with_count`` compacts (``[P, W]`` fp32 mask plus
#: ``[P, W + 1]`` int32 ids, about half of a partition's SBUF). A wider row is compacted in
#: chunks of this width and the chunk lists joined (module docstring).
WHOLE_ROW_COLUMNS = 16384
#: Up to this many candidates a request folds over at most :data:`SMALL_AXIS_FOLD`
#: partitions, past it over every one: the faster fold on the device. On a short axis the
#: 32 bit steps cost the same cross-partition sum whatever the fold, and a 16-partition
#: fold keeps the per-step tiles smallest; on a longer axis the per-step column work
#: dominates and every partition shortens it.
SMALL_AXIS_COLUMNS = 2048
SMALL_AXIS_FOLD = 16
#: The widest candidate axis served: a lone request folds over 128 partitions of
#: :data:`FOLD_COLUMNS_LIMIT` columns. A physical (SBUF) bound, not a policy.
MAX_SELECT_CANDIDATES = PARTITIONS * FOLD_COLUMNS_LIMIT
#: What a chunk list holds where it holds no id. It stays negative when shifted by any
#: chunk's first candidate (below ``MAX_SELECT_CANDIDATES``), so the join's maximum never
#: takes it over an id. A power of two, exact as an immediate.
_LIST_PAD = -MAX_SELECT_CANDIDATES
#: The whole-row chunks of the widest candidate axis whose chunk walk has run on trn2 in a
#: decode graph: the capture build of the served graph (the same nodes plus copies of the
#: selection), at one request per step, selected the torch oracle's ids on every DSA layer
#: and rank. The walk is also checked against the torch oracle in the NKI simulator.
DEVICE_CHECKED_ROW_CHUNKS = 4
#: The widest candidate axis the decode indexer serves at one request per decode step; the
#: runner refuses a ``max_model_len`` past it (:func:`check_decode_index_context`).
MAX_DECODE_INDEX_CANDIDATES = DEVICE_CHECKED_ROW_CHUNKS * WHOLE_ROW_COLUMNS
#: The widest candidate axis the decode indexer serves when a decode step batches more
#: than one request (``max_num_seqs > 1``): the whole-row width. The chunk walk has run on
#: trn2 at one request per step only.
MAX_BATCHED_DECODE_INDEX_CANDIDATES = WHOLE_ROW_COLUMNS
#: The widest per-rank candidate axis decode context parallelism serves
#: (:func:`dsa_decode_select_pools`, :func:`check_decode_pools_context`): the whole-row
#: width, the only path that reads each selected id's score.
MAX_DECODE_POOLS_CANDIDATES = WHOLE_ROW_COLUMNS
#: The longest chunk (columns per partition) whose ``k``-th search takes four-bit digits;
#: a longer one takes single bits (module docstring, step 1). A digit step counts 15
#: trials where a bit step counts one, and both end in one cross-partition sum: on a
#: short chunk the sums and their engine hand-offs dominate, on a long one the counts.
RADIX_COLUMNS = 128
#: The trials of a four-bit digit, its nonzero values.
RADIX_TRIALS = 15
#: The four-bit digits below the sign bit, their lowest bits ``27, 23, .., 3``; a
#: three-bit digit takes bits ``2 .. 0``.
RADIX_STEPS = 7
#: What a row of :func:`dsa_decode_select_pools`' local top-``k`` holds past its real
#: entries, beside the id :data:`POOL_PAD_ID`: the causal sentinel value, ``BOUND_FILL``
#: (-1.0e30), which the score kernel writes at every pool that is not complete. Every real
#: entry is above :data:`BOUND_FILL_MARK`, so a pad ranks after every real entry in the
#: merge whatever its tie rule.
POOL_PAD_VALUE = BOUND_FILL
#: The pool id of a pad entry: the selection's ``SENTINEL`` (-1).
POOL_PAD_ID = SENTINEL
#: fp32 columns of a ``[P, 1]`` scratch tile: one whole 32-byte line.
_LINE = 8
#: The sign bit as an int32. Every immediate here is a power of two or ``-1``, so it is
#: exact whatever width the engine carries an immediate in.
_SIGN_BIT = -2147483648
_SUPPORTED_DTYPES = (torch.float32,)

#: This file's content digest, handed to the kernel as a trace-time int: the compiled
#: kernel cache keys on a kernel's own source and its arguments, and the kernel calls
#: helpers whose edits it would otherwise not see.
SOURCE_DIGEST = source_digest(__file__)


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


def row_chunks(width: int) -> int:
    """Chunks of :data:`WHOLE_ROW_COLUMNS` columns a ``width``-wide mask row is compacted
    in: one, the whole row, up to that width."""
    return _ceil_div(int(width), WHOLE_ROW_COLUMNS)


def mask_span(width: int) -> int:
    """The selection mask's width for a ``width``-wide candidate axis: the axis up to
    :data:`WHOLE_ROW_COLUMNS`, past it :func:`row_chunks` whole chunks (the scores past the
    axis are ``BOUND_FILL``, never selected)."""
    chunks = row_chunks(width)
    return int(width) if chunks == 1 else chunks * WHOLE_ROW_COLUMNS


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

    The MSB-first search the module docstring describes: four-bit digits on a chunk of up
    to :data:`RADIX_COLUMNS` columns (:func:`_digit_search`), single bits past it. Step 0
    tests ``+0.0``, whose key is the sign bit alone; it fixes the key-to-bits map of every
    later trial: XOR with the sign bit for a non-negative result, with ``-1`` for a
    negative one.
    """
    junk = _sb((parts, chunk), nl.float32)
    count = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=junk, data=scores, op0=nl.greater_equal, operand0=0.0,
                              reduce_op=nl.add, reduce_res=count)
    total = _request_sum(same, count, parts)
    taken = _col(parts, nl.int32)  # -1 where the bit is taken, else 0
    nisa.tensor_scalar(dst=taken, data=total, op0=nl.greater_equal,
                       operand0=float(select_k), op1=nl.multiply, operand1=-1.0)
    if chunk <= RADIX_COLUMNS:
        return _digit_search(scores, same, taken, parts, chunk, select_k), junk
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


def _digit_search(scores, same, taken, parts, chunk, select_k):
    """The four-bit search below the sign bit, ``[parts, 1]`` fp32: the ``k``-th value.

    The search carries the bits found so far XOR the result's key-to-bits map, ``-1`` for
    a negative result and 0 for a non-negative one (``taken`` inverted), so that a trial
    is that value XOR ``t << shift`` (the digit's bits are clear in it) and the value is
    the result's bits after the last digit.
    """
    digits = _sb((parts, RADIX_TRIALS + 1), nl.int32)  # 1 .. 15 on every partition
    nisa.iota(dst=digits[:, 0:RADIX_TRIALS], pattern=[[1, RADIX_TRIALS]], offset=1,
              channel_multiplier=0)
    found = _col(parts, nl.int32)
    nisa.tensor_scalar(dst=found, data=taken, op0=nl.bitwise_xor, operand0=-1)
    for step in range(RADIX_STEPS):  # bits 30 .. 3, four at a time
        found = _digit_step(scores, same, digits, found, parts, chunk, select_k,
                            27 - 4 * step, RADIX_TRIALS)
    # Bits 2 .. 0: a three-bit digit, seven trials.
    found = _digit_step(scores, same, digits, found, parts, chunk, select_k, 0, 7)
    return found.view(nl.float32)


def _digit_step(scores, same, digits, found, parts, chunk, select_k, shift, trials):
    """``found`` with the digit at ``shift`` (``trials`` nonzero values) decided,
    ``[parts, 1]`` int32 (:func:`_digit_search`).

    A request's count of scores ``>=`` a trial falls as the trial rises, so the digit is
    the number of trials that at least ``select_k`` scores reach.
    """
    trial = _sb((parts, RADIX_TRIALS + 1), nl.int32)
    nisa.tensor_scalar(dst=trial[:, 0:trials], data=digits[:, 0:trials], op0=nl.left_shift,
                       operand0=shift, op1=nl.bitwise_xor, operand1=found)
    reach = _sb((parts, trials * chunk), nl.float32)  # score >= trial, trial-major
    nisa.tensor_tensor(
        dst=reach, data1=scores.ap(pattern=[[chunk, parts], [0, trials], [1, chunk]]),
        data2=trial.view(nl.float32).ap(pattern=[[RADIX_TRIALS + 1, parts], [1, trials],
                                                 [0, chunk]]),
        op=nl.greater_equal)
    count = _sb((parts, RADIX_TRIALS + 1), nl.float32)
    nisa.tensor_reduce(dst=count[:, 0:trials], op=nl.add,
                       data=reach.reshape((parts, trials, chunk)), axis=2)
    total = nl.ndarray((parts, RADIX_TRIALS + 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=total[:, 0:trials], stationary=same, moving=count[:, 0:trials])
    met = _sb((parts, RADIX_TRIALS + 1), nl.float32)
    digit = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=met[:, 0:trials], data=total[:, 0:trials],
                              op0=nl.greater_equal, operand0=float(select_k),
                              reduce_op=nl.add, reduce_res=digit)
    digit_i = _col(parts, nl.int32)  # a count of at most 15: exact
    nisa.tensor_copy(dst=digit_i, src=digit)
    grown = _col(parts, nl.int32)
    nisa.tensor_scalar(dst=grown, data=digit_i, op0=nl.left_shift, operand0=shift,
                       op1=nl.bitwise_xor, operand1=found)
    return grown


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


def _compact_row_chunks(mask_hbm, ids_hbm, lists_hbm, counts_hbm, r0, n, chunks, select_k):
    """Each request's mask row compacted in ``chunks`` chunks of :data:`WHOLE_ROW_COLUMNS`
    and the chunk lists joined: its ids to row ``r`` of ``ids_hbm``, ``-1`` after them.

    Row ``q = r * chunks + c`` (``r`` counted from ``r0``) is chunk ``c`` of request ``r``
    in each scratch: ``mask_hbm`` read as rows of :data:`WHOLE_ROW_COLUMNS`, ``lists_hbm``
    (``[N, 1]`` int32, so that a window offset counts single ids) as rows of ``2 k``, and
    ``counts_hbm`` (``[N, 1]`` int32).

    1. One ``nonzero_with_count`` per eight rows, on partitions ``16 i``. Its first ``k``
       columns go to the second half of the row's list, behind ``k`` columns of
       :data:`_LIST_PAD` written for every row by one DMA, and its count to ``counts_hbm``.
       One DMA per row (:func:`_gpsimd_copy`).
    2. On partition 0: ``start[q] = k - p``, ``p`` the ids of the request's chunks below
       ``c``. A request selects at most ``k``, so ``0 <= start <= k``.
    3. One DMA per row reads ``k`` columns of its list from column ``start[q]``, an SBUF
       offset read when the DMA runs, into columns ``[c k, (c + 1) k)`` of the request's
       partition. The chunk's ids land at their places in the joined list (``p`` onward);
       the rest of the window is pad, the prefix's or the list's own.
    4. Each window is shifted by its chunk's first candidate (a pad stays negative) and
       the windows join by an element-wise maximum; a place that no chunk fills becomes
       ``-1``.
    """
    chunk_cols = WHOLE_ROW_COLUMNS
    list_cols = 2 * select_k
    rows_n = n * chunks
    row0 = r0 * chunks

    # ---- 1. chunk lists behind a pad prefix, and their counts ---------------------------
    prefix = _sb((rows_n, select_k), nl.int32)
    nisa.memset(dst=prefix, value=_LIST_PAD)
    nisa.dma_copy(dst=lists_hbm.ap(pattern=[[list_cols, rows_n], [1, select_k]],
                                   offset=row0 * list_cols), src=prefix)
    for g in range(_ceil_div(rows_n, COMPACT_ROWS)):
        m = min(COMPACT_ROWS, rows_n - g * COMPACT_ROWS)
        group_row = row0 + g * COMPACT_ROWS
        found = _gpsimd_dst((PARTITIONS, chunk_cols + 1), nl.int32)
        rows = _sb((PARTITIONS, chunk_cols), nl.float32)
        _gpsimd_copy(rows, mask_hbm, group_row, m, chunk_cols, chunk_cols, 0, True)
        nisa.nonzero_with_count(dst=found, src=rows, index_offset=0, padding_val=_LIST_PAD)
        _gpsimd_copy(found, lists_hbm, group_row, m, select_k, list_cols, select_k, False)
        for i in range(m):
            part = GPSIMD_STRIDE * i
            nisa.dma_copy(dst=counts_hbm.ap(pattern=[[1, 1], [1, 1]], offset=group_row + i),
                          src=found[part:part + 1, chunk_cols:chunk_cols + 1])

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

    # ---- 3. the windows: each chunk's ids at their places in the joined list -----------
    windows = _sb((n, chunks * select_k), nl.int32)
    for r in range(n):
        for c in range(chunks):
            q = r * chunks + c
            nisa.dma_copy(dst=windows[r:r + 1, c * select_k:(c + 1) * select_k],
                          src=lists_hbm.ap(pattern=[[list_cols, 1], [1, select_k]],
                                           offset=(row0 + q) * list_cols,
                                           scalar_offset=start[0:1, q:q + 1],
                                           indirect_dim=0))

    # ---- 4. the join: candidate ids, the largest of the windows at each place ----------
    ids = _sb((n, select_k), nl.int32)
    nisa.tensor_copy(dst=ids, src=windows[0:n, 0:select_k])
    for c in range(1, chunks):
        nisa.scalar_tensor_tensor(dst=ids, data=windows[0:n, c * select_k:(c + 1) * select_k],
                                  op0=nl.add, operand0=float(c * chunk_cols), op1=nl.maximum,
                                  operand1=ids)
    nisa.tensor_scalar(dst=ids, data=ids, op0=nl.maximum, operand0=-1)
    nisa.dma_copy(dst=ids_hbm.ap(pattern=[[select_k, n], [1, select_k]], offset=r0 * select_k),
                  src=ids)


def _rows_past_pad(tile, hbm, row0, m, cols):
    """:func:`_gpsimd_copy` of the first ``cols`` columns of rows ``row0 .. row0 + m - 1``
    of ``hbm`` into columns ``[1, cols + 1)`` of partitions ``0, 16, .., 16 (m - 1)`` of
    ``tile``, past its column 0."""
    row_cols = hbm.shape[1]
    for i in range(m):
        part = GPSIMD_STRIDE * i
        _dma(tile[part:part + 1, 1:cols + 1],
             hbm.ap(pattern=[[row_cols, 1], [1, cols]], offset=(row0 + i) * row_cols), True)


def _compact_pool_rows(mask_hbm, bounded_hbm, ids_hbm, values_hbm, r0, n, span, select_k):
    """:func:`_compact_whole_rows` with each id's score read beside it: requests ``r0 .. r0 +
    n - 1``' ids to ``ids_hbm`` (ascending, :data:`POOL_PAD_ID` after them) and their scores
    to ``values_hbm`` (:data:`POOL_PAD_VALUE` after them).

    The mask rows come into columns ``1 .. span`` of a tile whose column 0 holds
    :data:`POOL_PAD_VALUE`, and the compaction writes the ids out as it finds them, a pad
    as :data:`POOL_PAD_ID` (``-1``). After it the score rows come into the same columns, on
    the same GpSimd partitions, and one ``nc_n_gather`` per eight requests reads column
    ``id + 1`` for every id: the score at a real id, column 0 at a pad. The partitions
    between the rows gather column 0 too. Every value is a copy of a score, so the list
    holds the scores bit for bit. The one immediate, ``1``, is exact in any width, and every
    gather index is at least 0, so the unsigned view reads it as it is.

    One tile for both rows, at 16384 candidates the only room there is beside the
    compaction's (a third row-wide tile does not fit a partition). The pad column is written
    before the mask rows and read by the gather after the score rows, so the score rows are
    a later write to the same tile, ordered after the compaction's read of the mask rows; a
    tile whose every read follows the new write is given a second address by the device
    compiler, which put it over the mask rows without that order (the ordering check, one
    request at 16384 candidates).
    """
    width = bounded_hbm.shape[1]
    for g in range(_ceil_div(n, COMPACT_ROWS)):
        m = min(COMPACT_ROWS, n - g * COMPACT_ROWS)
        row0 = r0 + g * COMPACT_ROWS
        # Partitions 0 .. 16 (m - 1): every compute instruction here starts on partition 0.
        parts = GPSIMD_STRIDE * (m - 1) + 1
        rows = _sb((PARTITIONS, span + 1), nl.float32)
        nisa.memset(dst=rows[:, 0:1], value=POOL_PAD_VALUE)
        found = _gpsimd_dst((PARTITIONS, span + 1), nl.int32)
        nisa.memset(dst=found[0:parts, 0:select_k], value=POOL_PAD_ID)
        _rows_past_pad(rows, mask_hbm, row0, m, span)
        nisa.nonzero_with_count(dst=found, src=rows[:, 1:span + 1], index_offset=0,
                                padding_val=POOL_PAD_ID)
        _gpsimd_copy(found, ids_hbm, row0, m, select_k, select_k, 0, False)
        at = _sb((PARTITIONS, select_k), nl.int32)  # id + 1: the column of its score
        nisa.tensor_scalar(dst=at[0:parts, 0:select_k], data=found[0:parts, 0:select_k],
                           op0=nl.add, operand0=1)
        # ---- the score rows over the mask rows: every id is below the width -------------
        _rows_past_pad(rows, bounded_hbm, row0, m, width)
        values = _gpsimd_dst((PARTITIONS, select_k), nl.float32)
        nisa.nc_n_gather(dst=values[0:parts, 0:select_k], data=rows[0:parts, 0:span + 1],
                         indices=at[0:parts, 0:select_k].view(nl.uint32))
        _gpsimd_copy(values, values_hbm, row0, m, select_k, select_k, 0, False)


def _select_rows(bounded_hbm, mask_hbm, r0, n, select_k, fold_cap, span):
    """Requests ``r0 .. r0 + n - 1``: their selection, 1.0 or 0.0 per candidate, into
    their rows of ``mask_hbm`` (``[B, span]`` fp32; ``span`` as :func:`_select_tile`)."""
    width = bounded_hbm.shape[1]
    fold, chunk, request_major = fold_layout(n, span, fold_cap)
    parts = fold * n

    # ---- the scores in the fold layout, pads at BOUND_FILL ------------------------------
    scores = _sb((parts, chunk), nl.float32)
    if width < fold * chunk:
        nisa.memset(dst=scores, value=BOUND_FILL)
    _fold_copy(bounded_hbm, scores, r0, n, width, chunk, fold, request_major, True)
    same, below = _request_matrices(n, fold, request_major)

    # ---- the selection, out row by row --------------------------------------------------
    mask = _select_mask(scores, same, below, parts, chunk, select_k)
    _fold_copy(mask_hbm, mask, r0, n, span, chunk, fold, request_major, False)


def _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, ids_hbm, lists_hbm, counts_hbm,
                 tok_hbm, r0, n, select_k, pool_size, out_cols, fold_cap, span, values_hbm):
    """Requests ``r0 .. r0 + n - 1``: select (into the scratch), compact, expand.

    ``span``: the mask width, wider than the candidate axis where the columns past it are
    ``BOUND_FILL``: :func:`mask_span`, or (:func:`dsa_decode_select_pools` on an axis of at
    most ``k`` pools) :func:`select_span`. ``lists_hbm`` / ``counts_hbm``: the chunk walk's
    scratches (:func:`_compact_row_chunks`), ``None`` on the whole-row path.
    ``values_hbm``: ``None``, or where the selected scores go beside the ids
    (:func:`_compact_pool_rows`, the whole-row path only).
    """
    _select_rows(bounded_hbm, mask_hbm, r0, n, select_k, fold_cap, span)

    # ---- compaction: rows on the GpSimd cores' first partitions -------------------------
    chunks = row_chunks(span)
    if values_hbm is not None:
        _compact_pool_rows(mask_hbm, bounded_hbm, ids_hbm, values_hbm, r0, n, span, select_k)
    elif chunks == 1:
        _compact_whole_rows(mask_hbm, ids_hbm, r0, n, span, select_k)
    else:
        _compact_row_chunks(mask_hbm, ids_hbm, lists_hbm, counts_hbm, r0, n, chunks, select_k)

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
    span = mask_span(bounded_hbm.shape[1])
    chunks = row_chunks(span)
    out_hbm = nl.ndarray((batch, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)
    mask_hbm = nl.ndarray((batch, span), dtype=nl.float32, buffer=nl.private_hbm)
    ids_hbm = nl.ndarray((batch, select_k), dtype=nl.int32, buffer=nl.private_hbm)
    tok_hbm = nl.ndarray((batch, select_k * pool_size), dtype=nl.int32,
                         buffer=nl.private_hbm)
    lists_hbm = None
    counts_hbm = None
    if chunks > 1:
        # The chunk walk's scratches (_compact_row_chunks): a list row of 2 k ids and a
        # count per (request, chunk), one column each so that an offset counts single ids.
        lists_hbm = nl.ndarray((batch * chunks * 2 * select_k, 1), dtype=nl.int32,
                               buffer=nl.private_hbm)
        counts_hbm = nl.ndarray((batch * chunks, 1), dtype=nl.int32, buffer=nl.private_hbm)
    lo, hi, per_tile, n_tiles = _program_share(batch, span)
    for t in range(n_tiles):
        r0 = lo + t * per_tile
        n = min(per_tile, hi - r0)
        _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, ids_hbm, lists_hbm, counts_hbm,
                     tok_hbm, r0, n, select_k, pool_size, out_cols, fold_cap, span, None)
    return out_hbm


def _program_share(batch, width):
    """This program's requests ``lo .. hi - 1`` and their tiles, ``(lo, hi, per_tile,
    n_tiles)``: the requests split evenly over the programs of the launch grid, each
    share in tiles of up to :func:`tile_requests` requests."""
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    share = (batch + n_prgs - 1) // n_prgs
    lo = prg * share
    hi = min(batch, lo + share)
    per_tile = tile_requests(width)
    return lo, hi, per_tile, (hi - lo + per_tile - 1) // per_tile


@nki.jit
def dsa_decode_select_pools_kernel(bounded_hbm, lens_hbm, select_k, pool_size, out_cols,
                                   fold_cap, span, source_digest):
    """:func:`dsa_decode_select_kernel` and, beside its tokens, the local top-``k`` before
    the expansion (the decode context parallel merge's input).

    Args:
        bounded_hbm / lens_hbm / select_k / pool_size / out_cols / source_digest: as
            :func:`dsa_decode_select_kernel`.
        fold_cap: python int, :func:`select_fold_cap` of ``span``.
        span: python int, :func:`select_span`: the candidate axis, or ``select_k + 1`` on
            an axis of at most ``select_k`` pools (every pool there is selected); at most
            :data:`WHOLE_ROW_COLUMNS`.

    Returns:
        ``(tokens, values, pool_ids)``: ``[B, out_cols]`` int32, the selection kernel's
        layout; ``[B, select_k]`` fp32, the selected scores, then :data:`POOL_PAD_VALUE`;
        ``[B, select_k]`` int32, their pool ids (candidate-axis indices) in ascending
        order, then :data:`POOL_PAD_ID`. Entry ``j`` of ``values`` is the score of entry
        ``j`` of ``pool_ids``.
    """
    batch = bounded_hbm.shape[0]
    out_hbm = nl.ndarray((batch, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)
    values_hbm = nl.ndarray((batch, select_k), dtype=nl.float32, buffer=nl.shared_hbm)
    # The compacted ids: an output, and the expansion's input.
    pool_ids_hbm = nl.ndarray((batch, select_k), dtype=nl.int32, buffer=nl.shared_hbm)
    mask_hbm = nl.ndarray((batch, span), dtype=nl.float32, buffer=nl.private_hbm)
    tok_hbm = nl.ndarray((batch, select_k * pool_size), dtype=nl.int32,
                         buffer=nl.private_hbm)
    lo, hi, per_tile, n_tiles = _program_share(batch, span)
    for t in range(n_tiles):
        r0 = lo + t * per_tile
        n = min(per_tile, hi - r0)
        # No chunk walk scratches: the whole-row path only.
        _select_tile(bounded_hbm, lens_hbm, out_hbm, mask_hbm, pool_ids_hbm, None, None,
                     tok_hbm, r0, n, select_k, pool_size, out_cols, fold_cap, span, values_hbm)
    return out_hbm, values_hbm, pool_ids_hbm


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def decode_select_programs(batch: int) -> int:
    """Programs the selection launches: both cores of an LNC2 core from two requests on."""
    if lnc_pair() and int(batch) >= 2:
        return 2
    return 1


def check_decode_index_context(max_model_len: int, pool_size: int | None,
                               max_num_seqs: int) -> None:
    """Refuse a ``max_model_len`` whose decode candidate axis is past the served width.

    Args:
        max_model_len: the longest context the runner serves.
        pool_size: the model's tokens per pool (``index_kpool``); ``None`` for a model
            with no DSA indexer, which is not checked.
        max_num_seqs: the most requests one decode step batches.

    Raises:
        DecodeSelectError: ``max_model_len // pool_size`` (the candidate axis of the
            ``max_model_len`` decode graph) is above :data:`MAX_DECODE_INDEX_CANDIDATES`,
            or above :data:`MAX_BATCHED_DECODE_INDEX_CANDIDATES` at ``max_num_seqs > 1``.
    """
    if int(max_num_seqs) > 1:
        _check_context(max_model_len, pool_size, MAX_BATCHED_DECODE_INDEX_CANDIDATES,
                       "MAX_BATCHED_DECODE_INDEX_CANDIDATES",
                       f"the DSA decode indexer at max_num_seqs={max_num_seqs}",
                       "Lower max_model_len or set max_num_seqs=1.")
    _check_context(max_model_len, pool_size, MAX_DECODE_INDEX_CANDIDATES,
                   "MAX_DECODE_INDEX_CANDIDATES", "the DSA decode indexer")


def check_decode_pools_context(rank_context: int, pool_size: int | None) -> None:
    """Refuse a decode context parallel rank whose candidate axis is past the width
    :func:`dsa_decode_select_pools` serves.

    Args:
        rank_context: the tokens one rank's decode indexer covers.
        pool_size: as :func:`check_decode_index_context`.

    Raises:
        DecodeSelectError: ``rank_context // pool_size`` is above
            :data:`MAX_DECODE_POOLS_CANDIDATES`.
    """
    _check_context(rank_context, pool_size, MAX_DECODE_POOLS_CANDIDATES,
                   "MAX_DECODE_POOLS_CANDIDATES", "the decode context parallel selection")


def _check_context(context, pool_size, limit, limit_name, selection,
                   remedy="Lower max_model_len."):
    """Raise :class:`DecodeSelectError` when ``context // pool_size`` candidates exceed
    ``limit`` (named ``limit_name``, served by ``selection``; the message ends with
    ``remedy``); no check at ``pool_size`` ``None``."""
    if pool_size is None:
        return
    candidates = int(context) // int(pool_size)
    if candidates > limit:
        longest = (limit + 1) * int(pool_size) - 1
        raise DecodeSelectError(
            f"max_model_len={context} needs {candidates} decode index candidates "
            f"(index_kpool={pool_size}); {selection} serves up to {limit_name}={limit} "
            f"(max_model_len <= {longest}). {remedy}")


def _validate_scores(bounded: Tensor, select_k) -> tuple[int, int]:
    """Check the scores and ``select_k`` of a selection; ``(B, C)`` of ``bounded``.

    Args:
        bounded: ``[B, C]`` candidate scores, ``B >= 1``, ``C <= MAX_SELECT_CANDIDATES``.
        select_k: a Python int, ``0 < select_k < C``.

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
    if width > MAX_SELECT_CANDIDATES:
        raise DecodeSelectError(
            f"a {width}-wide candidate axis folds into more than {FOLD_COLUMNS_LIMIT} "
            f"columns per partition, past what one partition's SBUF holds; the widest "
            f"served is {MAX_SELECT_CANDIDATES} pools")
    return batch, width


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
    if not isinstance(pool_size, int) or not is_power_of_two(pool_size):
        raise DecodeSelectError(f"pool_size must be a power-of-two python int; got "
                                f"{pool_size!r}")
    batch, width = _validate_scores(bounded, select_k)
    if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != batch:
        raise DecodeSelectError(f"seq_lens must be [B] = [{batch}]; got "
                                f"{tuple(seq_lens.shape)}")
    return batch, width


def can_run_dsa_decode_select(bounded: Tensor, select_k: int, pool_size: int) -> bool:
    """Whether the kernel serves this call; ``False`` sends it to the torch oracle."""
    return _kernel_serves(bounded, select_k)


def _kernel_serves(bounded: Tensor, select_k: int) -> bool:
    """Whether a selection kernel serves these scores (either entry)."""
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
    tokens = call(bounded.contiguous(), seq_lens.reshape(batch, 1).to(torch.int32).contiguous(),
                  int(select_k), int(pool_size), out_cols, int(select_fold_cap(width)),
                  SOURCE_DIGEST)
    # Under a tensor-capture config the kernel's inputs and output become graph outputs,
    # to compare with dsa_decode_select_torch_oracle offline; without one these calls
    # trace nothing. A local import, as in functional/sampling.py: debug code that the
    # kernel module's importers do not need.
    from vllm_neuron.accuracy.tensor_capture import capture_tensor

    capture_tensor("dsa_decode_select.bounded", bounded)
    capture_tensor("dsa_decode_select.seq_lens", seq_lens)
    capture_tensor("dsa_decode_select.tokens", tokens)
    return tokens


# ---------------------------------------------------------------------------------------
# Decode context parallelism: the local top-k before the expansion
# ---------------------------------------------------------------------------------------


def select_span(width: int, select_k: int) -> int:
    """The mask width :func:`dsa_decode_select_pools` selects over: the candidate axis, or
    ``select_k + 1`` columns (the ones past the axis at ``BOUND_FILL``) when the axis holds
    no more than ``select_k`` pools, where every complete pool is selected."""
    return max(int(width), int(select_k) + 1)


def _validate_pools(bounded: Tensor, seq_lens: Tensor, select_k, pool_size) -> tuple[int, int]:
    if bounded.ndim != 2:
        raise DecodeSelectError(f"bounded must be [B, C]; got {tuple(bounded.shape)}")
    batch, width = int(bounded.shape[0]), int(bounded.shape[1])
    if batch < 1 or width < 1:
        raise DecodeSelectError(f"bounded must carry at least one request and one pool; got "
                                f"{tuple(bounded.shape)}")
    if not isinstance(select_k, int) or isinstance(select_k, bool) or select_k < 1:
        raise DecodeSelectError(f"select_k must be a positive python int; got {select_k!r}")
    if not isinstance(pool_size, int) or not is_power_of_two(pool_size):
        raise DecodeSelectError(f"pool_size must be a power-of-two python int; got "
                                f"{pool_size!r}")
    span = select_span(width, select_k)
    if span > MAX_DECODE_POOLS_CANDIDATES:
        raise DecodeSelectError(
            f"a {width}-wide candidate axis is past MAX_DECODE_POOLS_CANDIDATES="
            f"{MAX_DECODE_POOLS_CANDIDATES}, the whole-row width: the only path that reads "
            f"each selected id's score")
    if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != batch:
        raise DecodeSelectError(f"seq_lens must be [B] = [{batch}]; got "
                                f"{tuple(seq_lens.shape)}")
    return batch, width


def dsa_decode_select_pools(bounded: Tensor, seq_lens: Tensor, *, select_k: int,
                            pool_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """:func:`dsa_decode_select`'s tokens and the local top-``k`` before the expansion.

    The input of trn2-2's decode context parallel merge (item 3): each rank selects over
    its own pools and the ``CP`` lists merge into the global one. Rows are requests
    (``R = B`` at one decode token each). The list is in pool order, not score order: the
    merge ranks the ``CP * k`` entries by score and global pool id, which does not depend on
    the order they come in, and this selection never sorts.

    Args:
        bounded: ``[B, C]`` fp32, a rank's bounded candidate scores over its local pool
            axis (``dsa_decode_scores`` with the rank's operands).
        seq_lens: ``[B]`` int, each request's length on this rank (its owned length).
        select_k: pools selected per request, a positive python int. ``C <= select_k`` is
            served: every complete pool is selected.
        pool_size: tokens per pool, a power of two.

    Returns:
        ``(tokens, values, pool_ids)``:

        * ``tokens``: ``[B, index_expand_width(select_k, pool_size)]`` int32, the
          selection's expanded token indices, :func:`dsa_decode_select`'s where ``C >
          select_k``;
        * ``values``: ``[B, select_k]`` fp32, the score of each entry of ``pool_ids``
          (bit for bit ``bounded[b, pool_ids[b, j]]``), :data:`POOL_PAD_VALUE` at a pad;
        * ``pool_ids``: ``[B, select_k]`` int32, the selected local candidate-axis indices
          in ascending order (the ``k`` largest scores, ties at the ``k``-th value to the
          lowest index), :data:`POOL_PAD_ID` after them in a row with fewer than
          ``select_k`` complete pools.
    """
    batch, width = _validate_pools(bounded, seq_lens, select_k, pool_size)
    span = select_span(width, select_k)
    out_cols = int(index_expand_width(select_k, pool_size))
    if not (can_run_kernel(bounded) and bounded.dtype in _SUPPORTED_DTYPES):
        _count_torch_fallback()
        return dsa_decode_select_pools_torch_oracle(bounded, seq_lens, select_k=select_k,
                                                    pool_size=pool_size)
    programs = decode_select_programs(batch)
    _count_dispatch("select", programs, batch)
    call = wrap_nki(dsa_decode_select_pools_kernel)
    if programs == 2:
        call = call[2]
    return call(bounded.contiguous(), seq_lens.reshape(batch, 1).to(torch.int32).contiguous(),
                int(select_k), int(pool_size), out_cols, int(select_fold_cap(span)), int(span),
                SOURCE_DIGEST)


# ---------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------


def _selected_pools_torch(bounded: Tensor, select_k: int) -> Tensor:
    """``[B, select_k]`` int64: each request's selected candidates in ascending order,
    ``C`` after them. A stable descending sort spells the tie rule: it keeps equal values
    in index order, so its first ``select_k`` are the ``select_k`` largest with ties to the
    lowest index; the masked pools among them are dropped (the ``dsa_causal_sentinel``
    rule)."""
    width = int(bounded.shape[1])
    top = torch.sort(bounded.to(torch.float32), dim=1, descending=True, stable=True)
    values, ids = top.values[:, :select_k], top.indices[:, :select_k]
    keep = values > BOUND_FILL_MARK
    return torch.where(keep, ids, torch.full_like(ids, width)).sort(dim=1).values


def dsa_decode_select_torch_oracle(bounded: Tensor, seq_lens: Tensor, *, select_k: int,
                                   pool_size: int) -> Tensor:
    """CPU reference and fallback: :func:`_selected_pools_torch`'s pools, ``-1`` after
    them, laid out as tokens by ``dsa_index_expand``'s torch reference."""
    width = int(bounded.shape[1])
    ids = _selected_pools_torch(bounded, select_k)
    pool_ids = torch.where(ids < width, ids, torch.full_like(ids, SENTINEL)).to(torch.int32)
    return _dsa_index_expand_torch(pool_ids, seq_lens.to(torch.int32), pool_size)


def dsa_decode_select_mask_torch_oracle(bounded: Tensor, *, select_k: int) -> Tensor:
    """:func:`_selected_pools_torch`'s pools as a ``[B, C]`` fp32 mask, 1.0 where the
    request selects candidate ``c``: the CPU reference of the pools ``mla_decode_masked``
    selects on chip (no tail tokens; the reader adds them from the row's length)."""
    batch, width = int(bounded.shape[0]), int(bounded.shape[1])
    ids = _selected_pools_torch(bounded, select_k)
    mask = torch.zeros(batch, width + 1, dtype=torch.float32)
    mask.scatter_(1, ids, 1.0)
    return mask[:, :width]


def dsa_decode_select_pools_torch_oracle(bounded: Tensor, seq_lens: Tensor, *,
                                         select_k: int, pool_size: int
                                         ) -> tuple[Tensor, Tensor, Tensor]:
    """CPU reference and fallback of :func:`dsa_decode_select_pools`: the scores padded with
    ``BOUND_FILL`` to :func:`select_span`, a stable descending sort (equal values in index
    order), its first ``select_k`` that are complete pools, back in ascending index order,
    the pads after them."""
    batch, width = int(bounded.shape[0]), int(bounded.shape[1])
    scores = bounded.to(torch.float32)
    span = select_span(width, select_k)
    if span > width:
        scores = torch.cat([scores, scores.new_full((batch, span - width), BOUND_FILL)], dim=1)
    top = torch.sort(scores, dim=1, descending=True, stable=True)
    first = top.indices[:, :select_k]
    keep = top.values[:, :select_k] > BOUND_FILL_MARK
    ids = torch.where(keep, first, torch.full_like(first, span)).sort(dim=1).values
    real = ids < span
    values = torch.gather(scores, 1, torch.where(real, ids, torch.zeros_like(ids)))
    values = torch.where(real, values, torch.full_like(values, POOL_PAD_VALUE))
    pool_ids = torch.where(real, ids, torch.full_like(ids, POOL_PAD_ID)).to(torch.int32)
    tokens = dsa_decode_select_torch_oracle(scores, seq_lens, select_k=select_k,
                                            pool_size=pool_size)
    return tokens, values, pool_ids
