# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's top-k under decode context parallelism (DCP).

Under DCP the indexer pool axis is sharded over the ``dcp_size`` ranks of a KV group: the
128-token block ``b`` lives on rank ``b % dcp_size`` as that rank's local block
``b // dcp_size``. Each rank therefore selects its own top ``k`` pools over the pools it
holds (prefill: ``dsa_topk_select`` after ``dsa_causal_bound``; decode: the batched decode
select), and this module merges those ``dcp_size`` local selections into the selection
CP=1 makes over the whole axis. One NKI kernel, :func:`dsa_dcp_merge_select_kernel`, does
the merge; :func:`dsa_dcp_select_candidates` is the call site's entry point.

**Global pool ids.** With ``ppb = block_size // index_kpool`` pools per block, local pool
``p`` of rank ``r`` is global pool ``g = (p // ppb) * ppb * CP + r * ppb + p % ppb``. Rank
``r`` owns ``g`` exactly when ``(g // ppb) % CP == r``, and then ``p = (g // ppb // CP) *
ppb + g % ppb``. Within one rank the map is increasing, so a rank's lowest local index is
its lowest global id.

**The rule.** Every rank sees the same ``CP * k`` candidates per row (one all-gather) and
selects the same ``k``: the largest values, and among equal values the lower global pool
id; a pad never. That is CP=1's own rule ("lowest candidate index first",
``decode_select.py``) on the global axis. A pad is a slot whose local id is ``-1`` or whose
value is at or below ``BOUND_FILL_MARK`` (above the causal bound's fill, below every real
score), whatever its id. Each rank returns the selected candidates it owns as local pool ids,
in its own input's slot order, ``-1`` in every other slot: the ``[R, k]`` int32 that
``dsa_index_expand`` expands and whose ``-1`` the sparse attention masks wherever it stands.
The candidates a rank owns are exactly its own input's, so no id crosses ranks on the way out.

**Exactness against CP=1.** A row's CP=1 selection whose ``k``-th value has no tie at its
boundary is a set every rank's local selection contains its part of (the part is at most
``k`` candidates, each above every candidate outside it), so the merge returns that set
exactly. At a boundary tie the selected scores agree as a multiset; the ids agree too when
every local selection broke its own ties by the lowest local index (the decode select
does), because the lowest local indices are the lowest global ids.

**Order.** The kernel reads no order from its inputs. Every count runs over a row's whole
candidate set and the selection is a threshold compare, so a rank's local selection may come
in any slot order (``dsa_topk_select``'s, the decode select's ascending ids, any permutation)
and the result is the same set, in that input's slot order. It merges no sorted runs and
never takes a slot, entry 0 or any other, for a row's maximum.

**The kernel.** Per row it finds the threshold ``(v*, g*)`` and keeps every candidate above
it. ``v*``, the ``k``-th largest value, is found by its bits: an MSB-first search on the
order-preserving integer key of an fp32 value, a ``b``-bit digit a step
(:func:`radix_bits`): ``2**b - 1`` independent trial counts, then the digit in one dependent
hop. A count is one Vector Engine ``tensor_scalar_reduce`` over a partition's candidates.
Every candidate ``>= v*`` is the selection unless a row holds more than ``k`` of them, an
excess tie at ``v*``. Only a group of tiles with such a row takes a device branch that finds
the ``c = k - #{value > v*}`` lowest global ids among the ties, the same search over
:data:`ID_BITS` bits of the id, and reselects. The count passes are the kernel's cost: 32
for the value at one bit a step, 24 more for the id on the branch (see the benchmark report).

Layout: rows ride the partitions, a tile of up to 128 rows, :data:`_TILES_IN_FLIGHT` tiles a
group. A tile of ``n < 128`` rows folds each row over ``f`` partitions (``n * f <= 128``,
:func:`tile_fold`) as whole ranks' inputs or equal pieces of one rank's input
(:func:`fold_groups`), so each rank's ``n`` rows are one run of partitions that one DMA loads;
a per-partition count is summed over a row's partitions by one matmul with the same-row
matrix. Both cores of an LNC2 pair split the rows.

**Invariant: every loop trip count and branch condition is identical on both cores by
construction.** Every static loop is fixed by the shapes, and both cores run as many groups
(:func:`program_groups`). The one data-dependent branch, a group's tie step, runs on the pair's
joint flag, which both cores hold after swapping their counts (:func:`_any_excess`). A core
alone in a loop body would wait for good on the barrier of both cores that ends it.

Preconditions the kernel reads no value to check: a real candidate's value is finite, a
rank's real local ids in one row are distinct, and every global id is below
:data:`GLOBAL_ID_LIMIT` (the id search runs in fp32, which holds every integer below it
exactly). A pad needs nothing: its value and id are replaced before anything reads them.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from torch import Tensor

from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL_MARK, SENTINEL
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import (
    SBUF_BYTES_PER_PARTITION,
    SBUF_PARTITIONS,
    can_run_kernel,
)

logger = logging.getLogger(__name__)

#: Rows one tile carries at the most, one per SBUF partition.
PARTITIONS = SBUF_PARTITIONS
#: Every global pool id must be below this: ``2**24``, the integers an fp32 holds exactly. The
#: id search covers ``ID_BITS`` bits. It is a 64M-token context at four tokens a pool.
ID_BITS = 24
GLOBAL_ID_LIMIT = 1 << ID_BITS
#: The value an invalid candidate (local id ``-1``) is given: the lowest finite fp32, below
#: every valid value, and finite, so the value search never meets an infinity.
LOWEST = -3.4028234663852886e38
#: SBUF bytes per candidate column of one tile: the value and the local id (4 bytes each), the
#: pad mask and the count scratch (1 byte each), the tie break's id key and its upper part
#: (4 bytes each), and the per-slot masks and result (7 bytes) of this rank's own columns, which
#: are at most every column.
_BYTES_PER_CANDIDATE = 4 + 4 + 1 + 1 + 4 + 4 + 7
#: Tiles one group selects together, so one tile's dependent search steps fill the gaps of the
#: other's; their SBUF fits at once.
_TILES_IN_FLIGHT = 2
#: Candidates per row served at the most: the widest ``CP * k`` whose tiles fit one partition.
MAX_CANDIDATES = SBUF_BYTES_PER_PARTITION // (_TILES_IN_FLIGHT * _BYTES_PER_CANDIDATE)
#: fp32 columns of a ``[P, 1]`` scratch tile: one whole 32-byte line.
_LINE = 8
#: The lowest exponent bit of an fp32. :data:`LOWEST`'s order-preserving key is ``1 << 23``
#: exactly, the lowest key any candidate holds.
_EXPONENT_LSB = 23
#: The sign bit as an int32. Every immediate of the value search is a power of two or ``-1``.
_SIGN_BIT = -2147483648
_VALUE_DTYPES = (torch.float32,)
_ID_DTYPES = (torch.int32,)

#: This file's content digest, handed to the kernel as a trace-time int: the compiled kernel
#: cache keys on a kernel's own source and its arguments, and the kernel calls helpers whose
#: edits it would otherwise not see.
SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)


class DcpCandidateMergeError(ValueError):
    """A malformed or unservable merge call."""


@dataclass
class _DispatchCounters:
    """Per-process record of how the merge seam was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _DispatchCounters()


def reset_dcp_merge_select_dispatch_counters() -> None:
    """Zero the merge seam's counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def dcp_merge_select_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def dcp_merge_select_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps."""
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)



# ---------------------------------------------------------------------------------------------
# Trace-time geometry, shared by the kernel and the host side
# ---------------------------------------------------------------------------------------------


def tile_fold(rows: int, cp: int, k: int) -> int:
    """Partitions one row of a ``rows``-row tile folds over: the largest admissible ``f`` with
    ``rows * f <= 128``.

    Admissible: a divisor of ``cp`` (a partition holds ``cp // f`` whole ranks' inputs) or
    ``cp * d`` for a divisor ``d`` of ``k`` (a partition holds ``k // d`` columns of one rank's
    input), so no partition's columns straddle two ranks unevenly.
    """
    best = 1
    for f in range(1, PARTITIONS // int(rows) + 1):
        if cp % f == 0 or (f % cp == 0 and k % (f // cp) == 0):
            best = f
    return best


def fold_groups(f: int, cp: int) -> tuple[int, int, int]:
    """``(groups, ranks, pieces)`` of a fold over ``f`` partitions a row.

    The row's partitions form ``groups`` groups of ``ranks`` whole ranks' inputs side by side,
    and each group's input is cut into ``pieces`` partitions. In an ``n``-row tile, partition
    ``(g * n + i) * pieces + c`` holds piece ``c`` of group ``g`` of row ``i``: each rank's
    ``n`` rows are then one run of partitions in their HBM order, loaded by one DMA.
    """
    if f <= cp:
        return (f, cp // f, 1)
    return (cp, 1, f // cp)


def rank_slot(f: int, cp: int, k: int, n: int, rank: int) -> tuple[int, int, int, int]:
    """``(first partition, partitions, first column, columns)`` of ``rank``'s input in an
    ``n``-row tile folded over ``f``: ``n * pieces`` partitions of ``k // pieces`` columns."""
    ranks, pieces = fold_groups(f, cp)[1:]
    return ((rank // ranks) * n * pieces, n * pieces, (rank % ranks) * k, k // pieces)


def program_groups(rows: int, programs: int) -> list[list[tuple[bool, list[tuple[int, int]]]]]:
    """Per program, its ``(stores, tiles)`` groups: an even share of the rows each, in tiles of up
    to :data:`PARTITIONS` rows, :data:`_TILES_IN_FLIGHT` ``(first row, rows)`` tiles a group.

    Every group breaks its ties on a device branch, and the two cores of an LNC2 pair must
    branch alike, so every program runs as many groups as the first: one with fewer repeats the
    first program's last group with ``stores`` false (computed, not stored). A program with no
    rows of its own, the second one of a single-row call, is the case that needs it.
    """
    share = (rows + programs - 1) // programs
    plans = []
    for program in range(programs):
        lo = program * share
        hi = min(rows, lo + share)
        tiles = []
        for start in range(lo, hi, PARTITIONS):
            tiles.append((start, min(PARTITIONS, hi - start)))
        groups = []
        for g0 in range(0, len(tiles), _TILES_IN_FLIGHT):
            groups.append((True, tiles[g0:g0 + _TILES_IN_FLIGHT]))
        plans.append(groups)
    for groups in plans[1:]:
        for _ in range(len(plans[0]) - len(groups)):
            groups.append((False, plans[0][-1][1]))
    return plans


def radix_bits(width: int, folded: bool) -> int:
    """Bits one search step resolves on a tile ``width`` candidates wide per partition.

    A step of ``b`` bits counts ``2**b - 1`` trial thresholds, independent of one another, then
    takes the digit in one dependent hop. An unfolded tile is bound by the count passes, so it
    resolves one bit a step, the fewest passes; a folded tile is narrow and bound by the hop
    (the per-step matmul that sums a row's partitions), so it trades passes for fewer hops.
    """
    if not folded or width >= 1024:
        return 1
    if width >= 256:
        return 2
    if width >= 64:
        return 3
    return 4


def digit_groups(bits: int, radix: int) -> list[tuple[int, int]]:
    """The MSB-first ``(shift, width)`` digits of ``radix`` bits (the last one narrower) that
    cover bits ``bits - 1 .. 0``."""
    groups = []
    for g in range((bits + radix - 1) // radix):
        top = bits - g * radix
        width = min(radix, top)
        groups.append((top - width, width))
    return groups


def value_digits(radix: int) -> list[tuple[int, int]]:
    """The MSB-first ``(shift, width)`` digits of the value search over key bits ``30 .. 0``,
    ``radix`` bits each, with a digit boundary at bit 23, the lowest exponent bit.

    Every candidate's key is at least :data:`LOWEST`'s, ``1 << 23``. A digit spanning bit 23 and
    lower bits, tried while every higher bit is 0, would try keys below ``1 << 23``: negative
    NaN bit patterns, which compare false with every value, so their counts would read 0 where
    every candidate lies above. With the boundary, bit 23 is settled before any lower bit is
    tried, and every trial is a key of at least ``1 << 23``.
    """
    digits = []
    for digit in digit_groups(31 - _EXPONENT_LSB, radix):
        digits.append((digit[0] + _EXPONENT_LSB, digit[1]))
    for digit in digit_groups(_EXPONENT_LSB, radix):
        digits.append((digit[0], digit[1]))
    return digits


# ---------------------------------------------------------------------------------------------
# Device helpers. Positional arguments only: the NKI front end drops keyword defaults.
# ---------------------------------------------------------------------------------------------


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    """A ``[parts, 1]`` view of a tile one 32-byte line wide."""
    return nl.ndarray((parts, _LINE), dtype=dtype, buffer=nl.sbuf)[:, 0:1]


def _same_row_matrix(n, groups, pieces):
    """``[P, P]`` fp32, ``P = groups * n * pieces``: 1 where partitions ``p`` and ``q`` carry
    the same row (``(p // pieces) % n``, :func:`fold_groups`), else 0. As a matmul's stationary
    operand it sums a per-partition column over each row's partitions."""
    parts = groups * n * pieces
    diff = _sb((n, parts), nl.float32)
    # diff[i, q] = row(q) - i: zero exactly where column q carries row i.
    nisa.iota(dst=diff, pattern=[[0, groups], [1, n], [0, pieces]], offset=0,
              channel_multiplier=-1)
    owner = _sb((n, parts), nl.float32)
    nisa.tensor_scalar(dst=owner, data=diff, op0=nl.equal, operand0=0.0)
    same_ps = nl.ndarray((parts, parts), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=same_ps, stationary=owner, moving=owner)
    same = _sb((parts, parts), nl.float32)
    nisa.tensor_copy(dst=same, src=same_ps)
    return same


def _rank_offsets(n, groups, ranks, pieces, ppb):
    """``[P, 1]`` fp32: ``ppb`` times the first rank of partition ``p``'s group,
    ``(p // (n * pieces)) * ranks``, built as a sum of one step a group boundary."""
    parts = groups * n * pieces
    part = _col(parts, nl.float32)
    nisa.iota(dst=part, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    acc = _col(parts, nl.float32)
    nisa.memset(dst=acc, value=0.0)
    for g in range(1, groups):
        rise = _col(parts, nl.float32)
        nisa.tensor_scalar(dst=rise, data=part, op0=nl.greater_equal,
                           operand0=float(g * n * pieces), op1=nl.multiply,
                           operand1=float(ranks * ppb))
        nxt = _col(parts, nl.float32)
        nisa.tensor_tensor(dst=nxt, data1=acc, data2=rise, op=nl.add)
        acc = nxt
    return acc


def _row_totals(counts, same, parts, cols):
    """``[parts, cols]`` per-partition counts summed over each row's partitions; the counts
    themselves when the tile is not folded."""
    if same is None:
        return counts
    lines = (cols + _LINE - 1) // _LINE
    out = nl.ndarray((parts, lines * _LINE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out[:, 0:cols], stationary=same, moving=counts)
    return out[:, 0:cols]


def _digit_steps(parts, trials, shift, offset, dtype):
    """``[parts, trials]``: ``(d << shift) + offset`` for ``d = 1 .. trials``, off the Vector
    Engine. An iota step must fit int16, so the iota counts ``d`` and the Scalar Engine scales
    it; every value is an integer an fp32 holds exactly."""
    counter = _sb((parts, trials), nl.float32)
    nisa.iota(dst=counter, pattern=[[1, trials]], offset=1, channel_multiplier=0)
    steps = _sb((parts, trials), dtype)
    nisa.tensor_scalar(dst=steps, data=counter, op0=nl.multiply, operand0=float(1 << shift),
                       op1=nl.add, operand1=float(offset), engine=nisa.scalar_engine)
    return steps


def _count_at_least(vals, junk, bounds, trials, parts):
    """``[parts, trials]`` fp32: per partition, how many of ``vals`` are ``>=`` each bound."""
    counts = _sb((parts, trials), nl.float32)
    for t in range(trials):
        nisa.tensor_scalar_reduce(dst=junk, data=vals, op0=nl.greater_equal,
                                  operand0=bounds[:, t:t + 1], reduce_op=nl.add,
                                  reduce_res=counts[:, t:t + 1])
    return counts


def _kth_value(vals, junk, same, parts, select_k, radix):
    """Each row's ``select_k``-th largest value, ``[parts, 1]`` fp32.

    MSB-first on the order-preserving key ``K`` (sign bit flipped for a non-negative value,
    every bit for a negative one). Step 0 tests ``+0.0``, whose key is the sign bit alone, and
    so fixes the sign of the answer and with it the map from a key to its value's bits: XOR with
    the sign bit for a non-negative answer, with ``-1`` for a negative one. The search carries
    ``bits``, the key found so far mapped to value bits, and every trial is ``bits`` XOR the
    trial digit (the found key and the digit share no bit). Each later step takes the next
    ``radix``-bit digit: the largest ``d`` for which at least ``select_k`` values are ``>=`` the
    trial with digit ``d``. The counts fall as ``d`` grows, so ``d`` is the number of trials
    that reach ``select_k``.

    A one-bit step carries the trial itself instead: the next trial is this one with the bit
    cleared where it fell short and the next bit set, two instructions on the dependent path.
    """
    count = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=junk, data=vals, op0=nl.greater_equal, operand0=0.0,
                              reduce_op=nl.add, reduce_res=count)
    total = _row_totals(count, same, parts, 1)
    kept = _col(parts, nl.int32)  # -1 where the answer is non-negative, else 0
    nisa.tensor_scalar(dst=kept, data=total, op0=nl.greater_equal, operand0=float(select_k),
                       op1=nl.multiply, operand1=-1.0)
    groups = value_digits(radix)
    if radix == 1:
        # The trial of bit 30: bits = ~kept (0 for a non-negative answer, -1 for a negative
        # one), with bit 30 flipped.
        trial = _col(parts, nl.int32)
        nisa.tensor_scalar(dst=trial, data=kept, op0=nl.bitwise_xor, operand0=~(1 << 30))
        for group in groups:
            bit = 1 << group[0]
            nxt_bit = 0
            if group[0] > 0:
                nxt_bit = bit >> 1
            count = _count_at_least(vals, junk, trial.view(nl.float32), 1, parts)
            total = _row_totals(count, same, parts, 1)
            short = _col(parts, nl.int32)  # the bit where the trial fell short, else 0
            nisa.tensor_scalar(dst=short, data=total, op0=nl.less, operand0=float(select_k),
                               op1=nl.multiply, operand1=float(bit))
            nxt = _col(parts, nl.int32)
            nisa.tensor_scalar(dst=nxt, data=short, op0=nl.bitwise_xor, operand0=trial,
                               op1=nl.bitwise_xor, operand1=nxt_bit)
            trial = nxt
        return trial.view(nl.float32)
    bits = _col(parts, nl.int32)
    nisa.tensor_scalar(dst=bits, data=kept, op0=nl.bitwise_xor, operand0=-1)
    for group in groups:
        shift = group[0]
        trials = (1 << group[1]) - 1
        digits = _digit_steps(parts, trials, shift, 0, nl.int32)
        bounds = _sb((parts, trials), nl.int32)
        nisa.tensor_scalar(dst=bounds, data=digits, op0=nl.bitwise_xor, operand0=bits)
        counts = _count_at_least(vals, junk, bounds.view(nl.float32), trials, parts)
        totals = _row_totals(counts, same, parts, trials)
        reached = _sb((parts, trials), nl.uint8)
        digit = _col(parts, nl.float32)
        nisa.tensor_scalar_reduce(dst=reached, data=totals, op0=nl.greater_equal,
                                  operand0=float(select_k), reduce_op=nl.add, reduce_res=digit)
        placed = _col(parts, nl.int32)
        nisa.tensor_scalar(dst=placed, data=digit, op0=nl.multiply, operand0=float(1 << shift))
        grown = _col(parts, nl.int32)
        nisa.tensor_tensor(dst=grown, data1=placed, data2=bits, op=nl.bitwise_xor,
                           engine=nisa.vector_engine)
        bits = grown
    return bits.view(nl.float32)


def _tie_cut(tie, junk, same, parts, quota, radix):
    """Each row's ``quota``-th lowest global id among its ties, as ``id - GLOBAL_ID_LIMIT``,
    ``[parts, 1]`` fp32; ``GLOBAL_ID_LIMIT - 1`` (minus the limit) when the row has fewer ties.

    ``tie`` holds ``id - GLOBAL_ID_LIMIT`` (in ``[-limit, -1]``) at a tie and ``0`` elsewhere.
    MSB-first over :data:`ID_BITS` bits, a ``radix``-bit digit a step, for the largest ``g``
    with fewer than ``quota`` ties below ``g``: that ``g`` is the ``quota``-th lowest tied id.
    Every value carried is an integer below the limit in magnitude, exact in fp32. A one-bit
    step carries the trial ``g - limit`` itself, as :func:`_kth_value` does.
    """
    groups = digit_groups(ID_BITS, radix)
    if radix == 1:
        trial = _col(parts, nl.float32)
        nisa.memset(dst=trial, value=float((1 << (ID_BITS - 1)) - GLOBAL_ID_LIMIT))
        for group in groups:
            bit = 1 << group[0]
            nxt_bit = 0
            if group[0] > 0:
                nxt_bit = bit >> 1
            count = _col(parts, nl.float32)
            nisa.tensor_scalar_reduce(dst=junk, data=tie, op0=nl.less, operand0=trial,
                                      reduce_op=nl.add, reduce_res=count)
            total = _row_totals(count, same, parts, 1)
            back = _col(parts, nl.float32)  # -bit where quota ties lie below the trial, else 0
            nisa.tensor_scalar(dst=back, data=total, op0=nl.greater_equal, operand0=quota,
                               op1=nl.multiply, operand1=float(-bit))
            nxt = _col(parts, nl.float32)
            nisa.scalar_tensor_tensor(dst=nxt, data=back, op0=nl.add, operand0=float(nxt_bit),
                                      op1=nl.add, operand1=trial)
            trial = nxt
        return trial
    found = _col(parts, nl.float32)
    nisa.memset(dst=found, value=0.0)
    for group in groups:
        shift = group[0]
        trials = (1 << group[1]) - 1
        digits = _digit_steps(parts, trials, shift, -GLOBAL_ID_LIMIT, nl.float32)
        bounds = _sb((parts, trials), nl.float32)
        nisa.tensor_scalar(dst=bounds, data=digits, op0=nl.add, operand0=found)
        counts = _sb((parts, trials), nl.float32)
        for t in range(trials):
            nisa.tensor_scalar_reduce(dst=junk, data=tie, op0=nl.less,
                                      operand0=bounds[:, t:t + 1], reduce_op=nl.add,
                                      reduce_res=counts[:, t:t + 1])
        totals = _row_totals(counts, same, parts, trials)
        short = _sb((parts, trials), nl.uint8)
        digit = _col(parts, nl.float32)
        nisa.tensor_scalar_reduce(dst=short, data=totals, op0=nl.less, operand0=quota,
                                  reduce_op=nl.add, reduce_res=digit)
        grown = _col(parts, nl.float32)
        nisa.scalar_tensor_tensor(dst=grown, data=digit, op0=nl.multiply,
                                  operand0=float(1 << shift), op1=nl.add, operand1=found)
        found = grown
    cut = _col(parts, nl.float32)
    nisa.tensor_scalar(dst=cut, data=found, op0=nl.add, operand0=float(-GLOBAL_ID_LIMIT))
    return cut


def _threshold_tile(values_hbm, ids_hbm, r0, n, cp_rank):
    """Rows ``r0 .. r0 + n - 1``: load every rank's candidates, find each row's ``k``-th value
    ``v*``, and select this rank's candidates ``>= v*``: the selection wherever a row has no
    excess tie at ``v*``. Returns the tile's state for :func:`_break_ties` and :func:`_store`.
    """
    cp = values_hbm.shape[0]
    rows = values_hbm.shape[1]
    k = values_hbm.shape[2]
    f = tile_fold(n, cp, k)
    pieces = fold_groups(f, cp)[2]
    parts = n * f
    width = (cp * k) // f
    # Each rank's [n, k] rows as [n * pieces, k // pieces]: one contiguous run of partitions.
    values_runs = values_hbm.reshape((cp, rows * pieces, k // pieces))
    ids_runs = ids_hbm.reshape((cp, rows * pieces, k // pieces))

    vals = _sb((parts, width), nl.float32)
    lids = _sb((parts, width), nl.int32)
    for rank in range(cp):
        slot = rank_slot(f, cp, k, n, rank)
        p0 = slot[0]
        c0 = slot[2]
        nisa.dma_copy(dst=vals[p0:p0 + slot[1], c0:c0 + slot[3]],
                      src=values_runs[rank, r0 * pieces:(r0 + n) * pieces, 0:slot[3]])
        nisa.dma_copy(dst=lids[p0:p0 + slot[1], c0:c0 + slot[3]],
                      src=ids_runs[rank, r0 * pieces:(r0 + n) * pieces, 0:slot[3]])
    # Pads are masked before any compare and before any global id is formed: a pad (id -1, or a
    # value at or below BOUND_FILL_MARK whatever its id) gets the value LOWEST and the id -1, so
    # it is never chosen and no slot ever emits it.
    junk = _sb((parts, width), nl.uint8)
    nisa.tensor_scalar(dst=junk, data=lids, op0=nl.less, operand0=0.0)
    invalid = _sb((parts, width), nl.uint8)
    nisa.scalar_tensor_tensor(dst=invalid, data=vals, op0=nl.less_equal,
                              operand0=BOUND_FILL_MARK, op1=nl.maximum, operand1=junk)
    lowest = _col(parts, nl.float32)
    nisa.memset(dst=lowest, value=LOWEST)
    nisa.tensor_copy_predicated(dst=vals, src=lowest.broadcast(1, width), predicate=invalid)
    no_pool = _col(parts, nl.int32)
    nisa.memset(dst=no_pool, value=SENTINEL)
    nisa.tensor_copy_predicated(dst=lids, src=no_pool.broadcast(1, width), predicate=invalid)

    same = None
    if f > 1:
        layout = fold_groups(f, cp)
        same = _same_row_matrix(n, layout[0], layout[2])
    kth = _kth_value(vals, junk, same, parts, k, radix_bits(width, f > 1))

    # Excess ties: more than k candidates >= v*, at a v* some valid candidate holds.
    at_least = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=junk, data=vals, op0=nl.greater_equal, operand0=kth,
                              reduce_op=nl.add, reduce_res=at_least)
    at_least_total = _row_totals(at_least, same, parts, 1)
    held = _col(parts, nl.float32)
    nisa.tensor_scalar(dst=held, data=kth, op0=nl.greater, operand0=LOWEST)
    excess = _col(parts, nl.float32)
    nisa.scalar_tensor_tensor(dst=excess, data=at_least_total, op0=nl.greater,
                              operand0=float(k), op1=nl.multiply, operand1=held)

    own = rank_slot(f, cp, k, n, cp_rank)
    c0 = own[2]
    cw = own[3]
    chosen = _sb((parts, cw), nl.uint8)
    nisa.tensor_scalar(dst=chosen, data=vals[:, c0:c0 + cw], op0=nl.greater_equal,
                       operand0=kth)
    result = _sb((parts, cw), nl.int32)
    nisa.memset(dst=result, value=SENTINEL, engine=nisa.gpsimd_engine)
    nisa.tensor_copy_predicated(dst=result, src=lids[:, c0:c0 + cw], predicate=chosen)
    return (r0, n, f, vals, lids, junk, same, kth, excess, result)


def _any_excess(states, programs, program):
    """A register holding 1 when some row of the group's tiles, on either core of an LNC2 pair,
    has an excess tie, else 0.

    The two cores swap their counts (``sendrecv``) and branch on the sum, so both take the
    branch or neither. The compiler ends every iteration of a dynamic loop on a barrier of both
    cores (a ``CoreBarrier`` in the lowered loop body); a core that ran the body while the
    other skipped it would wait on that barrier for good, a device hang.
    """
    total = None
    for state in states:
        parts = state[2] * state[1]
        ones = _col(parts, nl.float32)
        nisa.memset(dst=ones, value=1.0)
        rows_ps = nl.ndarray((1, _LINE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rows_ps[0:1, 0:1], stationary=state[8], moving=ones)
        nxt = _sb((1, _LINE), nl.float32)
        if total is None:
            nisa.tensor_copy(dst=nxt[0:1, 0:1], src=rows_ps[0:1, 0:1])
        else:
            nisa.tensor_tensor(dst=nxt[0:1, 0:1], data1=total, data2=rows_ps[0:1, 0:1],
                               op=nl.add)
        total = nxt[0:1, 0:1]
    if programs == 2:
        theirs = _sb((1, _LINE), nl.float32)
        nisa.sendrecv(src=total, dst=theirs[0:1, 0:1], send_to_rank=1 - program,
                      recv_from_rank=1 - program, pipe_id=0)
        both = _sb((1, _LINE), nl.float32)
        nisa.tensor_tensor(dst=both[0:1, 0:1], data1=total, data2=theirs[0:1, 0:1], op=nl.add)
        total = both[0:1, 0:1]
    flag = _sb((1, _LINE), nl.int32)
    nisa.tensor_scalar(dst=flag[0:1, 0:1], data=total, op0=nl.minimum, operand0=1.0)
    reg = nisa.register_alloc()
    nisa.register_load(dst=reg, src=flag[0:1, 0:1])
    return reg


def _break_ties(state, cp, k, cp_rank, ppb):
    """Reselect a tile's ties at ``v*`` by the lower global id: of each row's ties, the
    ``k - #{value > v*}`` with the lowest global ids. A row without an excess tie keeps every
    tie, as :func:`_threshold_tile` selected it."""
    n = state[1]
    f = state[2]
    vals = state[3]
    lids = state[4]
    junk = state[5]
    same = state[6]
    kth = state[7]
    result = state[9]
    parts = n * f
    width = (cp * k) // f
    layout = fold_groups(f, cp)

    # The id key, global id - GLOBAL_ID_LIMIT, built in place:
    #   g = (p // ppb) * ppb * cp + rank * ppb + p % ppb
    #     = (p & ~(ppb - 1)) * (cp - 1) + p + rank * ppb
    upper = _sb((parts, width), nl.int32)
    nisa.tensor_scalar(dst=upper, data=lids, op0=nl.bitwise_and, operand0=~(ppb - 1))
    tie = _sb((parts, width), nl.float32)
    nisa.scalar_tensor_tensor(dst=tie, data=upper, op0=nl.multiply, operand0=float(cp - 1),
                              op1=nl.add, operand1=lids)
    first = None
    if layout[0] > 1:
        first = _rank_offsets(n, layout[0], layout[1], layout[2], ppb)
    # A partition's column block b (k columns, or every column of a piece) holds the input of
    # its group's first rank plus b.
    block_cols = width // layout[1]
    for b in range(layout[1]):
        lo = b * block_cols
        shift = float(b * ppb - GLOBAL_ID_LIMIT)
        offset = shift
        if first is not None:
            offset = _col(parts, nl.float32)
            nisa.tensor_scalar(dst=offset, data=first, op0=nl.add, operand0=shift)
        nisa.tensor_scalar(dst=tie[:, lo:lo + block_cols], data=tie[:, lo:lo + block_cols],
                           op0=nl.add, operand0=offset)
    # The tie key: the id key at a value equal to v*, 0 elsewhere. An invalid candidate's key
    # is meaningless, and harmless: it ties only in a row whose v* is LOWEST, which has no
    # excess tie and whose invalid slots come out -1 whatever is chosen.
    nisa.scalar_tensor_tensor(dst=tie, data=vals, op0=nl.equal, operand0=kth, op1=nl.multiply,
                              operand1=tie)

    above = _col(parts, nl.float32)
    nisa.tensor_scalar_reduce(dst=junk, data=vals, op0=nl.greater, operand0=kth,
                              reduce_op=nl.add, reduce_res=above)
    above_total = _row_totals(above, same, parts, 1)
    quota = _col(parts, nl.float32)  # k - #{> v*}: the ties the row takes
    nisa.tensor_scalar(dst=quota, data=above_total, op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=float(k))
    cut = _tie_cut(tie, junk, same, parts, quota, radix_bits(width, f > 1))

    own = rank_slot(f, cp, k, n, cp_rank)
    c0 = own[2]
    cw = own[3]
    picked = _sb((parts, cw), nl.uint8)
    nisa.tensor_scalar(dst=picked, data=tie[:, c0:c0 + cw], op0=nl.less_equal, operand0=cut)
    chosen = _sb((parts, cw), nl.uint8)
    nisa.scalar_tensor_tensor(dst=chosen, data=vals[:, c0:c0 + cw], op0=nl.greater,
                              operand0=kth, op1=nl.maximum, operand1=picked)
    nisa.memset(dst=result, value=SENTINEL, engine=nisa.gpsimd_engine)
    nisa.tensor_copy_predicated(dst=result, src=lids[:, c0:c0 + cw], predicate=chosen)


def _store(state, out_hbm, cp, k, cp_rank):
    """DMA a tile's result, this rank's slots, to its rows of ``out_hbm``: one run of
    partitions, as :func:`_threshold_tile` loaded it."""
    r0 = state[0]
    n = state[1]
    f = state[2]
    result = state[9]
    rows = out_hbm.shape[0]
    pieces = fold_groups(f, cp)[2]
    own = rank_slot(f, cp, k, n, cp_rank)
    out_runs = out_hbm.reshape((rows * pieces, k // pieces))
    nisa.dma_copy(dst=out_runs[r0 * pieces:(r0 + n) * pieces, 0:own[3]],
                  src=result[own[0]:own[0] + own[1], 0:own[3]])


@nki.jit
def dsa_dcp_merge_select_kernel(values_hbm, ids_hbm, cp_rank, pools_per_block, source_digest):
    """This rank's share of the global top-k over every DCP rank's local top-k.

    Args:
        values_hbm: ``[CP, R, k]`` float32, rank ``r``'s local selection values for row ``i`` at
            ``[r, i]``, in any order. A pad (id ``-1``, or a value at or below
            ``BOUND_FILL_MARK``) is never selected.
        ids_hbm: ``[CP, R, k]`` int32, the matching local pool ids, ``-1`` where a slot holds no
            pool.
        cp_rank: python int, this rank's index in the KV group.
        pools_per_block: python int, a power of two: pools per ownership block
            (``block_size // index_kpool``).
        source_digest: python int, :data:`SOURCE_DIGEST`; not read.

    Returns:
        ``[R, k]`` int32 in this rank's own slot order: ``ids_hbm[cp_rank, i, j]`` where that
        candidate is among row ``i``'s ``k`` selected (value descending, then global id
        ascending, over the ``CP * k`` candidates that are not pads), ``-1`` in every other slot.
        Nothing is compacted.

    The rows split over the programs of the launch grid as :func:`program_groups` plans them.
    A group breaks ties, on the device's own branch, only when one of its rows has an excess
    tie; under two programs, a row of either program's group (:func:`_any_excess`), so both
    cores take the branch or neither (the module docstring's invariant).
    """
    cp = values_hbm.shape[0]
    rows = values_hbm.shape[1]
    k = values_hbm.shape[2]
    out_hbm = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)
    programs = nl.num_programs(axes=0)
    program = nl.program_id(0)
    for group in program_groups(rows, programs)[program]:
        states = []
        for tile in group[1]:
            states.append(_threshold_tile(values_hbm, ids_hbm, tile[0], tile[1], cp_rank))

        # fori_loop traces the body here, in this iteration, so ``states`` is this group's.
        def break_group_ties(_iteration):
            for state in states:  # noqa: B023
                _break_ties(state, cp, k, cp_rank, pools_per_block)

        nl.fori_loop(0, _any_excess(states, programs, program), break_group_ties)
        if group[0]:
            for state in states:
                _store(state, out_hbm, cp, k, cp_rank)
    return out_hbm


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


def merge_programs() -> int:
    """Programs the merge launches: both cores of an LNC2 pair (:func:`program_groups`)."""
    if lnc_pair():
        return 2
    return 1


def _is_power_of_two(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0 \
        and value & (value - 1) == 0


def _validate(values: Tensor, local_ids: Tensor, cp_rank, pools_per_block) -> tuple[int, int, int]:
    """Host-side validation from shapes, dtypes and python ints; returns ``(cp, rows, k)``."""
    if values.ndim != 3 or tuple(values.shape) != tuple(local_ids.shape):
        raise DcpCandidateMergeError(
            f"values and local_ids must both be [CP, R, k]; got {tuple(values.shape)} and "
            f"{tuple(local_ids.shape)}")
    cp, rows, k = (int(d) for d in values.shape)
    if cp < 1 or rows < 1 or k < 1:
        raise DcpCandidateMergeError(f"CP, R and k must be positive; got {(cp, rows, k)}")
    if values.dtype not in _VALUE_DTYPES or local_ids.dtype not in _ID_DTYPES:
        raise DcpCandidateMergeError(
            f"values must be float32 and local_ids int32; got {values.dtype} and "
            f"{local_ids.dtype}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp:
        raise DcpCandidateMergeError(f"cp_rank must be a python int in [0, {cp}); got {cp_rank!r}")
    if not _is_power_of_two(pools_per_block):
        raise DcpCandidateMergeError(
            f"pools_per_block (block_size // index_kpool) must be a power-of-two python int; "
            f"got {pools_per_block!r}")
    if cp * k > MAX_CANDIDATES:
        raise DcpCandidateMergeError(
            f"DCP top-k merge: CP * k = {cp} * {k} = {cp * k} candidates per row is past the "
            f"MAX_CANDIDATES={MAX_CANDIDATES} whose tiles fit one SBUF partition; a wider merge "
            f"needs the candidate axis tiled (not served)")
    return cp, rows, k


def can_run_dsa_dcp_merge_select(values: Tensor, local_ids: Tensor) -> bool:
    """Whether the NKI kernel serves this merge. ``False`` sends it to the torch oracle.

    Every malformed call has already raised in :func:`_validate`, so the only question left is
    whether NKI is available on ``values``' device. ``local_ids`` is accepted so the gate has the
    same signature as the call it is about, but is not read.
    """
    return can_run_kernel(values)


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(cp: int, rows: int, k: int, programs: int) -> None:
    """Count one kernel dispatch and log it, off the compiled graph."""
    _COUNTERS.nki_dispatch += 1
    _COUNTERS.last_kernel = _kernel_identity_of(dsa_dcp_merge_select_kernel)
    logger.info("[dsa-dcp-merge] kernel=nki cp=%d rows=%d k=%d programs=%d", cp, rows, k,
                programs)


def dsa_dcp_merge_select(values: Tensor, local_ids: Tensor, *, cp_rank: int,
                         pools_per_block: int) -> Tensor:
    """This rank's selected local pool ids from every rank's local selection.

    Args:
        values: ``[CP, R, k]`` float32, every rank's local selection values, rank-major (the
            all-gather's order), in any slot order. Finite at every real candidate; a value at
            or below ``BOUND_FILL_MARK`` marks a pad.
        local_ids: ``[CP, R, k]`` int32, the matching local pool ids, ``-1`` where a slot holds
            no pool; distinct within one rank's row.
        cp_rank: this rank's index in the KV group.
        pools_per_block: ``block_size // index_kpool``, a power of two.

    Returns:
        ``[R, k]`` int32 in ``local_ids[cp_rank]``'s own slot order: the local id where the
        candidate is among the row's ``k`` selected (the module docstring's rule), ``-1`` for
        every slot not selected. Nothing is compacted.

    Raises:
        DcpCandidateMergeError: a shape, dtype, rank or block geometry the merge does not serve.
    """
    cp, rows, k = _validate(values, local_ids, cp_rank, pools_per_block)
    if not can_run_dsa_dcp_merge_select(values, local_ids):
        _count_torch_fallback()
        return dsa_dcp_merge_select_torch_oracle(values, local_ids, cp_rank=cp_rank,
                                                 pools_per_block=pools_per_block)
    programs = merge_programs()
    _record_nki_dispatch(cp, rows, k, programs)
    call = wrap_nki(dsa_dcp_merge_select_kernel)
    if programs == 2:
        call = call[2]
    return call(values.contiguous(), local_ids.contiguous(), int(cp_rank),
                int(pools_per_block), SOURCE_DIGEST)


def dsa_dcp_select_candidates(values: Tensor, local_ids: Tensor, *, dcp_group, cp_rank: int,
                              dcp_size: int, block_size: int,
                              index_kpool: int) -> Tensor:
    """The DCP indexer selection for this rank: every rank's local top-k merged.

    Args:
        values: ``[R, k]`` float32, this rank's local selection values (any order).
        local_ids: ``[R, k]`` int32, the matching local pool ids, ``-1`` where none.
        dcp_group: the DCP KV group coordinator (``_NEURON_DCP_KV_GROUP``); its rank order is
            the ``cp_rank`` order.
        cp_rank: this rank's index in the group, ``tp_rank % dcp_size``.
        dcp_size: the group's size.
        block_size: tokens per ownership block (the KV cache block).
        index_kpool: tokens per indexer pool.

    Returns:
        ``[R, k]`` int32 in ``local_ids``' own slot order: this rank's selected pools as local
        pool ids, ``-1`` for every slot not selected; nothing is compacted
        (:func:`dsa_dcp_merge_select`).

    Raises:
        DcpCandidateMergeError: a group that is not ``dcp_size`` ranks with this rank at
            ``cp_rank``, a block that is not a whole power-of-two number of pools, or a merge
            :func:`dsa_dcp_merge_select` does not serve.
    """
    if values.ndim != 2 or tuple(values.shape) != tuple(local_ids.shape):
        raise DcpCandidateMergeError(
            f"values and local_ids must both be [R, k]; got {tuple(values.shape)} and "
            f"{tuple(local_ids.shape)}")
    if int(dcp_group.world_size) != int(dcp_size) or int(dcp_group.rank_in_group) != int(cp_rank):
        raise DcpCandidateMergeError(
            f"dcp_group must be the {dcp_size}-rank KV group with this rank at cp_rank="
            f"{cp_rank}; got world_size={dcp_group.world_size}, "
            f"rank_in_group={dcp_group.rank_in_group}")
    if index_kpool < 1 or block_size % index_kpool != 0:
        raise DcpCandidateMergeError(
            f"block_size={block_size} must be a whole number of index_kpool={index_kpool} "
            f"pools, so that no pool straddles two owners")
    rows, k = (int(d) for d in values.shape)
    # all_gather on dim 0 concatenates in group-rank order with no layout copy.
    gathered_values = dcp_group.all_gather(values.contiguous(), dim=0)
    gathered_ids = dcp_group.all_gather(local_ids.contiguous(), dim=0)
    return dsa_dcp_merge_select(gathered_values.view(int(dcp_size), rows, k),
                                gathered_ids.view(int(dcp_size), rows, k), cp_rank=int(cp_rank),
                                pools_per_block=int(block_size) // int(index_kpool))


# ---------------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------------


def global_pool_ids(local_ids: Tensor, pools_per_block: int) -> Tensor:
    """``[CP, R, k]`` int64 global pool ids of ``[CP, R, k]`` local ids (rank-major); ``-1``
    where the local id is ``-1``."""
    cp = int(local_ids.shape[0])
    ids = local_ids.to(torch.int64)
    rank = torch.arange(cp, dtype=torch.int64, device=ids.device).view(cp, 1, 1)
    ppb = int(pools_per_block)
    gid = torch.div(ids, ppb, rounding_mode="floor") * (ppb * cp) + rank * ppb \
        + torch.remainder(ids, ppb)
    return torch.where(ids >= 0, gid, torch.full_like(gid, -1))


def dsa_dcp_merge_select_torch_oracle(values: Tensor, local_ids: Tensor, *, cp_rank: int,
                                      pools_per_block: int) -> Tensor:
    """CPU reference and fallback: the rule spelled as a sort.

    Three stable sorts, least significant key first (global id ascending, then value
    descending, then real before pad), order every row's ``CP * k`` candidates; the first
    ``k`` real ones are the selection. The kernel finds a threshold instead and sorts nothing,
    so agreeing with this is not the module agreeing with itself.
    """
    cp, rows, k = (int(d) for d in values.shape)
    gid = global_pool_ids(local_ids, pools_per_block).permute(1, 0, 2).reshape(rows, cp * k)
    vals = values.to(torch.float32).permute(1, 0, 2).reshape(rows, cp * k)
    valid = (gid >= 0) & (vals > BOUND_FILL_MARK)
    order = torch.argsort(gid, dim=1, stable=True)
    order = order.gather(1, torch.argsort(vals.gather(1, order), dim=1, descending=True,
                                          stable=True))
    order = order.gather(1, torch.argsort((~valid).gather(1, order).to(torch.int8), dim=1,
                                          stable=True))
    taken = torch.zeros(rows, cp * k, dtype=torch.bool, device=values.device)
    taken.scatter_(1, order[:, :k], True)
    own = (taken & valid).view(rows, cp, k)[:, cp_rank, :]
    return torch.where(own, local_ids[cp_rank], torch.full_like(local_ids[cp_rank], SENTINEL))
