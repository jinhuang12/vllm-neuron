# SPDX-License-Identifier: Apache-2.0
"""Fused key pooling and Hadamard-128 rotation for the sparse-attention indexer.

One kernel does both halves, so the pooled vector never leaves the on-chip tile
between the reduction and the rotation. Given ``slot_k[n_pools, pool_size, 128]``, a
matching ``slot_score`` and a per-slot additive bias ``ape[pool_size, 128]``::

    w[p, s, d]   = softmax_over_s( slot_score[p, s, d] + ape[s, d] )
    pooled[p, d] = sum_s w[p, s, d] * slot_k[p, s, d]
    out[p, :]    = FWHT_128( pooled[p, :] ) * (1 / sqrt(128))

The softmax is per (pool, channel), not per pool: one independent ``pool_size``-way
softmax for every ``(p, d)`` pair. A whole-vector softmax over the 128 channels is a
plausible-looking different function. ``ape`` is applied inside the softmax, and the
gate arrives already evaluated as ``slot_score``. Both kernels rotate by multiplying
with ``H_128`` on the Tensor Engine, and both build that matrix on chip by running the
butterfly (``_fwht128_inplace``) over ``I_128``; the decode kernels run the same
butterfly on their pooled vectors directly, so every path applies one transform.

The prefill indexer pools a sliding window at every position: position ``p``'s pool is
the ``pool_size`` keys ending at ``p``, clamped at row 0 (``sliding_windows``). With
``windows=True``, ``dsa_kpool_hadamard`` takes ``key[tokens, 128]`` and the gate's score
as they are, and its kernel (``_kpool_hadamard_window_nki``) reads each window from them
in place, into the same tiles the fused kernel loads from ``slot_k``. So it skips the
gather that builds ``slot_k`` and computes the same bits.

Under LNC2 (``NEURON_LOGICAL_NC_CONFIG=2``, the serving setting) every kernel here
launches two programs, one per physical core, and each program serves a contiguous half
of the rows or pools.

The kernels only ever see complete pools; pool formation, slot mapping and the trailing
partial pool belong to the indexer, and the fp8
quantisation, its scale and the cache write belong to the adapter. ``n_pools`` and
``pool_size`` are python ints because they select the tile count and the number of
unrolled slot loads, so a tensor would force a data-dependent trace; the compiled
graph specialises on the exact ``(n_pools, pool_size, head_dim, dtype)`` tuple.
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
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

INDEX_HEAD_DIM = 128
"""The indexer head dimension. The Hadamard path is a 128-point transform and nothing else."""

DEFAULT_POOL_SIZE = 4
"""``index_kpool`` on the target checkpoint's config. Not a hardcoded limit -- see ``_validate``."""

HADAMARD_STAGES: tuple[tuple[int, int], ...] = (
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
    (4, 16),
    (2, 32),
    (1, 64),
)
"""``(groups, stride)`` per butterfly stage, in the origin's order (``kpool_compress.py:38-44``).

``groups * 2 * stride == 128`` for every entry, and the strides are ``2**0 .. 2**6``. Kept as data
rather than as seven call sites so that the sequence can be asserted by the test as a sequence.
"""

HADAMARD_STRIDES: tuple[int, ...] = tuple(stride for _groups, stride in HADAMARD_STAGES)
"""Just the strides, for the device loop to walk.

The device loop cannot walk ``HADAMARD_STAGES`` directly. The NKI front end requires a ``for``
target that is a single variable and refuses one that unpacks a tuple, which
``for _groups, stride in HADAMARD_STAGES:`` does. It is a lowering-time refusal, so the simulator
never sees it and only a compile does::

    error: expecting simple variable
        for _groups, stride in HADAMARD_STAGES:
            ^

Derived here rather than retyped as seven literals so the strides cannot drift from
``HADAMARD_STAGES``, which the test asserts as a whole sequence. This comprehension runs on the host
at import, so NKI never traces it.
"""

HADAMARD_SCALE = 0.08838834764831845
"""``1 / sqrt(128)``, copied as a literal from ``kpool_compress.py:45``.

Copied rather than computed so the shipped constant is bit-identical to the origin's.

This is the correctly rounded value, and not every spelling produces it. ``128 ** -0.5``,
``math.sqrt(1/128)``, ``2 ** -3.5`` and ``math.sqrt(2)/16`` all produce this exact double;
``1.0 / math.sqrt(128)`` produces one ULP lower (``0x1.6a09e667f3bccp-4`` against this value's
``0x1.6a09e667f3bcdp-4``), because the division rounds down. The difference is 1.4e-17 and cannot
move any tolerance here, but it is written down because a test that asserted the division
spelling would fail a correct kernel.
"""

_SUPPORTED_DTYPES = (torch.bfloat16,)
"""``slot_k`` dtypes that take the NKI route. bf16 is the indexer path's dtype."""

_CHUNK_TILES = 16
"""128-row tiles the rotation moves per DMA: 4 KiB per partition at bf16.

Measured on trn2 at the served prefill row counts: 8 and 32 tiles are slower, and
chunks that ramp up and down in size to shorten the pipeline's fill and drain are
slower still, because each extra DMA costs more than the fill it saves.
"""

_GROUP_TILES = 8
"""128-row tiles per PSUM round trip of the rotation.

A group's bf16 transposes fill one 2 KiB PSUM bank and its fp32 products two, so the
next group's transposes and this group's products fit in PSUM's eight banks side by
side. 4 and 16 were slower on trn2.
"""

_POOL_TILE_COLUMNS = 3
"""Pools each partition carries per tile of the fused kernel.

A tile holds 128 partitions of this many consecutive pools, so one DMA moves this many
whole pools per partition and every Vector Engine instruction covers them all. Wider
tiles cost fewer instructions, narrower ones let the next tile's load overlap this one's
compute. Measured on trn2 at the served 1024 and 2048 pools, 3 is within 0.5 us of the
fastest width at both; 1 is slower at both, 2 at 2048 pools and 4 at 1024.
"""


class KpoolHadamardError(ValueError):
    """A malformed call: wrong rank, mismatched shapes, or a pool size that does not divide."""


@dataclass
class _KpoolHadamardDispatchCounters:
    """Per-process record of how this module's two entry points were reached.

    One ``nki_dispatch`` counter serves both the fused kernel and the stage-alone
    rotation, so a test reads the module's total from one place.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _KpoolHadamardDispatchCounters()


def reset_kpool_hadamard_dispatch_counters() -> None:
    """Zero this module's dispatch counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def kpool_hadamard_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset, summed over both entry points."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def kpool_hadamard_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel a seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator
    instead: ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose
    ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is
    ``None``. The wrapped function is reachable at ``__wrapped__`` (the
    ``functools.wraps`` convention) and at ``.func`` (this decorator's own name).
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


def _fwht128_inplace(buf_a, buf_b, head_dim: int):
    """The 7-stage FWHT butterfly along the free axis. Returns the tile holding the result.

    ``buf_a`` holds the input and ``buf_b`` is scratch of the same shape. The free axis may
    carry several ``head_dim``-wide vectors side by side (``(rows, k * head_dim)``); each is
    transformed on its own, because no butterfly block of ``2 * stride <= head_dim`` channels
    crosses a ``head_dim`` boundary.

    Ping-pongs between the two tiles because a stage reads two slices and writes both of
    them: writing into the tile being read would let a later block of the same stage consume
    an already-rotated value. Seven stages is an odd number, so the result is in whichever
    tile this returns, and the caller must use the return value. Each stage is two
    instructions over every block at once, for every block of ``2 * stride`` channels::

        out[lo] = in[lo] + in[hi]
        out[hi] = in[lo] - in[hi]

    The ``1/sqrt(128)`` scale is not applied here; the caller applies it once, after all
    seven stages.
    """
    vectors = buf_a.shape[1] // head_dim
    src, dst = buf_a, buf_b
    for stride in HADAMARD_STRIDES:
        blocks = vectors * (head_dim // (2 * stride))
        src_pairs = src.reshape_dim(1, (blocks, 2, stride))
        dst_pairs = dst.reshape_dim(1, (blocks, 2, stride))
        lo_in = src_pairs.select(2, 0)
        hi_in = src_pairs.select(2, 1)
        nisa.tensor_tensor(dst=dst_pairs.select(2, 0), data1=lo_in, data2=hi_in, op=nl.add)
        nisa.tensor_tensor(dst=dst_pairs.select(2, 1), data1=lo_in, data2=hi_in, op=nl.subtract)
        src, dst = dst, src
    return src


def _fwht128_blocks(buf_a, buf_b, parts: int, rows: int):
    """The butterfly over ``rows`` 128-channel rows on each of ``parts`` partitions.

    ``buf_a`` and ``buf_b`` are ``(parts, rows * INDEX_HEAD_DIM)`` fp32 tiles, row ``j`` of a
    partition in columns ``128 j .. 128 j + 127``; ``buf_a`` holds the input, ``buf_b`` is
    scratch, and the result is in the returned tile. This is the name and signature the
    batched decode ring step calls (``_fwht128_blocks(pooled, scratch, h, 1)``, one row on
    each of ``h`` partitions). :func:`_fwht128_inplace` already transforms every
    ``INDEX_HEAD_DIM``-wide vector of its free axis on its own, two instructions a stage, so
    this is that call on the two tiles viewed at the stated shape: the same instructions on
    the same operands.
    """
    width = rows * INDEX_HEAD_DIM
    return _fwht128_inplace(buf_a.reshape((parts, width)), buf_b.reshape((parts, width)),
                            INDEX_HEAD_DIM)


def _broadcast_row(hbm, rows: int, head_dim: int, row: int):
    """One row of a 2-D HBM buffer replicated across ``rows`` partitions, as an fp32 tile.

    A zero partition stride is what replicates: ``pattern=[[0, rows], ...]`` reads
    the same source row for every partition. ``nisa.tensor_scalar`` cannot serve
    this instead -- when its ``operand0`` is a tile it is a per-partition scalar and
    must carry one entry per partition of ``dst``, so a ``(1, head_dim)`` row is
    refused by the MLIR verifier.

    The DMA lands in a tile of the source dtype and a separate ``tensor_copy`` does the
    widening. The staging is unconditional rather than guarded by a dtype test, because
    comparing a NKI tensor's dtype against a ``nki.language`` dtype object is a
    trace-time equality; always staging is correct for every input dtype at the cost of
    one extra copy of one row per slot, once per kernel.
    """
    out = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    staged = nl.ndarray((rows, head_dim), dtype=hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=staged, src=hbm.ap(pattern=[[0, rows], [1, head_dim]], offset=row * head_dim)
    )
    nisa.tensor_copy(dst=out, src=staged)
    return out


def _program_share(n_items: int):
    """``(first, owned)``: the contiguous run of ``n_items`` rows (pools) this program serves.

    The launch's programs split the rows into ``ceil(n_items / programs)``-long runs in
    program order, the last one shorter; every program reaches here with ``owned >= 1``
    because the host launches more than one program only for ``n_items >= 2``.
    """
    programs = nl.num_programs(axes=0)
    share = (n_items + programs - 1) // programs
    first = nl.program_id(axis=0) * share
    return first, min(share, n_items - first)


def _row_blocks(first: int, owned: int, pmax: int, columns_cap: int):
    """The ``owned`` rows from ``first`` as blocks: a list of ``(row0, partitions, columns)``.

    Partition ``p`` of a block holds the ``columns`` consecutive rows from
    ``row0 + p * columns``, so the block is ``partitions * columns`` consecutive rows and its
    DMA reads ``columns`` whole rows per partition in one contiguous run. Whole 128-row
    tiles are grouped ``columns_cap`` at a time, the last group shorter; the
    ``owned % pmax`` rows left over form one block of that many partitions and one column.
    """
    blocks = []
    row = first
    whole_tiles = owned // pmax
    for start in range(0, whole_tiles, columns_cap):
        columns = min(columns_cap, whole_tiles - start)
        blocks.append((row, pmax, columns))
        row = row + pmax * columns
    if owned % pmax > 0:
        blocks.append((row, owned % pmax, 1))
    return blocks


def _block_rows(hbm, block, unit):
    """``hbm`` viewed as one block of ``_row_blocks``: ``[partitions, columns, *unit]``.

    ``unit`` is the shape of one of the block's rows in ``hbm``'s elements: ``(head_dim,)`` for
    a rotated row, ``(pool_size, head_dim)`` for a pool of a flattened
    ``[n_pools * pool_size, head_dim]`` buffer.
    """
    unit_elements = 1
    for extent in unit:
        unit_elements = unit_elements * extent
    pattern = [[block[2] * unit_elements, block[1]], [unit_elements, block[2]]]
    stride = unit_elements
    for extent in unit:
        stride = stride // extent
        pattern.append([stride, extent])
    return hbm.ap(pattern=pattern, offset=block[0] * unit_elements)


def _slot_fold(values, op, pool_size: int):
    """``op`` folded over the slot axis of ``values[rows, columns, pool_size, head_dim]``.

    In slot order, ``op(...op(op(v[0], v[1]), v[2])..., v[pool_size - 1])``: the order the
    reference sums in, so an ``add`` fold rounds the same way. Each step writes its own slot of
    one fresh tile, so no step's destination is one of its sources. Returns a
    ``[rows, columns, head_dim]`` view of the last step, or of slot 0 when there is one slot.
    """
    if pool_size == 1:
        return values[:, :, 0, :]
    rows, columns, _, head_dim = values.shape
    steps = nl.ndarray((rows, pool_size - 1, columns, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=steps[:, 0, :, :], data1=values[:, :, 0, :], data2=values[:, :, 1, :], op=op)
    for slot in range(2, pool_size):
        nisa.tensor_tensor(
            dst=steps[:, slot - 1, :, :], data1=steps[:, slot - 2, :, :], data2=values[:, :, slot, :],
            op=op,
        )
    return steps[:, pool_size - 2, :, :]


def _load_slots(tile, hbm, block):
    """One block's pools of a flattened ``[n_pools * pool_size, head_dim]`` buffer, into ``tile``.

    ``tile`` is the block's ``[partitions, columns, pool_size, head_dim]`` tile; partition ``p``
    gets the ``columns`` whole pools from ``row0 + p * columns``, one contiguous run.
    """
    nisa.dma_copy(dst=tile, src=_block_rows(hbm, block, (tile.shape[2], tile.shape[3])))


def _window_head(block, pool_size: int) -> int:
    """How many leading partitions of ``block`` hold pools whose windows clamp at row 0.

    Pool ``p``'s window starts at row ``p - (pool_size - 1)``; below ``p = pool_size - 1``
    that start is clamped to row 0 (``sliding_windows``), so the window is rows
    ``0 .. pool_size - 1``. A partition holds ``columns`` consecutive pools, so these pools
    fill the partitions whose first pool is below ``pool_size - 1``. ``_window_head_fits``
    says whether every pool on those partitions has that same window.
    """
    row0, partitions, columns = block
    short = pool_size - 1 - row0
    if short <= 0:
        return 0
    return min(partitions, (short + columns - 1) // columns)


def _window_head_fits(block, pool_size: int) -> bool:
    """Whether every pool on ``block``'s head partitions has the window ``0 .. pool_size - 1``.

    True when the head's last pool is at most ``pool_size - 1``: the clamped pools and pool
    ``pool_size - 1`` read that window, and a later pool does not.
    """
    head = _window_head(block, pool_size)
    return head == 0 or block[0] + head * block[2] <= pool_size


def _load_windows(tile, hbm, block):
    """One block's sliding windows of ``hbm[tokens, head_dim]``, read in place into ``tile``.

    ``tile`` is the block's ``[partitions, columns, pool_size, head_dim]`` tile, the one
    ``_load_slots`` fills from the gathered windows, and it gets the same values: slot ``s`` of
    pool ``p`` is row ``max(p - (pool_size - 1), 0) + s``. An unclamped window is
    ``pool_size`` consecutive rows ending at row ``p``, and pool ``p + 1``'s window starts one
    row later, so one access pattern reads the block: pool and slot strides of one row (the
    windows overlap), and a partition stride of ``columns`` rows. The head partitions
    (``_window_head``) all read rows ``0 .. pool_size - 1``, by a second pattern with zero
    partition and pool strides. Both patterns stay inside rows ``0 .. row0 + partitions *
    columns - 1``.
    """
    row0, partitions, columns = block
    pool_size, head_dim = tile.shape[2], tile.shape[3]
    kernel_assert(
        _window_head_fits(block, pool_size),
        f"block {block}: a head partition holds a pool past {pool_size - 1}, whose window "
        f"is not rows 0 .. {pool_size - 1}",
    )
    head = _window_head(block, pool_size)
    if head > 0:
        nisa.dma_copy(
            dst=tile[0:head, :, :, :],
            src=hbm.ap(pattern=[[0, head], [0, columns], [head_dim, pool_size], [1, head_dim]],
                       offset=0),
        )
    if head < partitions:
        first = row0 + head * columns
        nisa.dma_copy(
            dst=tile[head:partitions, :, :, :],
            src=hbm.ap(
                pattern=[[columns * head_dim, partitions - head], [head_dim, columns],
                         [head_dim, pool_size], [1, head_dim]],
                offset=(first - (pool_size - 1)) * head_dim,
            ),
        )


def _pool_block(load, key_hbm, score_hbm, out_hbm, biases, transform, block, pool_size: int):
    """Pool, rotate, scale and store one block of ``_row_blocks`` over the pools.

    ``load`` fills the block's key and score tiles from ``key_hbm`` and ``score_hbm``:
    ``_load_slots`` from gathered pools, ``_load_windows`` from the keys in place.
    ``biases`` holds one ``(pmax, head_dim)`` fp32 tile of ``ape`` per slot and ``transform`` is
    the fp32 ``H_128`` of ``_transform_matrix``. Every Vector Engine step is one instruction
    over the whole block, so a block of ``columns`` pools per partition costs the instructions
    of one pool. The softmax is the reference's, step for step: the max, the shift, ``exp`` on
    the Scalar Engine, the slot sums in slot order. Two steps move off the Vector Engine, which
    bounds this kernel: the reciprocal of the sum runs on the Scalar Engine, and the rotation
    is the rotation kernel's, a multiply by ``H`` on the Tensor Engine with the scale applied
    once on the way out of PSUM.
    """
    rows = block[1]
    columns = block[2]
    head_dim = key_hbm.shape[1]
    shape = (rows, columns, pool_size, head_dim)
    score = nl.ndarray(shape, dtype=score_hbm.dtype, buffer=nl.sbuf)
    load(score, score_hbm, block)
    key = nl.ndarray(shape, dtype=key_hbm.dtype, buffer=nl.sbuf)
    load(key, key_hbm, block)

    # Softmax over the slots, per (pool, channel), shifted by the max for stability.
    total = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    for slot in range(pool_size):
        nisa.tensor_tensor(
            dst=total[:, :, slot, :], data1=score[:, :, slot, :],
            data2=biases[slot][0:rows, :].expand_dim(1).broadcast(1, columns), op=nl.add,
        )
    peak = _slot_fold(total, nl.maximum, pool_size)
    shifted = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=shifted, data1=total, data2=peak.expand_dim(2).broadcast(2, pool_size), op=nl.subtract
    )
    weight = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=weight, op=nl.exp, data=shifted)
    denom = _slot_fold(weight, nl.add, pool_size)
    weighted = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=weighted, data1=weight, data2=key, op=nl.multiply)
    acc = _slot_fold(weighted, nl.add, pool_size)

    # Reciprocal then multiply: no engine divides one tile by another. Measured on trn2 at
    # every fp32 from 1 to 4, a 4-slot sum's whole range, the Scalar Engine's reciprocal is
    # within 1.2e-5 relative of the exact one; the Vector Engine's is correctly rounded but
    # takes about 6 cycles an element, and this kernel is bound by the Vector Engine.
    inv = nl.ndarray((rows, columns, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.reciprocal, data=denom)
    pooled = nl.ndarray((rows, columns, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)

    # One PSUM group for the whole block; the Scalar Engine moves the transposes out of PSUM,
    # because the Vector Engine is this kernel's busy one.
    group = _tile_groups([block], columns)[0]
    transposed = _transpose_group(pooled, block, group, nisa.scalar_engine)
    result = nl.ndarray((rows, columns, head_dim), dtype=key_hbm.dtype, buffer=nl.sbuf)
    _rotate_group(transposed, result, block, group, transform)
    nisa.dma_copy(dst=_block_rows(out_hbm, block, (head_dim,)), src=result)


def _transform_matrix(head_dim: int, dtype):
    """``H_128`` in ``dtype``, built on chip as the butterfly applied to ``I_128``.

    Row ``r`` of the butterfly's output on ``I`` is the transform of the unit vector ``e_r``,
    which is row ``r`` of ``H``; ``H`` is symmetric, so ``x @ H`` is the butterfly applied to
    every row of ``x``. Every entry is ``+-1``, exact in any float dtype.
    """
    identity = nl.shared_identity_matrix(n=head_dim, dtype=nl.float32)
    halves = nl.ndarray((head_dim, 2 * head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=halves[:, 0:head_dim], src=identity)
    rows = _fwht128_inplace(halves[:, 0:head_dim], halves[:, head_dim:2 * head_dim], head_dim)
    transform = nl.ndarray((head_dim, head_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=transform, src=rows)
    return transform


def _tile_groups(blocks, group_cap: int):
    """The 128-row tiles of every block, ``group_cap`` at a time.

    A list of ``(block_index, first_tile, tiles, opens_block, closes_block)`` in block order:
    ``block_index`` indexes ``blocks``, ``first_tile`` is the group's first tile within that
    block, and the two flags mark a block's first and last group.
    """
    groups = []
    for index in range(len(blocks)):
        columns = blocks[index][2]
        for start in range(0, columns, group_cap):
            tiles = min(group_cap, columns - start)
            groups.append((index, start, tiles, start == 0, start + tiles == columns))
    return groups


def _load_block(x_hbm, block):
    """Allocate one block's input and output tiles and start the input's DMA."""
    shape = (block[1], block[2], x_hbm.shape[1])
    rows_in = nl.ndarray(shape, dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows_in, src=_block_rows(x_hbm, block, (x_hbm.shape[1],)))
    rows_out = nl.ndarray(shape, dtype=x_hbm.dtype, buffer=nl.sbuf)
    return rows_in, rows_out


def _transpose_group(rows_in, block, group, engine):
    """One group's tiles transposed on the Tensor Engine: ``[head_dim, tiles, pmax]`` in SBUF.

    Tensor Engine transposes are bit-exact on trn2. A partial block's tiles fill the first
    ``partitions`` columns only, and only those are copied out of PSUM, by ``engine``: the
    Vector or the Scalar Engine, whichever the caller's other work leaves idle. Both copy a
    same-dtype tile exactly.
    """
    partitions = block[1]
    first_tile = group[1]
    tiles = group[2]
    head_dim = rows_in.shape[2]
    pmax = nl.tile_size.pmax
    staged = nl.ndarray((head_dim, tiles, pmax), dtype=rows_in.dtype, buffer=nl.psum)
    for k in range(tiles):
        nisa.nc_transpose(
            dst=staged[:, k, 0:partitions], data=rows_in[:, first_tile + k, :],
            engine=nisa.tensor_engine,
        )
    transposed = nl.ndarray((head_dim, tiles, pmax), dtype=rows_in.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=transposed[:, :, 0:partitions], src=staged[:, :, 0:partitions], engine=engine
    )
    return transposed


def _rotate_group(transposed, rows_out, block, group, transform):
    """``tile @ H * (1 / sqrt(128))`` for one group's tiles, into the block's output tile.

    The products accumulate in fp32 PSUM; the Scalar Engine applies the scale while it moves
    them to SBUF in the output dtype, so the scale is applied once and rounded once.
    """
    partitions = block[1]
    first_tile = group[1]
    tiles = group[2]
    head_dim = rows_out.shape[2]
    product = nl.ndarray((nl.tile_size.pmax, tiles, head_dim), dtype=nl.float32, buffer=nl.psum)
    for k in range(tiles):
        nisa.nc_matmul(
            dst=product[0:partitions, k, :], stationary=transposed[:, k, 0:partitions],
            moving=transform,
        )
    nisa.activation(
        dst=rows_out[:, first_tile:first_tile + tiles, :], op=nl.copy,
        data=product[0:partitions, :, :], scale=HADAMARD_SCALE,
    )


# ---------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------


def _kpool_hadamard_body(load, key_hbm, score_hbm, ape_hbm, n_pools, pool_size):
    """The pooling kernels' shared body: ``load`` fills each block's key and score tiles."""
    head_dim = key_hbm.shape[1]
    out_hbm = nl.ndarray((n_pools, head_dim), dtype=key_hbm.dtype, buffer=nl.shared_hbm)
    pmax = nl.tile_size.pmax
    first, owned = _program_share(n_pools)
    biases = []
    for slot in range(pool_size):
        biases.append(_broadcast_row(ape_hbm, pmax, head_dim, slot))
    transform = _transform_matrix(head_dim, nl.float32)
    for block in _row_blocks(first, owned, pmax, _POOL_TILE_COLUMNS):
        _pool_block(load, key_hbm, score_hbm, out_hbm, biases, transform, block, pool_size)
    return out_hbm


@nki.jit
def _kpool_hadamard_nki(slot_k_hbm, slot_score_hbm, ape_hbm, n_pools, pool_size):
    """Fused per-(pool, channel) softmax-weighted pooling and Hadamard-128 rotation.

    Args:
        slot_k_hbm: ``[n_pools * pool_size, head_dim]`` -- ``slot_k`` flattened so that a pool and
            a slot together address one row. bf16 on the shipped path.
        slot_score_hbm: ``[n_pools * pool_size, head_dim]`` -- the gate's per-token score,
            flattened the same way. bf16 as served; any float dtype is read as is.
        ape_hbm: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        n_pools: pools in the batch. A compile-time constant.
        pool_size: tokens per pool. A compile-time constant.

    Returns:
        ``[n_pools, head_dim]`` in ``slot_k_hbm``'s dtype.

    The pool axis is the partition axis and the head dimension is the free axis, which is
    what keeps the whole reduction elementwise: the softmax runs over the slot axis of one
    tile, so each step is one ``tensor_tensor`` and nothing reduces across partitions. Each
    program takes its run of pools (``_program_share``) as blocks of up to
    ``_POOL_TILE_COLUMNS`` consecutive pools per partition (``_row_blocks``); a block of fewer
    than 128 partitions is narrowed rather than masked, so no padded row can reach the
    output. The softmax is Vector Engine element passes, which bound the kernel, so a wider
    block divides the per-instruction cost and not the element count; the rotation runs on
    the Tensor Engine with the rotation kernel's ``H`` in fp32, so the two kernels apply one
    transform.
    """
    return _kpool_hadamard_body(_load_slots, slot_k_hbm, slot_score_hbm, ape_hbm, n_pools,
                                pool_size)


@nki.jit
def _kpool_hadamard_window_nki(key_hbm, score_hbm, ape_hbm, n_pools, pool_size):
    """``_kpool_hadamard_nki`` over every position's sliding window, read in place.

    Args:
        key_hbm: ``[n_pools, head_dim]`` -- one key per position; pool ``p`` is the window of
            ``pool_size`` rows ending at row ``p``, clamped at row 0. bf16 on the shipped path.
        score_hbm: ``[n_pools, head_dim]`` -- the gate's per-token score, one row per position.
        ape_hbm: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        n_pools: positions, one pool each. A compile-time constant, at least ``pool_size``.
        pool_size: tokens per pool. A compile-time constant.

    Returns:
        ``[n_pools, head_dim]`` in ``key_hbm``'s dtype.

    The fused kernel's programs, blocks and steps, with each block's key and score tiles read
    from the windows in place (``_load_windows``) instead of from a gathered copy: the tiles
    hold the same values, so the output is the same bits.
    """
    return _kpool_hadamard_body(_load_windows, key_hbm, score_hbm, ape_hbm, n_pools, pool_size)


@nki.jit
def _hadamard128_nki(x_hbm, n_rows):
    """The rotation stage alone: ``FWHT_128(row) * (1 / sqrt(128))`` for every row.

    Args:
        x_hbm: ``[n_rows, head_dim]`` -- the rows to rotate.
        n_rows: rows in the batch. A compile-time constant.

    Returns:
        ``[n_rows, head_dim]`` in ``x_hbm``'s dtype.

    A multiply by ``H_128`` on the Tensor Engine, with ``H`` built on chip from the butterfly
    the fused kernel uses (``_transform_matrix``), so the two kernels apply one transform. The
    products are exact and accumulate in fp32, and the result is scaled and rounded once, as
    the butterfly's is; the two differ only in the order of the fp32 additions.

    Rows sit on partitions as they arrive and the transform contracts over the head
    dimension, so each 128-row tile is transposed on the Tensor Engine first and then
    multiplied: two passes of the array per tile, which is what bounds this kernel. Each
    program takes its run of rows (``_program_share``) in blocks of up to ``_CHUNK_TILES``
    tiles (``_row_blocks``), one DMA each way per block, and works through them in groups of
    ``_GROUP_TILES`` tiles per PSUM round trip. The program order is software-pipelined: the
    next group's transposes are issued before this group's multiplies, so the array does not
    wait while the Vector Engine moves a group out of PSUM, and the next block's load is
    issued when the previous block's first group starts.
    """
    head_dim = x_hbm.shape[1]
    out_hbm = nl.ndarray((n_rows, head_dim), dtype=x_hbm.dtype, buffer=nl.shared_hbm)
    first, owned = _program_share(n_rows)
    blocks = _row_blocks(first, owned, nl.tile_size.pmax, _CHUNK_TILES)
    groups = _tile_groups(blocks, _GROUP_TILES)
    transform = _transform_matrix(head_dim, x_hbm.dtype)

    rows_in = []
    rows_out = []
    for index in range(min(2, len(blocks))):
        tiles = _load_block(x_hbm, blocks[index])
        rows_in.append(tiles[0])
        rows_out.append(tiles[1])
    pending = _transpose_group(rows_in[0], blocks[0], groups[0], nisa.vector_engine)
    for index in range(len(groups)):
        group = groups[index]
        ready = pending
        if index + 1 < len(groups):
            following = groups[index + 1]
            if following[3] and following[0] + 1 < len(blocks):
                tiles = _load_block(x_hbm, blocks[following[0] + 1])
                rows_in.append(tiles[0])
                rows_out.append(tiles[1])
            pending = _transpose_group(
                rows_in[following[0]], blocks[following[0]], following, nisa.vector_engine
            )
        _rotate_group(ready, rows_out[group[0]], blocks[group[0]], group, transform)
        if group[4]:
            nisa.dma_copy(
                dst=_block_rows(out_hbm, blocks[group[0]], (head_dim,)), src=rows_out[group[0]]
            )
    return out_hbm


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(entry: str, n_pools: int, pool_size: int, head_dim: int) -> None:
    """Record which kernel a seam dispatched, and log it, off the compiled graph.

    The dispatch branches are traced under ``fullgraph=True``, so a host call Dynamo
    refuses would break them. A folded helper may take ints, strings and dtypes only:
    Dynamo converts every non-tensor argument into a python constant at trace time,
    and an ``@nki.jit`` kernel is a frozen dataclass it cannot reconstruct. The entry
    point therefore arrives as a ``str`` and each kernel is read as a module global.
    """
    kernel = {"fused": _kpool_hadamard_nki, "window": _kpool_hadamard_window_nki}.get(
        entry, _hadamard128_nki
    )
    _COUNTERS.last_kernel = _kernel_identity_of(kernel)
    logger.info(
        "[dsa-kpool-hadamard] kernel=nki entry=%s n_pools=%d pool_size=%d head_dim=%d",
        entry,
        n_pools,
        pool_size,
        head_dim,
    )


def _programs(n_items: int) -> int:
    """Programs to launch over ``n_items`` rows or pools: one per physical core under LNC2.

    ``NEURON_LOGICAL_NC_CONFIG=2`` is the serving setting, one logical core made of two
    physical cores, and the variable other DSA kernels read for the same choice
    (``decode_batch._programs``). A single row cannot be split, so it takes one program.
    """
    if lnc_pair() and n_items >= 2:
        return 2
    return 1


def _validate(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> tuple[int, int, int]:
    """Host-side shape and dtype validation. Returns ``(n_pools, pool_size, head_dim)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor value, so nothing here
    forces a device-to-host synchronisation or a data-dependent trace.
    """
    if slot_k.ndim != 3:
        raise KpoolHadamardError(
            f"slot_k must be 3-D [n_pools, pool_size, head_dim]; got shape {tuple(slot_k.shape)}"
        )
    if tuple(slot_score.shape) != tuple(slot_k.shape):
        raise KpoolHadamardError(
            f"slot_score must match slot_k; got {tuple(slot_score.shape)} against "
            f"{tuple(slot_k.shape)}"
        )
    n_pools, pool_size, head_dim = (int(d) for d in slot_k.shape)
    if ape.ndim != 2 or tuple(ape.shape) != (pool_size, head_dim):
        raise KpoolHadamardError(
            f"ape must be [pool_size, head_dim] = {(pool_size, head_dim)}; got "
            f"{tuple(ape.shape)}"
        )
    if n_pools <= 0:
        raise KpoolHadamardError(f"n_pools must be positive; got {n_pools}")
    if pool_size <= 0:
        raise KpoolHadamardError(f"pool_size must be positive; got {pool_size}")
    if head_dim != INDEX_HEAD_DIM:
        raise KpoolHadamardError(
            f"the Hadamard path is a {INDEX_HEAD_DIM}-point transform; got head_dim {head_dim}"
        )
    return n_pools, pool_size, head_dim


def can_run_dsa_kpool_hadamard(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> bool:
    """Whether the fused NKI kernel serves a slot-form call. ``False`` sends it to the torch path."""
    if not can_run_kernel():
        return False
    if slot_k.dtype not in _SUPPORTED_DTYPES:
        return False
    if slot_k.ndim != 3 or tuple(slot_score.shape) != tuple(slot_k.shape):
        return False
    if int(slot_k.shape[2]) != INDEX_HEAD_DIM:
        return False
    return ape.ndim == 2 and tuple(ape.shape) == tuple(slot_k.shape[1:])


def can_run_dsa_hadamard128(x: Tensor) -> bool:
    """Whether the stage-alone NKI kernel serves this call."""
    if not can_run_kernel():
        return False
    return x.ndim == 2 and int(x.shape[1]) == INDEX_HEAD_DIM


def sliding_windows(tokens: int, pool_size: int, device=None) -> Tensor:
    """``[tokens, pool_size]`` row indices: position ``p``'s pool, clamped at row 0.

    Row ``p`` is ``max(p - (pool_size - 1), 0) + [0 .. pool_size - 1]``: the ``pool_size``
    positions ending at ``p``, and for the first ``pool_size - 1`` positions the first
    ``pool_size`` rows. ``key[sliding_windows(...)]`` is the ``slot_k`` of every position.
    """
    positions = torch.arange(tokens, device=device)
    offsets = torch.arange(pool_size, device=device)
    return (positions - (pool_size - 1)).clamp_min(0)[:, None] + offsets[None, :]


def _validate_window(key: Tensor, gate_score: Tensor, ape: Tensor) -> tuple[int, int, int]:
    """``_validate`` for the window entry. Returns ``(tokens, pool_size, head_dim)``."""
    if key.ndim != 2:
        raise KpoolHadamardError(
            f"key must be 2-D [tokens, head_dim]; got shape {tuple(key.shape)}"
        )
    if tuple(gate_score.shape) != tuple(key.shape):
        raise KpoolHadamardError(
            f"gate_score must match key; got {tuple(gate_score.shape)} against "
            f"{tuple(key.shape)}"
        )
    tokens, head_dim = int(key.shape[0]), int(key.shape[1])
    if head_dim != INDEX_HEAD_DIM:
        raise KpoolHadamardError(
            f"the Hadamard path is a {INDEX_HEAD_DIM}-point transform; got head_dim {head_dim}"
        )
    if ape.ndim != 2 or int(ape.shape[1]) != head_dim or int(ape.shape[0]) <= 0:
        raise KpoolHadamardError(
            f"ape must be [pool_size, head_dim] with pool_size >= 1 and head_dim {head_dim}; "
            f"got {tuple(ape.shape)}"
        )
    pool_size = int(ape.shape[0])
    if tokens < pool_size:
        raise KpoolHadamardError(f"no window of {pool_size} rows fits in {tokens} token(s)")
    return tokens, pool_size, head_dim


def _window_blocks_fit(tokens: int, pool_size: int, programs: int) -> bool:
    """Whether every block the window kernel reads has a head ``_load_windows`` can read.

    The blocks are the kernel's own: each program's share (``_program_share``) as
    ``_row_blocks``, computed here from the same ints.
    """
    share = (tokens + programs - 1) // programs
    for program in range(programs):
        first = program * share
        owned = min(share, tokens - first)
        for block in _row_blocks(first, owned, nl.tile_size.pmax, _POOL_TILE_COLUMNS):
            if not _window_head_fits(block, pool_size):
                return False
    return True


def _can_run_window_kernel(key: Tensor, ape: Tensor) -> bool:
    """Whether the window kernel serves a ``windows=True`` call. ``False`` gathers the windows
    and takes the slot form's routes."""
    if not can_run_kernel():
        return False
    if key.dtype not in _SUPPORTED_DTYPES:
        return False
    tokens, pool_size = int(key.shape[0]), int(ape.shape[0])
    return _window_blocks_fit(tokens, pool_size, _programs(tokens))


def dsa_kpool_hadamard(
    slot_k: Tensor, slot_score: Tensor, ape: Tensor, *, windows: bool = False
) -> Tensor:
    """Pool ``pool_size`` keys into one per pool and rotate the result.

    Args:
        slot_k: ``[n_pools, pool_size, head_dim]`` -- raw per-token indexer keys, one complete pool
            per row of the first axis. ``bfloat16`` takes the NKI route; any other dtype is served
            by the torch path. With ``windows``, ``[tokens, head_dim]``: one key per position.
        slot_score: the gate's per-token score, in ``slot_k``'s form.
        ape: ``[pool_size, head_dim]`` -- the per-slot additive position bias.
        windows: pool every position's sliding window: pool ``p`` is the ``pool_size`` rows of
            ``slot_k`` and ``slot_score`` ending at row ``p``, clamped at row 0
            (``sliding_windows``). The result is the same bits as the slot form on the gathered
            windows, ``slot_k[w]`` and ``slot_score[w]`` with ``w = sliding_windows(tokens,
            pool_size)``: the NKI route reads each window in place into the tiles the slot form's
            kernel loads (``_kpool_hadamard_window_nki``), and the other routes gather.

    Returns:
        ``[n_pools, head_dim]`` (``[tokens, head_dim]`` with ``windows``) in ``slot_k``'s dtype: the
        pooled, rotated key per pool. No fp8 output and no scale -- the adapter owns that half.

    Raises:
        KpoolHadamardError: for a malformed call -- a ``slot_k`` of the wrong rank, a ``slot_score``
            that does not match it, an ``ape`` of the wrong shape, a head dimension that is not
            128, or with ``windows`` fewer tokens than one pool.
    """
    if windows:
        return _pool_windows(slot_k, slot_score, ape)
    return _pool_slots(slot_k, slot_score, ape)


def _pool_windows(key: Tensor, gate_score: Tensor, ape: Tensor) -> Tensor:
    """``dsa_kpool_hadamard(..., windows=True)``."""
    tokens, pool_size, head_dim = _validate_window(key, gate_score, ape)

    if not _can_run_window_kernel(key, ape):
        windows = sliding_windows(tokens, pool_size, key.device)
        return _pool_slots(key[windows], gate_score[windows], ape)

    _count_nki_dispatch()
    _record_nki_dispatch("window", tokens, pool_size, head_dim)
    return wrap_nki(_kpool_hadamard_window_nki)[_programs(tokens)](
        key.contiguous(), gate_score.contiguous(), ape.contiguous(), tokens, pool_size
    )


def _pool_slots(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> Tensor:
    """``dsa_kpool_hadamard`` on complete pools."""
    n_pools, pool_size, head_dim = _validate(slot_k, slot_score, ape)

    if not can_run_dsa_kpool_hadamard(slot_k, slot_score, ape):
        return _dsa_kpool_hadamard_torch(slot_k, slot_score, ape)

    flat_k = slot_k.reshape(n_pools * pool_size, head_dim).contiguous()
    flat_score = slot_score.reshape(n_pools * pool_size, head_dim).contiguous()

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter store
    # inside the trace becomes a value guard that fails on the first call after warmup.
    _record_nki_dispatch("fused", n_pools, pool_size, head_dim)
    return wrap_nki(_kpool_hadamard_nki)[_programs(n_pools)](
        flat_k, flat_score, ape.contiguous(), n_pools, pool_size
    )


def dsa_hadamard128(x: Tensor) -> Tensor:
    """``FWHT_128(row) / sqrt(128)`` for every row.

    Args:
        x: ``[n_rows, head_dim]`` with ``head_dim == 128``.

    Returns:
        ``[n_rows, head_dim]`` in ``x``'s dtype.

    Raises:
        KpoolHadamardError: if ``x`` is not 2-D or its head dimension is not 128.
    """
    if x.ndim != 2:
        raise KpoolHadamardError(f"x must be 2-D [n_rows, head_dim]; got shape {tuple(x.shape)}")
    n_rows, head_dim = int(x.shape[0]), int(x.shape[1])
    if head_dim != INDEX_HEAD_DIM:
        raise KpoolHadamardError(
            f"the Hadamard path is a {INDEX_HEAD_DIM}-point transform; got head_dim {head_dim}"
        )
    if n_rows <= 0:
        raise KpoolHadamardError(f"n_rows must be positive; got {n_rows}")

    if not can_run_dsa_hadamard128(x):
        return _dsa_hadamard128_torch(x)

    _count_nki_dispatch()
    _record_nki_dispatch("stage", n_rows, 1, head_dim)
    return wrap_nki(_hadamard128_nki)[_programs(n_rows)](x.contiguous(), n_rows)


# ---------------------------------------------------------------------------------------------
# Torch reference paths
# ---------------------------------------------------------------------------------------------


def hadamard_matrix(head_dim: int = INDEX_HEAD_DIM, dtype=torch.float32, device=None) -> Tensor:
    """The unnormalised Sylvester ``H_n``, built by doubling. ``H @ H.T == n * I``.

    Built rather than transcribed so that no 128x128 literal has to be trusted.
    ``device`` follows the activation's when a fallback builds it, so a trace on one
    device never meets a second one.
    """
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise KpoolHadamardError(f"head_dim must be a positive power of two; got {head_dim}")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < head_dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h


def _dsa_kpool_hadamard_torch(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> Tensor:
    """Unfused torch composition -- pool, then rotate. The CPU reference, and the fallback path.

    Deliberately unfused, so that it is an independent check on the fused kernel.
    ``dim=1`` is the slot axis, which is what makes the softmax per
    ``(pool, channel)``; a ``dim=-1`` here would be the whole-vector softmax this
    module is not.
    """
    _count_torch_fallback()
    weights = torch.softmax(slot_score.float() + ape.float().unsqueeze(0), dim=1)
    pooled = (weights * slot_k.float()).sum(dim=1)
    rotated = pooled @ hadamard_matrix(int(slot_k.shape[2]), device=slot_k.device).t()
    return (rotated * HADAMARD_SCALE).to(slot_k.dtype)


def _dsa_hadamard128_torch(x: Tensor) -> Tensor:
    """The rotation alone, in torch. The CPU reference, and the fallback path."""
    _count_torch_fallback()
    rotated = x.float() @ hadamard_matrix(int(x.shape[1]), device=x.device).t()
    return (rotated * HADAMARD_SCALE).to(x.dtype)
