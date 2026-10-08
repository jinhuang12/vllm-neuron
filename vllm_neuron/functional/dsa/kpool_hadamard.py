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
gate arrives already evaluated as ``slot_score``. The fused kernel rotates with the
butterfly on the Vector Engine. The stage-alone rotation multiplies by ``H_128`` on
the Tensor Engine instead; it builds that matrix on chip by running the same
butterfly over ``I_128``, so both kernels apply one transform.

Under LNC2 (``NEURON_LOGICAL_NC_CONFIG=2``, the serving setting) the stage-alone
rotation launches two programs, one per physical core, and each program serves a
contiguous half of the rows.

The kernel only ever sees complete pools; pool formation, the sliding window, slot
mapping and the trailing partial pool belong to the indexer, and the fp8
quantisation, its scale and the cache write belong to the adapter. ``n_pools`` and
``pool_size`` are python ints because they select the tile count and the number of
unrolled slot loads, so a tensor would force a data-dependent trace; the compiled
graph specialises on the exact ``(n_pools, pool_size, head_dim, dtype)`` tuple.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

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


def _load_fp32(hbm, rows: int, head_dim: int, row_stride: int, offset: int):
    """A ``(rows, head_dim)`` fp32 tile from a 2-D HBM buffer, widened on the way in.

    ``row_stride`` is in elements, so a caller reading slot ``s`` of a flattened
    ``[n_pools * pool_size, head_dim]`` buffer passes ``pool_size * head_dim``.

    The DMA lands in a tile of the source dtype and a separate ``tensor_copy`` does
    the widening. The staging is unconditional rather than guarded by a
    ``hbm.dtype == nl.float32`` test, because comparing a NKI tensor's dtype against
    a ``nki.language`` dtype object is a trace-time equality; always staging is
    correct for every input dtype at the cost of one extra copy when the source is
    already fp32.
    """
    out = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    staged = nl.ndarray((rows, head_dim), dtype=hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=staged, src=hbm.ap(pattern=[[row_stride, rows], [1, head_dim]], offset=offset))
    nisa.tensor_copy(dst=out, src=staged)
    return out


def _broadcast_row(hbm, rows: int, head_dim: int, row: int):
    """One row of a 2-D HBM buffer replicated across ``rows`` partitions, as an fp32 tile.

    A zero partition stride is what replicates: ``pattern=[[0, rows], ...]`` reads
    the same source row for every partition. ``nisa.tensor_scalar`` cannot serve
    this instead -- when its ``operand0`` is a tile it is a per-partition scalar and
    must carry one entry per partition of ``dst``, so a ``(1, head_dim)`` row is
    refused by the MLIR verifier.

    Stages through the source dtype unconditionally, for the reason ``_load_fp32``
    gives.
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


def _transpose_group(rows_in, block, group):
    """One group's tiles transposed on the Tensor Engine: ``[head_dim, tiles, pmax]`` in SBUF.

    Tensor Engine transposes are bit-exact on trn2. A partial block's tiles fill the first
    ``partitions`` columns only, and only those are copied out of PSUM.
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
        dst=transposed[:, :, 0:partitions], src=staged[:, :, 0:partitions],
        engine=nisa.vector_engine,
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


@nki.jit
def _kpool_hadamard_nki(slot_k_hbm, slot_score_hbm, ape_hbm, n_pools, pool_size):
    """Fused per-(pool, channel) softmax-weighted pooling and Hadamard-128 rotation.

    Args:
        slot_k_hbm: ``[n_pools * pool_size, head_dim]`` -- ``slot_k`` flattened so that a pool and
            a slot together address one row. bf16 on the shipped path.
        slot_score_hbm: ``[n_pools * pool_size, head_dim]`` -- the gate's per-token score,
            flattened the same way.
        ape_hbm: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        n_pools: pools in the batch. A compile-time constant.
        pool_size: tokens per pool. A compile-time constant.

    Returns:
        ``[n_pools, head_dim]`` in ``slot_k_hbm``'s dtype.

    The pool axis is the partition axis and the head dimension is the free axis,
    which is what keeps the whole reduction elementwise: the softmax runs over the
    slot tiles, so each of its steps is a ``tensor_tensor`` between two
    ``(rows, head_dim)`` tiles and nothing reduces across partitions or along the
    free axis. A short final tile is narrowed rather than masked, so no padded row
    can reach the output.
    """
    head_dim = slot_k_hbm.shape[1]
    out_hbm = nl.ndarray((n_pools, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.shared_hbm)
    pmax = nl.tile_size.pmax
    row_stride = pool_size * head_dim

    for t in range((n_pools + pmax - 1) // pmax):
        rows = min(pmax, n_pools - t * pmax)
        base = t * pmax * row_stride

        # Pass 1: per-(pool, channel) max of slot_score + ape, for softmax stability.
        # The sums are kept rather than recomputed in pass 2: a tile holds up to 128 pools, so
        # holding pool_size fp32 tiles costs less than that many more strided DMA reads.
        totals = []
        running_max = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        for slot in range(pool_size):
            score = _load_fp32(slot_score_hbm, rows, head_dim, row_stride, base + slot * head_dim)
            bias = _broadcast_row(ape_hbm, rows, head_dim, slot)
            total = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=total, data1=score, data2=bias, op=nl.add)
            totals.append(total)
            if slot == 0:
                nisa.tensor_copy(dst=running_max, src=total)
            else:
                nisa.tensor_tensor(
                    dst=running_max, data1=running_max, data2=total, op=nl.maximum
                )

        # Pass 2: softmax-weighted sum of slot_k.
        acc = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        denom = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        nisa.memset(dst=denom, value=0.0)
        for slot in range(pool_size):
            shifted = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=shifted, data1=totals[slot], data2=running_max, op=nl.subtract)
            weight = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=weight, op=nl.exp, data=shifted)
            nisa.tensor_tensor(dst=denom, data1=denom, data2=weight, op=nl.add)
            key = _load_fp32(slot_k_hbm, rows, head_dim, row_stride, base + slot * head_dim)
            weighted = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=weighted, data1=weight, data2=key, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=weighted, op=nl.add)

        # Reciprocal then multiply, not a divide: one op per tile either way, and the reciprocal
        # is the form the ISA exposes directly.
        inv = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=denom)
        pooled = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)

        # Rotate, scale once, cast, store.
        scratch = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        rotated = _fwht128_inplace(pooled, scratch, head_dim)
        scaled = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
        result = nl.ndarray((rows, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=scaled)
        nl.store(
            out_hbm.ap(pattern=[[head_dim, rows], [1, head_dim]], offset=t * pmax * head_dim),
            value=result,
        )
    return out_hbm


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
    pending = _transpose_group(rows_in[0], blocks[0], groups[0])
    for index in range(len(groups)):
        group = groups[index]
        ready = pending
        if index + 1 < len(groups):
            following = groups[index + 1]
            if following[3] and following[0] + 1 < len(blocks):
                tiles = _load_block(x_hbm, blocks[following[0] + 1])
                rows_in.append(tiles[0])
                rows_out.append(tiles[1])
            pending = _transpose_group(rows_in[following[0]], blocks[following[0]], following)
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
    kernel = _kpool_hadamard_nki if entry == "fused" else _hadamard128_nki
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
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and n_items >= 2:
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
    """Whether the fused NKI kernel serves this call. ``False`` sends it to the torch path."""
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


def dsa_kpool_hadamard(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> Tensor:
    """Pool ``pool_size`` keys into one per pool and rotate the result.

    Args:
        slot_k: ``[n_pools, pool_size, head_dim]`` -- raw per-token indexer keys, one complete pool
            per row of the first axis. ``bfloat16`` takes the NKI route; any other dtype is served
            by the torch path.
        slot_score: ``[n_pools, pool_size, head_dim]`` -- the gate's per-token score.
        ape: ``[pool_size, head_dim]`` -- the per-slot additive position bias.

    Returns:
        ``[n_pools, head_dim]`` in ``slot_k``'s dtype: the pooled, rotated key per pool. No fp8
        output and no scale -- the adapter owns that half.

    Raises:
        KpoolHadamardError: for a malformed call -- a non-3D ``slot_k``, a ``slot_score`` that does
            not match it, an ``ape`` of the wrong shape, or a head dimension that is not 128.
    """
    n_pools, pool_size, head_dim = _validate(slot_k, slot_score, ape)

    if not can_run_dsa_kpool_hadamard(slot_k, slot_score, ape):
        return _dsa_kpool_hadamard_torch(slot_k, slot_score, ape)

    flat_k = slot_k.reshape(n_pools * pool_size, head_dim).contiguous()
    flat_score = slot_score.reshape(n_pools * pool_size, head_dim).contiguous()

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter store
    # inside the trace becomes a value guard that fails on the first call after warmup.
    _record_nki_dispatch("fused", n_pools, pool_size, head_dim)
    return wrap_nki(_kpool_hadamard_nki)(
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
