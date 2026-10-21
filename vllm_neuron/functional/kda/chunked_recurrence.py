# SPDX-License-Identifier: Apache-2.0
"""KDA chunked recurrence, intra-chunk half: stages 1 to 3, in NKI.

Kimi Delta Attention's chunked form splits in two. This module is the
intra-chunk half: the gate cumulative sum, the L2-normalisation of ``q`` and
``k``, the two chunk-local products ``A`` and ``Aqk``, the inverse
``(I + A)**-1``, and the WY representation ``w`` / ``u`` together with the gated
key ``kg``. The state carried across chunks and the output belong to the
inter-chunk half. The torch code here is a CPU oracle, never the shipped path.

The rule
--------
Per token, with state ``H`` shaped ``[V, K]``::

    H *= exp(gk)          # per key channel; the generic gated delta rule
                          # decays by a single scalar instead
    u = beta * (v - H k)
    H += u k^T
    o = H q

``q`` and ``k`` are L2-normalised as ``x / sqrt(sum(x**2) + eps)`` with the
epsilon **inside** the root, and ``q *= K ** -0.5``. :data:`L2_NORM_EPS` is that
epsilon, declared here rather than inherited from a default: both real KDA paths
use ``1e-6``, while the ``1e-5`` default belongs to a CPU shim that is not
callable in this image, and at ``atol=1e-5`` the wrong pick moves a comparison
by about the size of the tolerance.

The stages
----------
For one chunk of ``C`` tokens, key width ``K`` and value width ``V``, with ``gc``
the inclusive cumulative gate::

    gc[t, c] = sum over s <= t of gk[s, c]

    A[t, j]   = beta[t] * sum_c k[t, c] k[j, c] exp(gc[t, c] - gc[j, c])   for t > j
    Aqk[t, j] = sum_c q[t, c] k[j, c] exp(gc[t, c] - gc[j, c])             for t >= j
    T         = (I + A)**-1
    u         = T @ (beta * v)
    w         = T @ (beta * k * exp(gc))
    kg[t]     = k[t] * exp(gc[C - 1] - gc[t])

``A`` is strictly lower triangular and therefore singular, so the inverted object
is ``(I + A)`` and never ``A``. ``Aqk`` and ``kg`` are returned although nothing
here consumes them: the inter-chunk half needs both and has no other producer,
and it holds no raw ``k`` from which to rebuild ``kg``.

The inverse, without a token loop
---------------------------------
``N = -A`` is strictly lower triangular, so ``(I + A)**-1 = (I - N)**-1`` is
unit lower triangular. Split a block of ``2w`` tokens into halves, so that its
``N`` is ``[[N11, 0], [N21, N22]]``; then exactly::

    (I - N)**-1 = [[T1, 0], [T2 N21 T1, T2]]     T1 = (I - N11)**-1, T2 = (I - N22)**-1

With ``T_w`` the block-diagonal of the inverses of width ``w`` and ``N_x`` the
cross blocks ``N21``, that is ``T_2w = T_w + T_w N_x T_w``. The kernel starts at
width 1, where ``T_1 = I``, and merges the blocks in pairs for
:func:`doubling_stages` == ``log2(C)`` levels of at most three matmuls each,
rather than ``C`` substitution steps. No loop in this module walks the token
axis: the cumulative sum is a matmul against a triangular ones matrix, both
chunk-local products are matmuls, and ``w`` / ``u`` are matmuls against the
returned inverse.

Every term a level adds is an entry of the exact inverse, so no sum cancels.
The terminating Neumann series ``I + N + ... + N**(C-1)`` reaches the same value
in exact arithmetic, but not in fp32: ``|A[t, j]| <= 1``, so an entry of
``N**m`` can reach the path count ``binomial(C - 2, m - 1)``, about 1.6e8 at
``C = 32``. Aligned keys, ``beta`` near 1 and a slow gate come close to that
bound, and such terms carry rounding errors of order 1 into an inverse whose
entries are not large. The state the inter-chunk half carries then grows from
chunk to chunk.

The torch oracle below reaches the same value by a different route,
``torch.linalg.solve_triangular``, which is what makes their agreement
informative rather than circular.

Entry points
------------
:func:`kda_intra_chunk` is the entry point callers use, and it performs exactly
one ``wrap_nki`` dispatch per call, because the chunk loop lives inside the
kernel. :func:`kda_intra_chunk_kernel` computes stages 1 to 3 together;
:func:`kda_stage3_kernel` computes stage 3 alone and takes the inverse as an
argument, which is the boundary upstream's ``recompute_w_u_fwd`` also uses. Both
emit stage 3 from the shared :func:`_emit_uw` and :func:`_emit_kg`, so stage 3
has one implementation.

Packed tiles
------------
A chunk of ``C`` tokens fills ``C`` of a tile's ``MAX_TILE`` partitions, so the
kernels place :func:`tile_rows` ``// C`` chunks on one tile. Every chunk-local
``[C, C]`` product of the tile is then a diagonal block of one ``[rows, rows]``
product, and the block-diagonal masks zero the cross-chunk blocks. The masks are
the host constants laid on every diagonal block in-kernel (:func:`_emit_chunk_rows`,
:func:`_emit_unpack`), so the inputs keep their ``[C, C]`` shapes. Each
kept entry of a masked product contracts the same terms as the per-chunk product,
and a product of two block-diagonal tiles adds only exact zeros from the other
blocks.

On an LNC2 runtime the entry points launch two programs, one per physical core:
the intra-chunk kernel splits the whole tiles between them
(:func:`intra_chunk_grid`), the inter-chunk kernel the value columns of the
state (:func:`inter_chunk_grid`).

Heads
-----
Both entry points take operands with a leading heads axis. Each head is its own
sequence, and the heads share nothing but the launch: the intra-chunk kernel runs
them as more chunks, and the inter-chunk kernel as more chains. A head's results
are those of a one-head call on its operands.

Gate blocks
-----------
The two chunk-local products factorise the gate difference as
``exp(gc[t]) * exp(-gc[j])`` so that one matmul contracts the channel axis, and
both factors must stay inside fp32's range (:data:`GATE_CUMSUM_ABS_LIMIT`). The
limit applies to one gate block of ``S`` tokens, where ``S`` is ``C`` or ``C / 2``
(the ``gate_block`` of :func:`kda_intra_chunk`). With two blocks, ``gs`` is the
cumulative gate restarted at each block, and for ``t`` in the second block and
``j`` in the first::

    exp(gc[t] - gc[j]) = exp(gs[t]) * exp(gs[S - 1] - gs[j])

The second factor gives ``f[j] = k[j] * exp(gs[S - 1] - gs[j])``, which is ``kg``
at the block width. So the cross block of ``A`` and of ``Aqk`` is one more matmul
against ``f``, and inside a block the products are the factorised ones at ``gs``.
For a non-positive gate, which the KDA gate is, every other exponential here
(``exp(gc)`` in ``w``, ``exp(gc[C - 1] - gc)`` in ``kg`` and both factors of the
cross block) is at most 1, so only the block's gate sum needs the limit. A chunk
of two blocks gives half as many chunks, so the inter-chunk half's chain of
dependent steps is half as long.

Four ``[C, C]`` constants are built on the host by :func:`chunk_constants` and
passed in as tensors: the upper-inclusive ones matrix that turns the cumulative
sum into a matmul, the identity, the strictly lower mask, and a row selector that
broadcasts the chunk's last gate row. A chunk of two gate blocks also takes three
of them at the block width: the ones matrix, the strictly lower mask and the row
selector, which restart the gate at each block, mask the in-block products and
select each block's last gate row. Passing masks in follows ``nkilib``'s own
convention -- its ``ssd`` asserts a non-``None`` ``causal_mask`` rather than
building one -- and these are 0/1 constants, not numerics. The inverse's level
masks are not passed: each is the difference of two of the block-diagonal masks
the kernel builds to lay the constants out (:func:`_emit_merge_masks`), so the
kernel entries keep their arguments.

On this image the tiles ``wrap_nki`` hands a kernel carry no Python operator
overloads, so ``a * b`` on two tiles raises ``TypeError``. Every elementwise step
below is therefore an explicit ``nisa.tensor_tensor`` / ``nisa.tensor_scalar`` /
``nisa.activation`` call, and every matmul writes into a PSUM tile that is copied
to SBUF before it is stored, because a direct PSUM-to-HBM store asserts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import can_run_kernel, values_are_readable

logger = logging.getLogger(__name__)


#: The L2-normalisation epsilon, **inside** the square root. See the module
#: docstring for why this is a declared value and not an inherited default.
L2_NORM_EPS = 1e-6

#: Widest chunk, key and value extent, one partition tile each. A literal rather
#: than ``nl.tile_size.pmax`` read at import time, because this module must import
#: on a host with no NKI device.
MAX_TILE = 128

#: fp32's range in nats: ``exp(x)`` is a normal fp32 number for ``x`` from
#: ``FP32_LOG_TINY`` (about -87.34) to ``FP32_LOG_MAX`` (about 88.72).
FP32_LOG_TINY = math.log(torch.finfo(torch.float32).tiny)
FP32_LOG_MAX = math.log(torch.finfo(torch.float32).max)

#: Headroom, in nats, between the largest admissible ``|gc|`` and fp32's range.
#: The chunk-local products scale each of up to ``MAX_TILE`` channel components by
#: ``exp(gc)`` and ``exp(-gc)``. A component of a unit vector is typically
#: ``K ** -0.5`` and the query carries ``K ** -0.5`` more, so ``log(MAX_TILE)``
#: keeps a ``1 / MAX_TILE`` component times the smaller factor a normal number, and
#: a sum of ``MAX_TILE`` terms of the larger factor finite.
GATE_EXPONENT_MARGIN = math.log(MAX_TILE)

#: Largest ``|gc|`` the intra-chunk kernel accepts, where ``gc`` is the inclusive
#: cumulative gate over one gate block (module docstring, Gate blocks). The two
#: chunk-local products are formed as ``exp(gc[t]) * exp(-gc[j])`` so that one
#: matmul contracts the channel axis. That factorisation is exact, but it evaluates
#: both signs of the exponent, so a cumulative gate far from zero takes one factor
#: out of fp32's range even where the product itself is moderate. The limit is the
#: narrower side of that range less ``GATE_EXPONENT_MARGIN``, about 82.48. Upstream
#: keeps the same quantity small by blocking to 16 or 64 tokens and re-referencing
#: the gate per block; this kernel re-references it at most once per chunk, at the
#: gate block, and checks the limit on one block's sum. The inter-chunk kernel
#: exponentiates the cumulative gate with one sign only, so it checks the limit on
#: the largest value rather than on the size.
GATE_CUMSUM_ABS_LIMIT = min(-FP32_LOG_TINY, FP32_LOG_MAX) - GATE_EXPONENT_MARGIN

#: Physical cores behind one logical core on an LNC2 runtime, and so the programs
#: of a two-program launch.
LNC2_PROGRAMS = 2


class ChunkedRecurrenceError(ValueError):
    """Raised for a geometry or gate range this kernel does not serve."""


class IntraChunkOutputs(NamedTuple):
    """The five values stages 1 to 3 produce.

    ``a_inv`` and ``aqk`` are side outputs: nothing in this module consumes them,
    and the inter-chunk half has no other producer for them.
    """

    w: Tensor
    u: Tensor
    kg: Tensor
    a_inv: Tensor
    aqk: Tensor


class Stage3Outputs(NamedTuple):
    """What stage 3 alone returns."""

    w: Tensor
    u: Tensor
    kg: Tensor


class ChunkConstants(NamedTuple):
    """The four ``[C, C]`` host-built constants both kernel entries take."""

    triu_ones: Tensor
    eye: Tensor
    mask_lower: Tensor
    last_row: Tensor


@dataclass
class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts ``wrap_nki`` dispatches, ``torch_fallback`` entries
    into the torch path. Two counters rather than one flag, so "the kernel ran"
    and "the fallback did not" are independent readings.

    The count is per dispatch, not per call: the chunk loop lives inside the
    kernel, so a design that dispatched once per chunk reads a different number.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. The
#: inter-chunk half owns its own counters.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def doubling_stages(chunk: int) -> int:
    """Levels the block-merge inverse needs for a ``chunk``-wide tile.

    ``log2(chunk)``, because level ``j`` doubles the width of the solved
    diagonal blocks to ``2**j`` (module docstring, The inverse).
    """
    return int(math.ceil(math.log2(chunk)))


def chunk_constants(
    chunk: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ChunkConstants:
    """Build the four ``[chunk, chunk]`` constants the kernel entries take.

    * ``triu_ones[s, t] = 1 for s <= t`` -- used as the matmul **stationary**
      operand, whose transpose is the lower-inclusive ones matrix, so
      ``triu_ones^T @ gk`` is the inclusive cumulative sum along tokens.
    * ``eye`` -- the identity, which the inverse's first level adds to ``N``.
    * ``mask_lower[t, j] = 1 for j < t`` -- strictly lower, matching upstream's
      ``A``. The causal mask ``Aqk`` needs is ``mask_lower + eye`` and is formed
      in the kernel rather than passed, so one fewer constant travels.
    * ``last_row[s, t] = 1 for s == chunk - 1`` -- as a stationary operand its
      transpose selects the chunk's last row and repeats it down every row,
      which is what ``kg`` needs without a partition-axis broadcast.
    """
    idx = torch.arange(chunk, device=device)
    rows = idx.unsqueeze(1)
    cols = idx.unsqueeze(0)
    return ChunkConstants(
        triu_ones=(rows <= cols).to(dtype),
        eye=torch.eye(chunk, device=device, dtype=dtype),
        mask_lower=(cols < rows).to(dtype),
        last_row=(rows == (chunk - 1)).to(dtype).expand(chunk, chunk).contiguous(),
    )


# NKI emitting helpers, shared by both kernel entry points.


def _sbuf(rows, cols):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows, cols):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _dma(dst, src, engine):
    """``dst = src`` by DMA, descriptors from ``engine``'s hardware DGE.

    ``engine`` is ``nisa.engine.sync`` or ``nisa.engine.scalar``, the two that
    drive a hardware descriptor ring. Each DMA instruction occupies its issuing
    queue for about a microsecond, so the kernels alternate the two and a tile's
    loads arrive in parallel; GpSimd, which software descriptors would occupy,
    stays free for the layout constants.
    """
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=engine)


def _emit_transpose(dst, src, rows, cols):
    """``dst = src^T`` for a ``[rows, cols]`` source, through PSUM.

    The PSUM destination is load-bearing: on this image an ``nc_transpose`` whose
    destination is in SBUF runs on the Vector engine and asserts above
    ``[32, 32]``, while a PSUM destination routes it to the tensor engine, which
    serves the full 128. Every transpose in this module is wider than 32 on at
    least one axis.
    """
    ps = _psum(cols, rows)
    nisa.nc_transpose(dst=ps, data=src)
    nisa.tensor_copy(dst=dst, src=ps)


def _emit_l2_normalise(dst, src, rows, cols):
    """``dst = src / sqrt(sum_c src**2 + L2_NORM_EPS)``, epsilon inside the root.

    The reduction is along the free axis, so ``nl.sum(..., axis=1)`` serves it and
    the reciprocal square root broadcasts back from a ``[rows, 1]`` tile through
    ``nisa.tensor_scalar``'s ``operand0``. No partition-axis reduction.
    """
    sq = _sbuf(rows, cols)
    nisa.tensor_tensor(dst=sq, data1=src, data2=src, op=nl.multiply)
    total = nl.sum(sq, axis=1, keepdims=True, dtype=nl.float32)
    den = _sbuf(rows, 1)
    nisa.tensor_scalar(dst=den, data=total, op0=nl.add, operand0=L2_NORM_EPS)
    inv_root = _sbuf(rows, 1)
    nisa.activation(dst=inv_root, data=den, op=nl.rsqrt)
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=inv_root)


def _emit_gate_cumsum(dst, gk_sb, triu_sb, chunk, width):
    """Inclusive cumulative gate along tokens, as one matmul.

    ``nisa.nc_matmul(stationary=S, moving=M)`` contracts the partition axis and
    computes ``S^T @ M``, so with ``S = triu_ones`` the result is
    ``lower_inclusive_ones @ gk``, which is the inclusive cumulative sum.
    """
    ps = _psum(chunk, width)
    nisa.nc_matmul(dst=ps, stationary=triu_sb, moving=gk_sb, accumulate=False)
    nisa.tensor_copy(dst=dst, src=ps)


def tile_rows(chunk: int) -> int:
    """Token rows of one packed tile: the most whole chunks one partition tile holds.

    Both kernels place ``MAX_TILE // chunk`` chunks on the partitions of one tile, so
    a ``chunk``-wide matrix of every chunk in the tile sits on the diagonal of one
    ``[tile_rows, tile_rows]`` block-diagonal tile.
    """
    return (MAX_TILE // chunk) * chunk


def _emit_same_chunk(rows, chunk):
    """``[rows, rows]`` 0/1 tile: 1 where tokens ``p`` and ``f`` share a chunk.

    With the free axis viewed as ``(f // chunk, f % chunk)``, ``p // chunk ==
    f // chunk`` is the pair of affine conditions ``p - chunk * (f // chunk) >= 0``
    and ``chunk * (f // chunk) + chunk - 1 - p >= 0``, one ``affine_select`` each.
    Returned as ``[rows, rows // chunk, chunk]``.
    """
    groups = rows // chunk
    ones = nl.ndarray((rows, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    not_before = nl.ndarray((rows, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.affine_select(
        dst=not_before, pattern=[[-chunk, groups], [0, chunk]], channel_multiplier=1,
        on_true_tile=ones, on_false_value=0.0, cmp_op=nl.greater_equal,
    )
    same = nl.ndarray((rows, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.affine_select(
        dst=same, pattern=[[chunk, groups], [0, chunk]], channel_multiplier=-1,
        on_true_tile=not_before, on_false_value=0.0, cmp_op=nl.greater_equal,
        offset=chunk - 1,
    )
    return same


def _emit_replicator(rows, chunk):
    """``[chunk, rows]`` 0/1 tile: 1 where ``f % chunk`` equals the partition.

    As a stationary operand it copies row ``f % chunk`` of a ``[chunk, X]`` moving
    operand onto output row ``f``: one nonzero term per output, so exact.
    """
    groups = rows // chunk
    ones = nl.ndarray((chunk, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    rep = nl.ndarray((chunk, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.affine_select(
        dst=rep, pattern=[[0, groups], [1, chunk]], channel_multiplier=-1,
        on_true_tile=ones, on_false_value=0.0, cmp_op=nl.equal,
    )
    return rep


def _emit_unpack(dst, packed, same, rows, chunk):
    """``dst [rows, rows]`` block-diagonal from ``packed [rows, chunk]``.

    Row ``p`` of a packed tile holds row ``p % chunk`` of its chunk's
    ``[C, C]`` matrix, which is the ``[NC, C, C]`` HBM layout read as rows.
    """
    groups = rows // chunk
    nisa.tensor_tensor(
        dst=dst.reshape((rows, groups, chunk)),
        data1=packed.reshape((rows, 1, chunk)).broadcast(1, groups),
        data2=same[0:rows, 0:groups, 0:chunk], op=nl.multiply,
    )


def _emit_chunk_rows(consts_hbm, rep, rows, chunk, engine):
    """Every ``[C, C]`` constant's row ``p % chunk`` on tile row ``p``.

    Returns ``[rows, len(consts_hbm) * chunk]``, constant ``i`` in columns
    ``i * chunk`` onward: that constant repeated in every chunk, in the packed
    layout :func:`_emit_unpack` reads. One matmul against the replicator serves
    all of them, and each output has one nonzero term, so it is exact. ``engine``
    issues the constants' DMAs.
    """
    count = len(consts_hbm)
    stacked = nl.ndarray((chunk, count, chunk), dtype=nl.float32, buffer=nl.sbuf)
    for index in range(count):
        _dma(stacked[0:chunk, index, 0:chunk], consts_hbm[index], engine)
    ps = _psum(rows, count * chunk)
    nisa.nc_matmul(
        dst=ps, stationary=rep.reshape((chunk, rows)),
        moving=stacked.reshape((chunk, count * chunk)), accumulate=False,
    )
    chunk_rows = _sbuf(rows, count * chunk)
    nisa.tensor_copy(dst=chunk_rows, src=ps)
    return chunk_rows


def _emit_block_diagonal(chunk_rows, index, same, rows, chunk):
    """``[rows, rows]``: constant ``index`` of :func:`_emit_chunk_rows` on every
    diagonal block, zero elsewhere."""
    dst = _sbuf(rows, rows)
    _emit_unpack(dst, chunk_rows[0:rows, index * chunk : (index + 1) * chunk], same,
                 rows, chunk)
    return dst


def _emit_pack(dst, full, rows, chunk):
    """``dst [rows, chunk]`` = the diagonal blocks of a block-diagonal ``full``.

    Every off-block entry of ``full`` is zero, so the sum over the column blocks
    is the diagonal block itself, exactly.
    """
    groups = rows // chunk
    nisa.tensor_reduce(
        dst=dst, data=full.reshape((rows, groups, chunk)).permute((0, 2, 1)),
        op=nl.add, axis=(2,),
    )


@dataclass(frozen=True)
class _IntraLayout(nl.NKIObject):
    """The block-diagonal constants of one packed tile, built once per launch.

    An ``NKIObject``, because the NKI front end admits no other user class.
    ``merge_masks`` holds one ``[rows, rows]`` tile per level of the inverse
    (:func:`_emit_merge_masks`).
    """

    triu: object
    neg_mask_lower: object
    causal: object
    last_row: object
    same_chunk: object
    eye: object
    merge_masks: tuple


def _emit_intra_layout(triu_hbm, eye_hbm, mask_lower_hbm, last_row_hbm, rows, chunk):
    """The four ``[C, C]`` constants, laid on the diagonal of a packed tile.

    ``neg_mask_lower`` is ``-mask_lower``, so the inverse's ``N = -A`` is one
    product; ``causal`` is ``mask_lower + eye``; ``merge_masks`` are the
    inverse's level masks.
    """
    same = _emit_same_chunk(rows, chunk)
    rep = _emit_replicator(rows, chunk)
    chunk_rows = _emit_chunk_rows(
        (triu_hbm, eye_hbm, mask_lower_hbm, last_row_hbm), rep, rows, chunk,
        nisa.engine.scalar,
    )
    triu = _emit_block_diagonal(chunk_rows, 0, same, rows, chunk)
    eye = _emit_block_diagonal(chunk_rows, 1, same, rows, chunk)
    mask_lower = _emit_block_diagonal(chunk_rows, 2, same, rows, chunk)
    last_row = _emit_block_diagonal(chunk_rows, 3, same, rows, chunk)
    neg_mask_lower = _sbuf(rows, rows)
    nisa.tensor_scalar(
        dst=neg_mask_lower, data=mask_lower, op0=nl.multiply, operand0=-1.0
    )
    causal = _sbuf(rows, rows)
    nisa.tensor_tensor(dst=causal, data1=mask_lower, data2=eye, op=nl.add)
    merge_masks = _emit_merge_masks(same, rows, chunk)
    return _IntraLayout(
        triu, neg_mask_lower, causal, last_row, same, eye, merge_masks
    )


def _emit_merge_masks(same_chunk, rows, chunk):
    """One ``[rows, rows]`` 0/1 mask per level of :func:`_emit_unit_lower_inverse`.

    Level ``l`` joins the solved diagonal blocks of width ``w = 2**(l - 1)`` in
    pairs, and its mask is 1 where two tokens share a ``2w``-aligned block of
    their chunk but not a ``w``-aligned one: :func:`_emit_same_chunk` at ``2w``
    less the same at ``w``, 0/1 arithmetic and exact. Every pair of distinct
    tokens of a chunk is in exactly one level's mask, the level of the highest
    bit in which their positions differ. At level 1 the ``w`` block is one token,
    the diagonal, where the product level 1 masks is zero, so its mask is the
    ``2``-block one alone. At the top level the ``2w`` block is the chunk,
    ``same_chunk``.

    The masks are symmetric: the products they mask are triangular already
    (:func:`_emit_unit_lower_inverse`), so a mask only has to pick the level.
    """
    levels = doubling_stages(chunk)
    masks = []
    inner = None
    for level in range(1, levels + 1):
        width = 2**level
        if width == chunk:
            outer = same_chunk.reshape((rows, rows))
        else:
            outer = _emit_same_chunk(rows, width).reshape((rows, rows))
        if inner is None:
            masks.append(outer)
        else:
            mask = _sbuf(rows, rows)
            nisa.tensor_tensor(dst=mask, data1=outer, data2=inner, op=nl.subtract)
            masks.append(mask)
        inner = outer
    return tuple(masks)


@dataclass(frozen=True)
class _GateBlockLayout(nl.NKIObject):
    """The gate-block constants of one packed tile, for a chunk of two gate blocks.

    ``triu`` restarts the cumulative gate at every block, and ``last_row`` selects
    every block's last gate row. ``neg_mask_lower`` and ``causal`` mask ``N`` and
    ``Aqk`` inside a block. ``cross`` is 1 where ``t`` is in a chunk's second block
    and ``j`` in its first, and ``neg_cross`` is ``-cross`` (module docstring, Gate
    blocks).
    """

    triu: object
    last_row: object
    neg_mask_lower: object
    causal: object
    cross: object
    neg_cross: object


def _emit_gate_block_layout(
    triu_hbm, mask_lower_hbm, last_row_hbm, chunk_layout, rows, block
):
    """:class:`_GateBlockLayout` from the three ``[S, S]`` constants of the block width.

    Two tokens of one chunk are either in one block or in the cross block. So
    ``neg_cross`` is ``chunk_layout``'s ``neg_mask_lower`` less the blocks', and the
    blocks' ``causal`` is ``chunk_layout``'s less ``cross``: 0/1 arithmetic, exact.
    """
    same = _emit_same_chunk(rows, block)
    rep = _emit_replicator(rows, block)
    block_rows = _emit_chunk_rows(
        (triu_hbm, mask_lower_hbm, last_row_hbm), rep, rows, block, nisa.engine.sync
    )
    triu = _emit_block_diagonal(block_rows, 0, same, rows, block)
    mask_lower = _emit_block_diagonal(block_rows, 1, same, rows, block)
    last_row = _emit_block_diagonal(block_rows, 2, same, rows, block)
    neg_mask_lower = _sbuf(rows, rows)
    nisa.tensor_scalar(
        dst=neg_mask_lower, data=mask_lower, op0=nl.multiply, operand0=-1.0
    )
    neg_cross = _sbuf(rows, rows)
    nisa.tensor_tensor(
        dst=neg_cross, data1=chunk_layout.neg_mask_lower, data2=neg_mask_lower,
        op=nl.subtract,
    )
    cross = _sbuf(rows, rows)
    nisa.tensor_scalar(dst=cross, data=neg_cross, op0=nl.multiply, operand0=-1.0)
    causal = _sbuf(rows, rows)
    nisa.tensor_tensor(
        dst=causal, data1=chunk_layout.causal, data2=cross, op=nl.subtract
    )
    return _GateBlockLayout(triu, last_row, neg_mask_lower, causal, cross, neg_cross)


def _emit_unit_lower_inverse(dst, n_bd, layout, rows, chunk):
    """``dst = (I - N)**-1 = (I + A)**-1`` in the packed layout, by block merge.

    ``n_bd`` is ``N = -A``, block-diagonal ``[rows, rows]`` and strictly lower in
    every block. ``t`` is the inverse solved on diagonal blocks of width ``w``,
    and ``tt`` its transpose, both block-diagonal ``[rows, rows]``. Level 1 is
    ``t = I + N`` on the level's mask, and each later level merges the blocks in
    pairs (module docstring, The inverse)::

        t <- t + t N_x t

    with ``N_x`` the level's cross block. ``t`` is zero outside its own ``w``
    blocks, so ``t N`` and ``t N_x`` agree on the cross block, and
    ``y = (N^T tt) * mask`` is ``(t N_x)^T`` exactly without a masked copy of
    ``N``. ``t`` and ``tt`` both stay current because the next level needs ``t``
    as the moving operand and ``tt`` as the stationary one:
    ``nc_matmul(stationary=y, moving=t)`` is ``t N_x t`` and
    ``nc_matmul(stationary=t, moving=y)`` its transpose. The last level needs no
    transpose and streams the packed ``t``, ``chunk`` columns rather than
    ``rows``. :func:`doubling_stages` == ``log2(chunk)`` levels, and no loop walks
    the tokens.

    ``N`` and every ``t`` are exactly zero above the diagonal, so ``N^T tt`` is
    exactly zero below it, and each level's symmetric mask
    (:func:`_emit_merge_masks`) only has to select the level.

    ``dst`` receives the packed inverse, which is the ``[NC, C, C]`` HBM layout
    as rows.
    """
    masks = layout.merge_masks
    eye = layout.eye[0:rows, 0:rows]
    first = _sbuf(rows, rows)
    nisa.tensor_tensor(dst=first, data1=n_bd, data2=masks[0][0:rows, 0:rows],
                       op=nl.multiply)
    t_cur = _sbuf(rows, rows)
    nisa.tensor_tensor(dst=t_cur, data1=first, data2=eye, op=nl.add)
    levels = len(masks)
    if levels == 1:
        _emit_pack(dst, t_cur, rows, chunk)
        return
    tt_cur = _sbuf(rows, rows)
    _emit_transpose(tt_cur, t_cur, rows, rows)
    for level in range(1, levels):
        # y = (t N_x)^T: the stationary N contracts its rows, so this is N^T tt.
        ps_y = _psum(rows, rows)
        nisa.nc_matmul(dst=ps_y, stationary=n_bd, moving=tt_cur, accumulate=False)
        y_sb = _sbuf(rows, rows)
        nisa.tensor_tensor(dst=y_sb, data1=ps_y,
                           data2=masks[level][0:rows, 0:rows], op=nl.multiply)
        if level == levels - 1:
            t_packed = _sbuf(rows, chunk)
            _emit_pack(t_packed, t_cur, rows, chunk)
            ps_x = _psum(rows, chunk)
            nisa.nc_matmul(dst=ps_x, stationary=y_sb, moving=t_packed,
                           accumulate=False)
            nisa.tensor_tensor(dst=dst, data1=t_packed, data2=ps_x, op=nl.add)
        else:
            ps_x = _psum(rows, rows)
            nisa.nc_matmul(dst=ps_x, stationary=y_sb, moving=t_cur, accumulate=False)
            ps_xt = _psum(rows, rows)
            nisa.nc_matmul(dst=ps_xt, stationary=t_cur, moving=y_sb, accumulate=False)
            t_next = _sbuf(rows, rows)
            nisa.tensor_tensor(dst=t_next, data1=t_cur, data2=ps_x, op=nl.add)
            tt_next = _sbuf(rows, rows)
            nisa.tensor_tensor(dst=tt_next, data1=tt_cur, data2=ps_xt, op=nl.add)
            t_cur, tt_cur = t_next, tt_next


def _blocks(tile, rows, n_blocks, width):
    """``tile [rows, n_blocks * width]`` viewed as ``[rows, n_blocks, width]``."""
    return tile.reshape((rows, n_blocks, width))


def _per_block(column, rows, n_blocks, width):
    """``column [rows, n_blocks]`` repeated along each block's ``width`` columns."""
    return column.reshape((rows, n_blocks, 1)).broadcast(2, width)


def _emit_l2_normalise_blocks(dst, src, rows, n_blocks, cols):
    """:func:`_emit_l2_normalise` of every ``[rows, cols]`` block of ``src``.

    ``src`` is ``n_blocks`` blocks side by side, ``[rows, n_blocks * cols]``, each
    normalised along its own row. The per-element operations are those of
    :func:`_emit_l2_normalise`; each runs once over every block instead of once
    per block.
    """
    sq = _sbuf(rows, n_blocks * cols)
    nisa.tensor_tensor(dst=sq, data1=src, data2=src, op=nl.multiply)
    total = _sbuf(rows, n_blocks)
    nisa.tensor_reduce(
        dst=total, op=nl.add, data=_blocks(sq, rows, n_blocks, cols), axis=(2,)
    )
    den = _sbuf(rows, n_blocks)
    nisa.tensor_scalar(dst=den, data=total, op0=nl.add, operand0=L2_NORM_EPS)
    inv_root = _sbuf(rows, n_blocks)
    nisa.activation(dst=inv_root, data=den, op=nl.rsqrt)
    nisa.tensor_tensor(
        dst=_blocks(dst, rows, n_blocks, cols),
        data1=_blocks(src, rows, n_blocks, cols),
        data2=_per_block(inv_root, rows, n_blocks, cols), op=nl.multiply,
    )


#: fp32 elements one PSUM bank holds per partition, so the widest matmul result.
PSUM_BANK_FP32 = 512


def _emit_row_matmul(dst, stationary, moving, rows, width):
    """``dst [rows, width] = stationary^T @ moving``, one PSUM bank at a time.

    For a ``moving`` of several side-by-side blocks that share one stationary
    operand: the gate cumulative sum and the last-row selection of every packed
    tile in one pass per bank.
    """
    for c0 in range(0, width, PSUM_BANK_FP32):
        cols = min(PSUM_BANK_FP32, width - c0)
        ps = _psum(rows, cols)
        nisa.nc_matmul(
            dst=ps, stationary=stationary, moving=moving[0:rows, c0 : c0 + cols],
            accumulate=False,
        )
        nisa.tensor_copy(dst=dst[0:rows, c0 : c0 + cols], src=ps)


def _emit_gate_terms(gc_sb, exps, gk_sb, triu_bd, rows, width):
    """The cumulative gate ``gc`` and its exponentials, one PSUM bank at a time.

    ``gk_sb`` is side-by-side packed tiles ``[rows, width]``; ``triu_bd`` the
    block-diagonal upper-inclusive ones, so the cumulative sum restarts at every
    chunk. ``exps`` is a tuple of ``(dst, scale)``: each ``dst = exp(scale * gc)``.
    The exponentials read the matmul's PSUM result while the Vector engine copies
    it out, so neither waits on the other.
    """
    for c0 in range(0, width, PSUM_BANK_FP32):
        cols = min(PSUM_BANK_FP32, width - c0)
        ps = _psum(rows, cols)
        nisa.nc_matmul(
            dst=ps, stationary=triu_bd, moving=gk_sb[0:rows, c0 : c0 + cols],
            accumulate=False,
        )
        nisa.tensor_copy(dst=gc_sb[0:rows, c0 : c0 + cols], src=ps,
                         engine=nisa.engine.vector)
        for index in range(len(exps)):
            nisa.activation(dst=exps[index][0][0:rows, c0 : c0 + cols], data=ps,
                            op=nl.exp, scale=exps[index][1])


def _emit_prepare(k_sb, gc_sb, egc_sb, k_raw, gk_sb, triu_bd, rows, n_tiles, kdim):
    """L2-normalise ``k`` into ``k_sb``, then form ``gc`` and ``exp(gc)``.

    The stage-3 entry point's preparation, from the same helpers the combined
    entry point uses, so it derives its normalised key and cumulative gate
    exactly as that one does.
    """
    _emit_l2_normalise_blocks(k_sb, k_raw, rows, n_tiles, kdim)
    _emit_gate_terms(gc_sb, ((egc_sb, 1.0),), gk_sb, triu_bd, rows, n_tiles * kdim)


def _emit_uw(uw_dst, k_sb, v_sb, beta_col, egc_sb, a_inv_packed, same_chunk, rows,
             chunk, kdim, vdim):
    """Stage 3's ``u`` and ``w`` for one packed tile.

    ``u = T @ (beta * v)`` and ``w = T @ (beta * k * exp(gc))`` are one matmul
    against the block-diagonal inverse ``T``, unpacked from ``a_inv_packed`` and
    transposed: the two moving operands are disjoint column ranges of one tile,
    and ``uw_dst [rows, V + K]`` receives ``u`` then ``w``.
    """
    t_bd = _sbuf(rows, rows)
    _emit_unpack(t_bd, a_inv_packed, same_chunk, rows, chunk)
    t_t = _sbuf(rows, rows)
    _emit_transpose(t_t, t_bd, rows, rows)
    vk = _sbuf(rows, vdim + kdim)
    nisa.tensor_scalar(
        dst=vk[0:rows, 0:vdim], data=v_sb, op0=nl.multiply, operand0=beta_col
    )
    nisa.scalar_tensor_tensor(
        dst=vk[0:rows, vdim : vdim + kdim], data=k_sb, op0=nl.multiply,
        operand0=beta_col, op1=nl.multiply, operand1=egc_sb,
    )
    ps_uw = _psum(rows, vdim + kdim)
    nisa.nc_matmul(dst=ps_uw, stationary=t_t, moving=vk, accumulate=False)
    nisa.tensor_copy(dst=uw_dst, src=ps_uw)


def _emit_kg(kg_dst, k_sb, gc_sb, last_row_bd, rows, width):
    """Stage 3's ``kg[t] = k[t] * exp(gc[last of t's chunk] - gc[t])``.

    Over side-by-side packed tiles ``[rows, width]``: the chunk's last gate row
    comes from a matmul against the block-diagonal row selector.
    """
    last = _sbuf(rows, width)
    _emit_row_matmul(last, last_row_bd[0:rows, 0:rows], gc_sb, rows, width)
    gl = _sbuf(rows, width)
    nisa.tensor_tensor(dst=gl, data1=last, data2=gc_sb, op=nl.subtract)
    decay = _sbuf(rows, width)
    nisa.activation(dst=decay, data=gl, op=nl.exp)
    nisa.tensor_tensor(dst=kg_dst, data1=k_sb, data2=decay, op=nl.multiply)


def _program_tiles(rows, layout_rows):
    """``(full tiles per program, first full tile, tail rows)`` for this program.

    A two-program launch splits whole tiles evenly; :func:`intra_chunk_grid`
    only asks for one where that is possible, and anything else is refused here
    rather than silently computed twice or skipped.
    """
    n_prog = nl.num_programs(0)
    full, tail = divmod(rows, layout_rows)
    # The NKI front end admits ``assert`` and no ``raise``.
    assert n_prog == 1 or (tail == 0 and full % n_prog == 0), (
        "the packed tiles of this geometry do not split over the launch's programs"
    )
    per_prog = full // n_prog
    return per_prog, nl.program_id(0) * per_prog, tail


def _rows_view(tensor_hbm):
    """``[NC, C, X]`` HBM as ``[NC * C, X]`` token rows."""
    n_chunks, chunk, width = tensor_hbm.shape
    return tensor_hbm.reshape((n_chunks * chunk, width))


def _tile_rows(tensor_rows, r0, n_tiles, rows, width):
    """``n_tiles`` packed tiles of ``tensor_rows`` from row ``r0``, as one DMA operand.

    Token row ``r0 + i * rows + p`` maps to partition ``p``, block ``i``: the
    ``[rows, n_tiles * width]`` SBUF layout every batched step reads.
    """
    return tensor_rows[nl.ds(r0, n_tiles * rows), 0:width].reshape(
        (n_tiles, rows, width)
    ).permute((1, 0, 2))


def _load_tiles(tensor_rows, r0, n_tiles, rows, width, engine):
    """Load the rows :func:`_tile_rows` names into a new ``[rows, n_tiles * width]`` tile.

    One DMA, its descriptors from ``engine`` (see :func:`_dma`); the inverse of
    :func:`_store_tiles`.
    """
    tile = _sbuf(rows, n_tiles * width)
    _dma(_blocks(tile, rows, n_tiles, width),
         _tile_rows(tensor_rows, r0, n_tiles, rows, width), engine)
    return tile


def _store_tiles(tensor_rows, src3, r0, n_tiles, rows, width, engine):
    """Store ``src3 [rows, n_tiles, width]`` to the rows :func:`_tile_rows` names."""
    _dma(_tile_rows(tensor_rows, r0, n_tiles, rows, width), src3, engine)


@nki.jit
def kda_intra_chunk_kernel(
    q_hbm, k_hbm, v_hbm, beta_hbm, gk_hbm, triu_hbm, eye_hbm, mask_lower_hbm,
    last_row_hbm, block_triu_hbm=None, block_mask_lower_hbm=None,
    block_last_row_hbm=None,
):
    """Stages 1 to 3 for every chunk, in one dispatch.

    ``q``, ``k`` and ``gk`` are ``[NC, C, K]``; ``v`` is ``[NC, C, V]``; ``beta``
    is ``[NC, C, 1]``; the four constants are ``[C, C]``; all float32. Returns
    ``w [NC, C, K]``, ``u [NC, C, V]``, ``kg [NC, C, K]``, ``a_inv [NC, C, C]`` and
    ``aqk [NC, C, C]``, float32.

    The three ``block_*`` constants are :func:`chunk_constants`'s ``triu_ones``,
    ``mask_lower`` and ``last_row`` at the gate block width ``S = C / 2``, for a
    chunk of two gate blocks (module docstring, Gate blocks), or all ``None`` for a
    chunk of one.

    Tokens are packed ``MAX_TILE // C`` chunks to a tile (:func:`tile_rows`), and
    every chunk-local ``[C, C]`` product of the tile is a diagonal block of one
    ``[rows, rows]`` product, masked by the block-diagonal constants. Stages 1 to
    3 are chunk-local with no carry, so the tiles are independent: a program
    loads all of its tiles side by side, runs every row-wise step once over them
    and the chunk-local products per tile, and a two-program launch gives each
    program a contiguous half of the whole tiles.
    """
    n_chunks, chunk, kdim = q_hbm.shape
    vdim = v_hbm.shape[2]
    scale = float(kdim) ** -0.5
    rows = n_chunks * chunk
    layout_rows = min(tile_rows(chunk), rows)
    blocked = block_triu_hbm is not None
    # The NKI front end admits ``assert`` and no ``raise``.
    assert blocked == (block_mask_lower_hbm is not None) == (
        block_last_row_hbm is not None
    ), "the three gate-block constants come together"
    assert not blocked or 2 * block_triu_hbm.shape[0] == chunk, (
        "a chunk of gate blocks holds exactly two"
    )

    w_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    u_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    kg_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    ainv_hbm = nl.ndarray((n_chunks, chunk, chunk), dtype=nl.float32, buffer=nl.shared_hbm)
    aqk_hbm = nl.ndarray((n_chunks, chunk, chunk), dtype=nl.float32, buffer=nl.shared_hbm)
    sources = (
        _rows_view(q_hbm), _rows_view(k_hbm), _rows_view(v_hbm), _rows_view(beta_hbm),
        _rows_view(gk_hbm),
    )
    sinks = (
        _rows_view(w_hbm), _rows_view(u_hbm), _rows_view(kg_hbm), _rows_view(ainv_hbm),
        _rows_view(aqk_hbm),
    )

    layout = _emit_intra_layout(
        triu_hbm, eye_hbm, mask_lower_hbm, last_row_hbm, layout_rows, chunk
    )
    block = None
    if blocked:
        block = _emit_gate_block_layout(
            block_triu_hbm, block_mask_lower_hbm, block_last_row_hbm, layout,
            layout_rows, block_triu_hbm.shape[0],
        )
    per_prog, first, tail = _program_tiles(rows, layout_rows)
    if per_prog:
        _emit_intra_tiles(
            sinks, sources, layout, block, first * layout_rows, per_prog,
            layout_rows, chunk, kdim, vdim, scale,
        )
    if tail:
        _emit_intra_tiles(
            sinks, sources, layout, block, rows - tail, 1, tail, chunk, kdim, vdim,
            scale,
        )
    return w_hbm, u_hbm, kg_hbm, ainv_hbm, aqk_hbm


def _emit_intra_tiles(
    sinks, sources, layout, block, r0, n_tiles, rows, chunk, kdim, vdim, scale
):
    """Stages 1 to 3 for ``n_tiles`` packed tiles of ``rows`` rows from row ``r0``.

    The loads, the stores, the normalisation, the cumulative gate and its
    exponentials run once over all the tiles side by side; the chunk-local
    products run per tile, so one tile's products overlap the next one's
    elementwise steps. The normalisation's reciprocal root runs before every
    exponential, so the Scalar engine switches activation tables once.

    ``block`` is the :class:`_GateBlockLayout` of a chunk of two gate blocks, or
    ``None``; with ``None`` the gate blocks are the chunks, and the program is the
    one-block program unchanged.
    """
    q_rows, k_rows, v_rows, beta_rows, gk_rows = sources
    w_rows, u_rows, kg_rows, ainv_rows, aqk_rows = sinks
    wk = n_tiles * kdim
    width = vdim + kdim

    gk_sb = _load_tiles(gk_rows, r0, n_tiles, rows, kdim, nisa.engine.sync)
    # q and k side by side, so one pass of each normalisation step serves both.
    qk_raw = _sbuf(rows, 2 * wk)
    _dma(_blocks(qk_raw[0:rows, 0:wk], rows, n_tiles, kdim),
         _tile_rows(q_rows, r0, n_tiles, rows, kdim), nisa.engine.scalar)
    _dma(_blocks(qk_raw[0:rows, wk : 2 * wk], rows, n_tiles, kdim),
         _tile_rows(k_rows, r0, n_tiles, rows, kdim), nisa.engine.sync)
    v_sb = _load_tiles(v_rows, r0, n_tiles, rows, vdim, nisa.engine.scalar)
    beta_sb = _load_tiles(beta_rows, r0, n_tiles, rows, 1, nisa.engine.sync)

    qk_norm = _sbuf(rows, 2 * wk)
    _emit_l2_normalise_blocks(qk_norm, qk_raw, rows, 2 * n_tiles, kdim)
    k_sb = qk_norm[0:rows, wk : 2 * wk]
    q_sb = _sbuf(rows, wk)
    nisa.tensor_scalar(dst=q_sb, data=qk_norm[0:rows, 0:wk], op0=nl.multiply,
                       operand0=scale)
    # The gate difference exp(gc[t] - gc[j]) is factorised so that one matmul
    # contracts the channel axis: exp(gc[t]) on the left rows, exp(-gc[j]) on
    # the right rows. GATE_CUMSUM_ABS_LIMIT is what bounds both factors, and
    # pairs from different chunks are bounded the same way before the mask
    # zeroes them. With gate blocks the factorised gate ``gs`` restarts at every
    # block, the cross block reads ``f``, and ``w`` and ``kg`` read the chunk's
    # own ``gc`` (module docstring, Gate blocks).
    gc_sb = _sbuf(rows, wk)
    egc_sb = _sbuf(rows, wk)
    if block is None:
        gs_sb, egs_sb, in_block = gc_sb, egc_sb, layout
        emgs_sb = _sbuf(rows, wk)
        _emit_gate_terms(gc_sb, ((egc_sb, 1.0), (emgs_sb, -1.0)), gk_sb,
                         layout.triu[0:rows, 0:rows], rows, wk)
    else:
        gs_sb, egs_sb, emgs_sb, in_block = (
            _sbuf(rows, wk), _sbuf(rows, wk), _sbuf(rows, wk), block
        )
        _emit_gate_terms(gs_sb, ((egs_sb, 1.0), (emgs_sb, -1.0)), gk_sb,
                         block.triu[0:rows, 0:rows], rows, wk)
        _emit_gate_terms(gc_sb, ((egc_sb, 1.0),), gk_sb, layout.triu[0:rows, 0:rows],
                         rows, wk)
        f_sb = _sbuf(rows, wk)
        _emit_kg(f_sb, k_sb, gs_sb, block.last_row, rows, wk)

    ainv_sb = _sbuf(rows, n_tiles * chunk)
    aqk_sb = _sbuf(rows, n_tiles * chunk)
    uw_sb = _sbuf(rows, n_tiles * width)
    for t in range(n_tiles):
        k0 = t * kdim
        c0 = t * chunk
        k_t = k_sb[0:rows, k0 : k0 + kdim]
        egc_t = egc_sb[0:rows, k0 : k0 + kdim]
        egs_t = egs_sb[0:rows, k0 : k0 + kdim]
        beta_col = beta_sb[0:rows, t : t + 1]
        kp_sb = _sbuf(rows, kdim)
        km_sb = _sbuf(rows, kdim)
        qp_sb = _sbuf(rows, kdim)
        nisa.tensor_tensor(dst=kp_sb, data1=k_t, data2=egs_t, op=nl.multiply)
        nisa.tensor_tensor(dst=km_sb, data1=k_t, data2=emgs_sb[0:rows, k0 : k0 + kdim],
                           op=nl.multiply)
        nisa.tensor_tensor(dst=qp_sb, data1=q_sb[0:rows, k0 : k0 + kdim], data2=egs_t,
                           op=nl.multiply)
        km_t = _sbuf(kdim, rows)
        kp_t = _sbuf(kdim, rows)
        qp_t = _sbuf(kdim, rows)
        _emit_transpose(km_t, km_sb, rows, kdim)
        _emit_transpose(kp_t, kp_sb, rows, kdim)
        _emit_transpose(qp_t, qp_sb, rows, kdim)

        # N = -A = -(beta * kk) * mask_lower, block-diagonal.
        ps_kk = _psum(rows, rows)
        nisa.nc_matmul(dst=ps_kk, stationary=kp_t, moving=km_t, accumulate=False)
        n_bd = _sbuf(rows, rows)
        nisa.scalar_tensor_tensor(
            dst=n_bd, data=ps_kk, op0=nl.multiply, operand0=beta_col,
            op1=nl.multiply, operand1=in_block.neg_mask_lower[0:rows, 0:rows],
        )
        if block is not None:
            # The cross block: N += -(beta * kp f^T) * cross.
            f_t = _sbuf(kdim, rows)
            _emit_transpose(f_t, f_sb[0:rows, k0 : k0 + kdim], rows, kdim)
            ps_kf = _psum(rows, rows)
            nisa.nc_matmul(dst=ps_kf, stationary=kp_t, moving=f_t, accumulate=False)
            n_cross = _sbuf(rows, rows)
            nisa.tensor_tensor(
                dst=n_cross, data1=ps_kf, data2=block.neg_cross[0:rows, 0:rows],
                op=nl.multiply,
            )
            n_all = _sbuf(rows, rows)
            nisa.scalar_tensor_tensor(
                dst=n_all, data=n_cross, op0=nl.multiply, operand0=beta_col,
                op1=nl.add, operand1=n_bd,
            )
            n_bd = n_all

        ps_qk = _psum(rows, rows)
        nisa.nc_matmul(dst=ps_qk, stationary=qp_t, moving=km_t, accumulate=False)
        aqk_bd = _sbuf(rows, rows)
        nisa.tensor_tensor(
            dst=aqk_bd, data1=ps_qk, data2=in_block.causal[0:rows, 0:rows],
            op=nl.multiply,
        )
        if block is not None:
            # The cross block: Aqk += (qp f^T) * cross.
            ps_qf = _psum(rows, rows)
            nisa.nc_matmul(dst=ps_qf, stationary=qp_t, moving=f_t, accumulate=False)
            aqk_cross = _sbuf(rows, rows)
            nisa.tensor_tensor(
                dst=aqk_cross, data1=ps_qf, data2=block.cross[0:rows, 0:rows],
                op=nl.multiply,
            )
            aqk_all = _sbuf(rows, rows)
            nisa.tensor_tensor(dst=aqk_all, data1=aqk_bd, data2=aqk_cross, op=nl.add)
            aqk_bd = aqk_all
        _emit_pack(aqk_sb[0:rows, c0 : c0 + chunk], aqk_bd, rows, chunk)
        a_inv_t = ainv_sb[0:rows, c0 : c0 + chunk]
        _emit_unit_lower_inverse(a_inv_t, n_bd, layout, rows, chunk)
        _emit_uw(uw_sb[0:rows, t * width : (t + 1) * width], k_t,
                 v_sb[0:rows, t * vdim : (t + 1) * vdim], beta_col, egc_t, a_inv_t,
                 layout.same_chunk, rows, chunk, kdim, vdim)

    kg_sb = _sbuf(rows, wk)
    _emit_kg(kg_sb, k_sb, gc_sb, layout.last_row, rows, wk)

    uw3 = _blocks(uw_sb, rows, n_tiles, width)
    _store_tiles(u_rows, uw3[0:rows, 0:n_tiles, 0:vdim], r0, n_tiles, rows, vdim,
                 nisa.engine.sync)
    _store_tiles(w_rows, uw3[0:rows, 0:n_tiles, vdim:width], r0, n_tiles, rows, kdim,
                 nisa.engine.scalar)
    _store_tiles(kg_rows, _blocks(kg_sb, rows, n_tiles, kdim), r0, n_tiles, rows, kdim,
                 nisa.engine.sync)
    _store_tiles(ainv_rows, _blocks(ainv_sb, rows, n_tiles, chunk), r0, n_tiles, rows,
                 chunk, nisa.engine.scalar)
    _store_tiles(aqk_rows, _blocks(aqk_sb, rows, n_tiles, chunk), r0, n_tiles, rows,
                 chunk, nisa.engine.sync)


def _emit_stage3_tiles(
    sinks, sources, same, triu, last_row, r0, n_tiles, rows, chunk, kdim, vdim
):
    """Stage 3 alone for ``n_tiles`` packed tiles of ``rows`` rows from row ``r0``."""
    k_rows, v_rows, beta_rows, gk_rows, ainv_rows = sources
    w_rows, u_rows, kg_rows = sinks
    wk = n_tiles * kdim
    gk_sb = _load_tiles(gk_rows, r0, n_tiles, rows, kdim, nisa.engine.sync)
    k_raw = _load_tiles(k_rows, r0, n_tiles, rows, kdim, nisa.engine.scalar)
    v_sb = _load_tiles(v_rows, r0, n_tiles, rows, vdim, nisa.engine.sync)
    beta_sb = _load_tiles(beta_rows, r0, n_tiles, rows, 1, nisa.engine.scalar)
    a_inv_packed = _load_tiles(ainv_rows, r0, n_tiles, rows, chunk, nisa.engine.sync)
    k_sb = _sbuf(rows, wk)
    gc_sb = _sbuf(rows, wk)
    egc_sb = _sbuf(rows, wk)
    _emit_prepare(k_sb, gc_sb, egc_sb, k_raw, gk_sb, triu[0:rows, 0:rows], rows,
                  n_tiles, kdim)
    width = vdim + kdim
    uw_sb = _sbuf(rows, n_tiles * width)
    for t in range(n_tiles):
        k0 = t * kdim
        _emit_uw(uw_sb[0:rows, t * width : (t + 1) * width],
                 k_sb[0:rows, k0 : k0 + kdim],
                 v_sb[0:rows, t * vdim : (t + 1) * vdim], beta_sb[0:rows, t : t + 1],
                 egc_sb[0:rows, k0 : k0 + kdim],
                 a_inv_packed[0:rows, t * chunk : (t + 1) * chunk], same, rows, chunk,
                 kdim, vdim)
    kg_sb = _sbuf(rows, wk)
    _emit_kg(kg_sb, k_sb, gc_sb, last_row, rows, wk)
    uw3 = _blocks(uw_sb, rows, n_tiles, width)
    _store_tiles(u_rows, uw3[0:rows, 0:n_tiles, 0:vdim], r0, n_tiles, rows, vdim,
                 nisa.engine.sync)
    _store_tiles(w_rows, uw3[0:rows, 0:n_tiles, vdim:width], r0, n_tiles, rows, kdim,
                 nisa.engine.scalar)
    _store_tiles(kg_rows, _blocks(kg_sb, rows, n_tiles, kdim), r0, n_tiles, rows, kdim,
                 nisa.engine.sync)


@nki.jit
def kda_stage3_kernel(
    k_hbm, v_hbm, beta_hbm, gk_hbm, a_inv_hbm, triu_hbm, last_row_hbm
):
    """Stage 3 alone, taking the inverse as an argument.

    ``k`` and ``gk`` are ``[NC, C, K]``, ``v`` is ``[NC, C, V]``, ``beta`` is
    ``[NC, C, 1]``, ``a_inv`` is ``[NC, C, C]``, the two constants are ``[C, C]``;
    returns ``w``, ``u``, ``kg`` as :func:`kda_intra_chunk_kernel` does. The
    boundary upstream's ``recompute_w_u_fwd`` also uses; the body is
    :func:`_emit_uw` and :func:`_emit_kg`, the ones the combined entry point emits.

    Because ``u = T @ (beta * v)`` is linear in ``T``, scaling a row of the
    supplied inverse scales the same row of ``u``.
    """
    n_chunks, chunk, kdim = k_hbm.shape
    vdim = v_hbm.shape[2]
    rows = n_chunks * chunk
    layout_rows = min(tile_rows(chunk), rows)

    w_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    u_hbm = nl.ndarray((n_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    kg_hbm = nl.ndarray((n_chunks, chunk, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    sources = (
        _rows_view(k_hbm), _rows_view(v_hbm), _rows_view(beta_hbm), _rows_view(gk_hbm),
        _rows_view(a_inv_hbm),
    )
    sinks = (_rows_view(w_hbm), _rows_view(u_hbm), _rows_view(kg_hbm))

    same = _emit_same_chunk(layout_rows, chunk)
    rep = _emit_replicator(layout_rows, chunk)
    chunk_rows = _emit_chunk_rows((triu_hbm, last_row_hbm), rep, layout_rows, chunk,
                                  nisa.engine.scalar)
    triu = _emit_block_diagonal(chunk_rows, 0, same, layout_rows, chunk)
    last_row = _emit_block_diagonal(chunk_rows, 1, same, layout_rows, chunk)

    per_prog, first, tail = _program_tiles(rows, layout_rows)
    if per_prog:
        _emit_stage3_tiles(
            sinks, sources, same, triu, last_row, first * layout_rows, per_prog,
            layout_rows, chunk, kdim, vdim,
        )
    if tail:
        _emit_stage3_tiles(
            sinks, sources, same, triu, last_row, rows - tail, 1, tail, chunk, kdim,
            vdim,
        )

    return w_hbm, u_hbm, kg_hbm


def _require_admissible(
    n_chunks: int,
    chunk: int,
    kdim: int,
    vdim: int,
    gate_abs_max: float,
    gate_block: int | None = None,
    gate_max: float = 0.0,
) -> None:
    """Raise unless this kernel serves the input, naming every cause at once.

    ``gate_abs_max`` is the largest ``|gc|`` over one gate block of ``gate_block``
    tokens (``chunk`` when ``None``), and ``gate_max`` the largest gate; the module
    docstring's Gate blocks says why a chunk of two blocks needs both.
    """
    block = chunk if gate_block is None else gate_block
    problems: list[str] = []
    if n_chunks < 1:
        problems.append(f"n_chunks={n_chunks} must be at least 1")
    if chunk < 2 or chunk > MAX_TILE:
        problems.append(
            f"chunk={chunk} must be in [2, {MAX_TILE}]; the kernel maps the chunk "
            f"onto the partition axis and declares no tiling, so one (I + A) tile "
            f"must fit one partition tile"
        )
    elif chunk & (chunk - 1):
        problems.append(
            f"chunk={chunk} must be a power of two; each level of the block-merge "
            f"inverse doubles the width of the solved blocks, from 1 to the chunk"
        )
    if kdim < 1 or kdim > MAX_TILE:
        problems.append(
            f"kdim={kdim} must be in [1, {MAX_TILE}]; it is one matmul operand width"
        )
    if vdim < 1 or vdim > MAX_TILE:
        problems.append(
            f"vdim={vdim} must be in [1, {MAX_TILE}]; it is one matmul operand width"
        )
    if block != chunk and (block < 2 or 2 * block != chunk):
        problems.append(
            f"gate_block={block} must be chunk={chunk} or half of it, and at least "
            f"2; the products re-reference the gate at most once per chunk"
        )
    if gate_abs_max > GATE_CUMSUM_ABS_LIMIT:
        problems.append(
            f"max|cumulative gate|={gate_abs_max:.3f} exceeds "
            f"{GATE_CUMSUM_ABS_LIMIT}; the chunk-local products factorise the gate "
            f"difference over one gate block as exp(gc[t]) * exp(-gc[j]), so a "
            f"cumulative gate this far from zero would overflow fp32 in the larger "
            f"factor"
        )
    if block != chunk and gate_max > 0:
        problems.append(
            f"max gate={gate_max:.3f} is above 0; a chunk of two gate blocks "
            f"exponentiates gate sums over the whole chunk, which stay at most 0, "
            f"and so inside fp32, only for a non-positive gate"
        )
    if problems:
        raise ChunkedRecurrenceError(
            "kda_intra_chunk cannot serve this input: " + "; ".join(problems)
        )


def intra_chunk_grid(n_chunks: int, chunk: int) -> tuple[int, ...]:
    """The launch grid of :func:`kda_intra_chunk_kernel` for this geometry.

    ``(LNC2_PROGRAMS,)`` on an LNC2 runtime (``NEURON_LOGICAL_NC_CONFIG=2``: two
    physical cores per logical core) when the packed tiles are all whole and split
    evenly over the programs, else ``()`` (one program). Each program then owns a
    contiguous run of whole tiles.
    """
    if not lnc_pair():
        return ()
    rows = n_chunks * chunk
    full, tail = divmod(rows, tile_rows(chunk))
    if tail or full % LNC2_PROGRAMS:
        return ()
    return (LNC2_PROGRAMS,)


def can_run_intra_chunk(
    reference: Tensor,
    n_chunks: int,
    chunk: int,
    kdim: int,
    vdim: int,
    gate_abs_max: float,
    gate_block: int | None = None,
    gate_max: float = 0.0,
) -> bool:
    """Is the NKI path available *and* admissible for this input?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_admissible` whether this kernel accepts
    these extents, this gate block and this gate range.
    """
    _require_admissible(
        n_chunks, chunk, kdim, vdim, gate_abs_max, gate_block, gate_max
    )
    return can_run_kernel(reference)


def kda_intra_chunk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gk: Tensor,
    *,
    gate_block: int | None = None,
) -> IntraChunkOutputs:
    """Stages 1 to 3 over every chunk of one head or of several, in one kernel dispatch.

    Args:
        q, k, gk: ``[NC, C, K]`` fp32. ``gk`` is the per-key-channel log gate,
            not yet accumulated; the kernel forms the cumulative sum. Every
            operand may carry a leading heads axis instead, ``[H, NC, C, K]``
            here and ``[H, NC, C]`` for ``beta``; then all of them do.
        v: ``[NC, C, V]`` fp32.
        beta: ``[NC, C]`` fp32, one scalar per token.
        gate_block: the tokens of one gate block, ``C`` or ``C / 2`` (module
            docstring, Gate blocks); ``None`` is ``C``. The gate limit applies to
            one block's gate sum, and ``C / 2`` also needs a non-positive gate.

    Returns:
        :class:`IntraChunkOutputs`, every field with the heads axis in front when
        the operands carry one. Stages 1 to 3 are chunk-local, so the heads' chunks
        run as the chunks of one call, and head ``h`` of every field is what a
        one-head call on head ``h``'s operands returns.

    Raises:
        ChunkedRecurrenceError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.
    """
    if q.dim() == 4:
        return _intra_chunk_heads(q, k, v, beta, gk, gate_block)
    ranks = (("q", q, 3), ("k", k, 3), ("v", v, 3), ("gk", gk, 3), ("beta", beta, 2))
    for name, tensor, rank in ranks:
        if tensor.dim() != rank:
            raise ChunkedRecurrenceError(
                f"{name} must be {rank}-D, got shape {tuple(tensor.shape)}"
            )
    n_chunks, chunk, kdim = (int(x) for x in q.shape)
    vdim = int(v.shape[2])
    if tuple(k.shape) != (n_chunks, chunk, kdim) or tuple(gk.shape) != (
        n_chunks,
        chunk,
        kdim,
    ):
        raise ChunkedRecurrenceError(
            f"k {tuple(k.shape)} and gk {tuple(gk.shape)} must both match q "
            f"{tuple(q.shape)}"
        )
    if tuple(v.shape)[:2] != (n_chunks, chunk) or tuple(beta.shape) != (
        n_chunks,
        chunk,
    ):
        raise ChunkedRecurrenceError(
            f"v {tuple(v.shape)} and beta {tuple(beta.shape)} must agree with q's "
            f"leading dimensions {(n_chunks, chunk)}"
        )

    # The gate range is an eager-call precondition, because forming it reads the
    # gate values. A traced or meta-built call has none to read and passes 0.0,
    # which is inside the limit and refuses nothing. Only `can_run_kernel` decides
    # the path, so a graph build loses the diagnostic message and nothing else.
    block = chunk if gate_block is None else int(gate_block)
    gate_abs_max = gate_max = 0.0
    if values_are_readable(gk) and block >= 1 and chunk % block == 0:
        blocks = gk.float().reshape(n_chunks, chunk // block, block, kdim)
        gate_abs_max = float(blocks.cumsum(dim=2).abs().max().item())
        if block != chunk:
            gate_max = float(gk.max().item())
    if not can_run_intra_chunk(
        q, n_chunks, chunk, kdim, vdim, gate_abs_max, block, gate_max
    ):
        _count_torch_fallback()
        logger.debug(
            "kda_intra_chunk: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_intra_chunk_torch_oracle(q, k, v, beta, gk)

    consts = chunk_constants(chunk, device=q.device, dtype=q.dtype)
    # One gate block passes no block operands, so that call is the one-block call.
    blocked = {}
    if block != chunk:
        block_consts = chunk_constants(block, device=q.device, dtype=q.dtype)
        blocked = {
            "block_triu_hbm": block_consts.triu_ones,
            "block_mask_lower_hbm": block_consts.mask_lower,
            "block_last_row_hbm": block_consts.last_row,
        }
    _count_nki_dispatch()
    call = wrap_nki(kda_intra_chunk_kernel)
    grid = intra_chunk_grid(n_chunks, chunk)
    if grid:
        call = call[grid]
    w, u, kg, a_inv, aqk = call(
        q_hbm=q,
        k_hbm=k,
        v_hbm=v,
        beta_hbm=beta.unsqueeze(-1).contiguous(),
        gk_hbm=gk,
        triu_hbm=consts.triu_ones,
        eye_hbm=consts.eye,
        mask_lower_hbm=consts.mask_lower,
        last_row_hbm=consts.last_row,
        **blocked,
    )
    return IntraChunkOutputs(w=w, u=u, kg=kg, a_inv=a_inv, aqk=aqk)


def _intra_chunk_heads(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor,
    gate_block: int | None,
) -> IntraChunkOutputs:
    """:func:`kda_intra_chunk` of operands with a leading heads axis.

    The heads' chunks are laid end to end on the chunk axis, head-major, and run
    as one call's chunks; each output goes back to ``[H, NC, ...]``.
    """
    ranks = (("q", q, 4), ("k", k, 4), ("v", v, 4), ("gk", gk, 4), ("beta", beta, 3))
    for name, tensor, rank in ranks:
        if tensor.dim() != rank:
            raise ChunkedRecurrenceError(
                f"{name} must be {rank}-D with a heads axis like q, got shape "
                f"{tuple(tensor.shape)}"
            )
    lead = tuple(int(x) for x in q.shape[:2])
    for name, tensor in (("k", k), ("v", v), ("gk", gk), ("beta", beta)):
        if tuple(tensor.shape)[:2] != lead:
            raise ChunkedRecurrenceError(
                f"{name} {tuple(tensor.shape)} must lead with q's [heads, NC] {lead}"
            )
    heads, n_chunks = lead
    flat = kda_intra_chunk(
        *(t.reshape((heads * n_chunks,) + tuple(t.shape[2:])) for t in (q, k, v, beta, gk)),
        gate_block=gate_block,
    )
    return IntraChunkOutputs(
        *(field.reshape(lead + tuple(field.shape[1:])) for field in flat)
    )


def kda_intra_chunk_torch_oracle(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> IntraChunkOutputs:
    """Stages 1 to 3 in torch, by different means than the kernel uses.

    Not mirroring the kernel is the point, so three differences are deliberate:

    * the gate difference is evaluated directly as ``exp(gc[t] - gc[j])`` over the
      whole chunk (:func:`_causal_gate_decay`), where the kernel factorises it into
      ``exp(gc[t]) * exp(-gc[j])`` over one gate block so one matmul can contract
      the channel axis. That makes this the better-conditioned of the two, which
      is what :data:`GATE_CUMSUM_ABS_LIMIT` keeps honest;
    * the inverse comes from ``torch.linalg.solve_triangular``, forward
      substitution, where the kernel merges solved diagonal blocks;
    * the two products are formed by broadcast sums rather than by transposes and
      matmuls.
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32, gk32 = beta.float(), gk.float()
    n_chunks, chunk, kdim = q32.shape

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)

    gc = gk32.cumsum(dim=1)
    # decay[n, t, j, c] = exp(gc[n, t, c] - gc[n, j, c]) for t >= j -- formed directly.
    decay = _causal_gate_decay(gc)

    idx = torch.arange(chunk, device=q32.device)
    strictly_lower = (idx.unsqueeze(0) < idx.unsqueeze(1)).to(q32.dtype)
    causal = (idx.unsqueeze(0) <= idx.unsqueeze(1)).to(q32.dtype)

    beta_col = beta32.unsqueeze(-1)
    kk = (kn.unsqueeze(2) * kn.unsqueeze(1) * decay).sum(-1)
    a = kk * beta_col * strictly_lower
    aqk = (qn.unsqueeze(2) * kn.unsqueeze(1) * decay).sum(-1) * causal

    eye = torch.eye(chunk, device=q32.device, dtype=q32.dtype)
    a_inv = torch.linalg.solve_triangular(
        eye + a, eye.expand(n_chunks, chunk, chunk), upper=False, unitriangular=True
    )

    u = a_inv @ (beta_col * v32)
    w = a_inv @ (beta_col * kn * torch.exp(gc))
    kg = kn * torch.exp(gc[:, -1:, :] - gc)
    return IntraChunkOutputs(w=w, u=u, kg=kg, a_inv=a_inv, aqk=aqk)


def _causal_gate_decay(gc: Tensor) -> Tensor:
    """``exp(gc[t] - gc[j])`` for ``t >= j`` and 0 above the diagonal.

    ``gc`` is ``[NC, C, K]``; returns ``[NC, C, C, K]``. The exponent is masked
    before it is exponentiated: above the diagonal it is a gate sum with the sign
    reversed, which a wide chunk takes out of fp32's range, and an infinite
    entry would turn the mask's zero into NaN.
    """
    idx = torch.arange(gc.shape[1], device=gc.device)
    above = (idx.unsqueeze(0) > idx.unsqueeze(1)).unsqueeze(-1)
    diff = gc.unsqueeze(2) - gc.unsqueeze(1)
    return torch.exp(diff.masked_fill(above, float("-inf")))


def rebuild_i_plus_a(k: Tensor, beta: Tensor, gk: Tensor) -> Tensor:
    """``I + A`` from the same inputs, for checking an inverse.

    Separate from :func:`kda_intra_chunk_torch_oracle` so that a caller can
    multiply the kernel's own inverse by a reference ``(I + A)`` without depending
    on any other reference value.
    """
    k32, beta32, gk32 = k.float(), beta.float(), gk.float()
    _, chunk, _ = k32.shape
    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    gc = gk32.cumsum(dim=1)
    decay = _causal_gate_decay(gc)
    idx = torch.arange(chunk, device=k32.device)
    strictly_lower = (idx.unsqueeze(0) < idx.unsqueeze(1)).to(k32.dtype)
    kk = (kn.unsqueeze(2) * kn.unsqueeze(1) * decay).sum(-1)
    a = kk * beta32.unsqueeze(-1) * strictly_lower
    return torch.eye(chunk, device=k32.device, dtype=k32.dtype) + a


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the intra-chunk kernel this module authors.

    Read off the unwrapped ``.func``, since ``nki.jit``'s wrapper reports its own
    module either way.
    """
    func = getattr(kda_intra_chunk_kernel, "func", None)
    target = func if func is not None else kda_intra_chunk_kernel
    return target.__module__, target.__qualname__


def stage3_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the stage-3 entry this module authors."""
    func = getattr(kda_stage3_kernel, "func", None)
    target = func if func is not None else kda_stage3_kernel
    return target.__module__, target.__qualname__


# Stages 4 and 5: the state carried across chunks, and the output.
#
# These share this file and the emitting helpers above with stages 1 to 3, and
# nothing else: separate kernel entry points, separate entry functions, separate
# counters, separate constants, separate admissibility checks.
#
# The output here is not ``o = H q``. That is the sequential formula and it
# belongs to the oracle. The chunked output is the gate-decayed inter-chunk part
# ``qg @ h_chunk`` plus the intra-chunk part ``Aqk @ v_new``. The two forms agree
# numerically, so writing the kernel in the oracle's form would make the
# comparison between them circular.
#
# One launch serves several heads: each head is its own chain over its own
# operands, and the heads share nothing but the launch. A state-only launch runs
# the same chains and leaves out what only the output reads, for a caller that
# needs the state a sequence leaves before it needs that sequence's output.


class InterChunkOutputs(NamedTuple):
    """The three values stages 4 and 5 produce.

    ``final_state`` is ``[V, K]``, the sequential scan's own orientation, so a
    comparison against it needs no transpose on either side. The kernel carries
    the state transposed internally, for the reason given on
    :func:`kda_inter_chunk_kernel`, and pays one transpose at the end. A call
    with a heads axis returns every field with that axis in front.

    ``v_new`` is a side output: nothing here consumes it, and it makes stage 4's
    derivation of it from ``u`` readable without reading the kernel body.

    A state-only call (``kda_inter_chunk(..., state_only=True)``) computes
    ``final_state`` alone, and ``o`` and ``v_new`` are ``None``.
    """

    o: Tensor | None
    final_state: Tensor
    v_new: Tensor | None


class SequentialOutputs(NamedTuple):
    """What the sequential oracle returns: flat ``o`` and the final state."""

    o: Tensor
    final_state: Tensor


class InterChunkConstants(NamedTuple):
    """The host-built constants the inter-chunk kernel takes.

    None is shared with the intra-chunk set, so a change to one cannot reach the
    other.
    """

    triu_ones: Tensor
    last_col: Tensor
    state_init: Tensor


@dataclass
class _InterDispatchCounters:
    """The inter-chunk path's own counters, never the intra-chunk pair's.

    Same shape and same per-dispatch rule as :class:`_DispatchCounters`, in a
    separate object so neither reading can pollute the other. The chunk loop is
    inside the kernel, so a host-side loop that dispatched once per chunk would
    read the chunk count rather than ``1``.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. A separate
#: object from :data:`_COUNTERS`; the two are never aliased.
_INTER_COUNTERS = _InterDispatchCounters()


def reset_inter_dispatch_counters() -> None:
    """Zero both inter-chunk counters."""
    _INTER_COUNTERS.nki_dispatch = 0
    _INTER_COUNTERS.torch_fallback = 0


def inter_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last inter-chunk reset."""
    return _INTER_COUNTERS.nki_dispatch, _INTER_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_inter_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _INTER_COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_inter_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _INTER_COUNTERS.nki_dispatch += 1


def inter_chunk_constants(
    chunk: int,
    kdim: int,
    vdim: int,
    *,
    heads: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> InterChunkConstants:
    """Build the constants the inter-chunk kernel takes.

    * ``triu_ones[s, t] = 1 for s <= t``. As a matmul stationary operand its
      transpose is the lower-inclusive ones matrix, so ``triu_ones^T @ gk`` is the
      inclusive cumulative gate along tokens within the chunk.
    * ``last_col[t, 0] = 1 for t == chunk - 1``. As a stationary operand's moving
      partner it selects the chunk's last cumulative-gate row and lands it as a
      ``[K, 1]`` column, which is the shape the per-key-channel state decay needs:
      one matmul, no transpose. That is why this constant is a column where the
      intra-chunk row selector is a full ``[C, C]`` tile.
    * ``state_init``, the zero entering state of ``heads`` heads, ``[heads * V, K]``
      with head ``h`` in rows ``h * V`` onward (``[V, K]`` for one head), built on
      the host and loaded rather than zeroed on device. Being an argument, it is
      also how a caller passes a non-zero entering state. It is stated in the
      orientation :attr:`InterChunkOutputs.final_state` returns, so a caller can
      loop one call's output into the next with no transpose of its own.
    """
    idx = torch.arange(chunk, device=device)
    return InterChunkConstants(
        triu_ones=(idx.unsqueeze(1) <= idx.unsqueeze(0)).to(dtype),
        last_col=(idx == (chunk - 1)).to(dtype).unsqueeze(1).contiguous(),
        state_init=torch.zeros(heads * vdim, kdim, device=device, dtype=dtype),
    )


@dataclass(frozen=True)
class _InterLayout(nl.NKIObject):
    """The packed-tile constants of the inter-chunk kernel, built once per launch.

    ``triu`` is the block-diagonal upper-inclusive ones, so one matmul forms every
    chunk's cumulative gate in the tile; ``last_sel[r, g]`` is 1 where ``r`` is
    chunk ``g``'s last token, so one matmul against it lands every chunk's last
    cumulative-gate row as a ``[K, 1]`` decay column.
    """

    triu: object
    last_sel: object


def _emit_inter_layout(triu_hbm, last_col_hbm, rows, chunk):
    """:class:`_InterLayout` for a packed tile of ``rows`` rows."""
    groups = rows // chunk
    same = _emit_same_chunk(rows, chunk)
    rep = _emit_replicator(rows, chunk)
    chunk_rows = _emit_chunk_rows((triu_hbm,), rep, rows, chunk, nisa.engine.sync)
    triu = _emit_block_diagonal(chunk_rows, 0, same, rows, chunk)
    last_col = _sbuf(chunk, 1)
    _dma(last_col, last_col_hbm, nisa.engine.scalar)
    ps_last = _psum(rows, 1)
    nisa.nc_matmul(
        dst=ps_last, stationary=rep.reshape((chunk, rows)), moving=last_col,
        accumulate=False,
    )
    last_rows = _sbuf(rows, 1)
    nisa.tensor_copy(dst=last_rows, src=ps_last)
    last_sel = _sbuf(rows, groups)
    nisa.tensor_scalar(
        dst=last_sel, data=same[0:rows, 0:groups, 0], op0=nl.multiply,
        operand0=last_rows,
    )
    return _InterLayout(triu, last_sel)


def _emit_inter_operands(sources, layout, g0, groups, chunk, kdim, v0, vpart, scale,
                         with_output):
    """The chain's per-chunk operands for the ``groups`` chunks starting at ``g0``.

    Everything here is off the carried state's critical path; the chain only
    reads it. The row-layout operands (``gk``, ``q``, ``w``, ``aqk``) are loaded
    one packed tile at a time; ``kg`` and this program's ``u`` columns are loaded
    chunk-local, ``[C, groups, *]``, because they enter the chain as the
    partition-axis operands of chunk-sized matmuls. ``g0`` counts chunks over
    every head's, so it also names the head.

    Returns ``(w_t, qg_t, aqk_t, decay, kg_c, u_c)``: per chunk ``g``, the
    stationary ``w^T``, ``qg^T`` and ``aqk^T`` at ``[:, g]``, and ``decay[:, g]``
    the state decay column. Without ``with_output`` only the chain's operands are
    formed, ``q`` and ``aqk`` are not read, and ``qg_t`` and ``aqk_t`` are
    ``None``.
    """
    kg_hbm, w_rows, u_hbm, gk_rows, q_rows, aqk_rows = sources
    r0 = g0 * chunk
    count = groups * chunk

    gk_sb = _load_tiles(gk_rows, r0, 1, count, kdim, nisa.engine.sync)
    gc_sb = _sbuf(count, kdim)
    _emit_gate_cumsum(gc_sb, gk_sb, layout.triu[0:count, 0:count], count, kdim)
    qg_t = aqk_t = None
    if with_output:
        egc_sb = _sbuf(count, kdim)
        nisa.activation(dst=egc_sb, data=gc_sb, op=nl.exp)

        # `q` arrives raw and is normalised and scaled here, as stage 2 does.
        q_raw = _load_tiles(q_rows, r0, 1, count, kdim, nisa.engine.scalar)
        q_norm = _sbuf(count, kdim)
        _emit_l2_normalise(q_norm, q_raw, count, kdim)
        q_sb = _sbuf(count, kdim)
        nisa.tensor_scalar(dst=q_sb, data=q_norm, op0=nl.multiply, operand0=scale)
        qg_sb = _sbuf(count, kdim)
        nisa.tensor_tensor(dst=qg_sb, data1=q_sb, data2=egc_sb, op=nl.multiply)

    w_sb = _load_tiles(w_rows, r0, 1, count, kdim, nisa.engine.sync)
    w_t = nl.ndarray((kdim, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
    ps_w = _psum(kdim, count)
    nisa.nc_transpose(dst=ps_w, data=w_sb)
    nisa.tensor_copy(dst=w_t, src=ps_w.reshape((kdim, groups, chunk)))
    if with_output:
        qg_t = nl.ndarray((kdim, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
        ps_q = _psum(kdim, count)
        nisa.nc_transpose(dst=ps_q, data=qg_sb)
        nisa.tensor_copy(dst=qg_t, src=ps_q.reshape((kdim, groups, chunk)))

        aqk_sb = _load_tiles(aqk_rows, r0, 1, count, chunk, nisa.engine.scalar)
        aqk_t = nl.ndarray((chunk, groups, chunk), dtype=nl.float32, buffer=nl.sbuf)
        ps_a = _psum(chunk, count)
        nisa.nc_transpose(dst=ps_a, data=aqk_sb)
        nisa.tensor_copy(dst=aqk_t, src=ps_a.reshape((chunk, groups, chunk)))

    ps_d = _psum(kdim, groups)
    nisa.nc_matmul(
        dst=ps_d, stationary=gc_sb, moving=layout.last_sel[0:count, 0:groups],
        accumulate=False,
    )
    decay = _sbuf(kdim, groups)
    nisa.activation(dst=decay, data=ps_d, op=nl.exp)

    kg_c = nl.ndarray((chunk, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    _dma(kg_c, kg_hbm[g0 : g0 + groups, 0:chunk, 0:kdim].permute((1, 0, 2)),
         nisa.engine.sync)
    u_c = nl.ndarray((chunk, groups, vpart), dtype=nl.float32, buffer=nl.sbuf)
    _dma(u_c, u_hbm[g0 : g0 + groups, 0:chunk, nl.ds(v0, vpart)].permute((1, 0, 2)),
         nisa.engine.scalar)
    return w_t, qg_t, aqk_t, decay, kg_c, u_c


def _emit_inter_step(ht, operands, g, vnew_c, o_c, chunk, kdim, vpart):
    """One chunk of stages 4 and 5 on this program's ``vpart`` state columns.

    ``ht`` is the entering state ``[K, vpart]``; returns the leaving one. The
    carried chain is ``w @ ht``, ``v_new``, ``kg^T @ v_new`` and the decayed sum.
    ``o``'s two products read the same entering state and ``v_new`` but feed
    nothing the chain reads, so they are separate matmuls issued after it: a
    product merged into the chain's matmul would put its stationary columns on
    every step's weight load. ``o_c`` is ``None`` in a state-only launch, and the
    step is then the chain alone.
    """
    w_t, qg_t, aqk_t, decay, kg_c, u_c = operands

    # Stage 4: v_new = u - w @ ht, on the entering state.
    ps_x = _psum(chunk, vpart)
    nisa.nc_matmul(dst=ps_x, stationary=w_t[0:kdim, g, 0:chunk], moving=ht,
                   accumulate=False)
    vnew = vnew_c[0:chunk, g, 0:vpart]
    nisa.tensor_tensor(
        dst=vnew, data1=u_c[0:chunk, g, 0:vpart], data2=ps_x, op=nl.subtract
    )

    # The carry: ht <- ht * exp(gc[C - 1]) + kg^T @ v_new, the product and the
    # sum in one vector instruction, so the step's last instruction waits on the
    # tensor engine alone.
    ps_h = _psum(kdim, vpart)
    nisa.nc_matmul(
        dst=ps_h, stationary=kg_c[0:chunk, g, 0:kdim], moving=vnew, accumulate=False
    )
    ht_next = _sbuf(kdim, vpart)
    nisa.scalar_tensor_tensor(
        dst=ht_next, data=ht, op0=nl.multiply, operand0=decay[0:kdim, g : g + 1],
        op1=nl.add, operand1=ps_h,
    )
    if o_c is None:
        return ht_next

    # Stage 5, off the chain: o = qg @ ht + aqk @ v_new, also on the entering
    # state, each product rounded on its own and then added. One addend leaves
    # PSUM first: an elementwise instruction reads one PSUM operand.
    ps_q = _psum(chunk, vpart)
    nisa.nc_matmul(dst=ps_q, stationary=qg_t[0:kdim, g, 0:chunk], moving=ht,
                   accumulate=False)
    o_rows = o_c[0:chunk, g, 0:vpart]
    nisa.tensor_copy(dst=o_rows, src=ps_q)
    ps_a = _psum(chunk, vpart)
    nisa.nc_matmul(dst=ps_a, stationary=aqk_t[0:chunk, g, 0:chunk], moving=vnew,
                   accumulate=False)
    nisa.tensor_tensor(dst=o_rows, data1=o_rows, data2=ps_a, op=nl.add)
    return ht_next


@nki.jit
def kda_inter_chunk_kernel(
    kg_hbm, w_hbm, u_hbm, gk_hbm, q_hbm, aqk_hbm, triu_hbm, last_col_hbm,
    state_init_hbm, HEADS: int = 1, STATE_ONLY: bool = False,
):
    """Stages 4 and 5 for every chunk of ``HEADS`` heads, in one dispatch.

    ``kg``, ``w``, ``gk`` and ``q`` are ``[HEADS * NC, C, K]``; ``u`` is
    ``[HEADS * NC, C, V]``; ``aqk`` is ``[HEADS * NC, C, C]``; head ``h`` owns
    chunks ``h * NC`` to ``(h + 1) * NC - 1`` of each. ``triu`` is ``[C, C]``;
    ``last_col`` is ``[C, 1]``; ``state_init`` is ``[HEADS * V, K]``, head ``h``'s
    entering state in rows ``h * V`` onward; all float32. Returns
    ``o [HEADS * NC, C, V]``, ``final_state [HEADS * V, K]`` and
    ``v_new [HEADS * NC, C, V]``, float32, in the same head order. With
    ``STATE_ONLY`` it returns ``final_state`` alone and reads neither ``q`` nor
    ``aqk``. One head (``HEADS = 1``) is ``[NC, C, *]`` and ``[V, K]``.

    There is no raw ``v`` argument: ``v_new`` is derived from ``u`` inside stage 4.

    The state is carried transposed, as ``ht`` shaped ``[K, V]``, where upstream
    carries ``[V, K]``. Two things follow: ``nc_matmul(stationary=kg,
    moving=v_new)`` computes ``kg^T @ v_new`` and lands ``[K, V]`` directly, so
    the update needs no transpose; and the per-key-channel decay becomes a
    ``[K, 1]`` operand broadcast along the free axis, which is what
    ``nisa.tensor_scalar`` serves. The transpose back to ``[V, K]`` is paid once
    after the loop, and the entering state is turned the same way once before it.

    Each chunk depends on the state the previous one leaves, so the chunk loop is
    a chain. Everything the chain reads but does not carry is formed per packed
    tile of ``MAX_TILE // C`` chunks (:func:`_emit_inter_operands`), so a chain
    step is four instructions on the state: ``w @ ht``, ``v_new``,
    ``kg^T @ v_new`` and the decayed sum; ``o``'s two products hang off it
    (:func:`_emit_inter_step`).

    Each head is its own chain on its own operands. Value columns are
    independent, so a launch of ``P`` programs gives each program ``V / P`` value
    columns of every head (an LNC2 pair: ``V / 2`` each), the split a one-head
    launch makes. Each head's chain in a program is then the one-head launch's
    program on that head's operands, instruction for instruction and at the same
    matmul widths, so a head's results do not depend on the heads beside it. Not
    whole heads per program: a matmul's result columns can depend on the moving
    operand's width (BIRSim rounds the last 8 columns of a 128-wide one
    differently from a 64-wide one), so ``V`` columns per program would not be
    the one-head launch's arithmetic. A program steps its heads' chains in turn,
    chunk by chunk, so one chain's matmuls can fill the other's vector waits.

    ``STATE_ONLY`` leaves out what only ``o`` needs: ``q``'s normalisation and
    gate, the transposes of ``qg`` and ``aqk``, each step's two output products,
    and the ``o`` and ``v_new`` stores. The chain's own instructions do not
    change, so its ``final_state`` is the full launch's.
    """
    total_chunks, chunk, kdim = kg_hbm.shape
    vdim = u_hbm.shape[2]
    scale = float(kdim) ** -0.5
    n_prog = nl.num_programs(0)
    # The NKI front end admits ``assert`` and no ``raise``.
    assert total_chunks % HEADS == 0, "every head holds the same number of chunks"
    n_chunks = total_chunks // HEADS
    rows = n_chunks * chunk
    layout_rows = min(tile_rows(chunk), rows)
    tile_chunks = layout_rows // chunk
    assert vdim % n_prog == 0, "the value width does not split over the programs"
    vpart = vdim // n_prog
    v0 = nl.program_id(0) * vpart
    with_output = not STATE_ONLY

    o_hbm = vnew_hbm = None
    if with_output:
        o_hbm = nl.ndarray(
            (total_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm
        )
        vnew_hbm = nl.ndarray(
            (total_chunks, chunk, vdim), dtype=nl.float32, buffer=nl.shared_hbm
        )
    state_hbm = nl.ndarray((HEADS * vdim, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    sources = (
        kg_hbm, _rows_view(w_hbm), u_hbm, _rows_view(gk_hbm), _rows_view(q_hbm),
        _rows_view(aqk_hbm),
    )
    layout = _emit_inter_layout(triu_hbm, last_col_hbm, layout_rows, chunk)

    # The entering states arrive in the orientation this kernel returns and are
    # turned here, so the conversion is emitted on this engine rather than by
    # torch on the host.
    carried = []
    for head in range(HEADS):
        entering = _sbuf(vpart, kdim)
        _dma(entering, state_init_hbm[nl.ds(head * vdim + v0, vpart), 0:kdim],
             nisa.engine.scalar)
        ht = _sbuf(kdim, vpart)
        _emit_transpose(ht, entering, vpart, kdim)
        carried.append(ht)

    for g0 in range(0, n_chunks, tile_chunks):
        groups = min(tile_chunks, n_chunks - g0)
        per_head = []
        for head in range(HEADS):
            c0 = head * n_chunks + g0
            operands = _emit_inter_operands(
                sources, layout, c0, groups, chunk, kdim, v0, vpart, scale, with_output
            )
            vnew_c = nl.ndarray((chunk, groups, vpart), dtype=nl.float32, buffer=nl.sbuf)
            o_c = None
            if with_output:
                o_c = nl.ndarray((chunk, groups, vpart), dtype=nl.float32,
                                 buffer=nl.sbuf)
            per_head.append((c0, operands, vnew_c, o_c))
        for g in range(groups):
            for head in range(HEADS):
                _, operands, vnew_c, o_c = per_head[head]
                carried[head] = _emit_inter_step(
                    carried[head], operands, g, vnew_c, o_c, chunk, kdim, vpart
                )
        if with_output:
            for head in range(HEADS):
                c0, _, vnew_c, o_c = per_head[head]
                _dma(vnew_hbm[c0 : c0 + groups, 0:chunk, nl.ds(v0, vpart)]
                     .permute((1, 0, 2)), vnew_c, nisa.engine.sync)
                _dma(o_hbm[c0 : c0 + groups, 0:chunk, nl.ds(v0, vpart)]
                     .permute((1, 0, 2)), o_c, nisa.engine.scalar)

    for head in range(HEADS):
        leaving = _sbuf(vpart, kdim)
        _emit_transpose(leaving, carried[head], kdim, vpart)
        _dma(state_hbm[nl.ds(head * vdim + v0, vpart), 0:kdim], leaving,
             nisa.engine.sync)
    if STATE_ONLY:
        return state_hbm
    return o_hbm, state_hbm, vnew_hbm


def _require_inter_admissible(
    n_chunks: int, chunk: int, kdim: int, vdim: int, gate_peak: float,
    heads: int = 1,
) -> None:
    """Raise unless the inter-chunk kernel serves the input.

    Unlike :func:`_require_admissible` this does not require ``chunk`` to be a
    power of two: that requirement comes from the intra-chunk kernel's block-merge
    inverse, and this kernel inverts nothing.

    ``gate_peak`` is the largest chunk-local cumulative gate, signed, so the bound
    applies to one chunk's gate sum and not to the whole sequence's. See
    :func:`kda_inter_chunk` for why that distinction keeps the state carry inside
    fp32. The size of a negative sum is not bounded: this kernel exponentiates the
    cumulative gate with one sign only, and a negative exponent goes toward zero,
    which is the value it approximates. ``n_chunks`` counts one head's chunks.
    """
    problems: list[str] = []
    if heads < 1:
        problems.append(f"heads={heads} must be at least 1")
    if n_chunks < 1:
        problems.append(f"n_chunks={n_chunks} must be at least 1")
    if chunk < 2 or chunk > MAX_TILE:
        problems.append(
            f"chunk={chunk} must be in [2, {MAX_TILE}]; the kernel maps the chunk "
            f"onto the partition axis and declares no tiling"
        )
    if kdim < 1 or kdim > MAX_TILE:
        problems.append(
            f"kdim={kdim} must be in [1, {MAX_TILE}]; it is the carried state's "
            f"partition extent"
        )
    if vdim < 1 or vdim > MAX_TILE:
        problems.append(
            f"vdim={vdim} must be in [1, {MAX_TILE}]; it is the carried state's "
            f"free extent"
        )
    if gate_peak > GATE_CUMSUM_ABS_LIMIT:
        problems.append(
            f"max chunk-local cumulative gate={gate_peak:.3f} exceeds "
            f"{GATE_CUMSUM_ABS_LIMIT}; the state decay and the output gate are "
            f"both exp of that quantity, so a chunk-local cumulative gate this "
            f"far above zero would overflow fp32"
        )
    if problems:
        raise ChunkedRecurrenceError(
            "kda_inter_chunk cannot serve this input: " + "; ".join(problems)
        )


def inter_chunk_grid(vdim: int) -> tuple[int, ...]:
    """The launch grid of :func:`kda_inter_chunk_kernel` for this value width.

    ``(LNC2_PROGRAMS,)`` on an LNC2 runtime (``NEURON_LOGICAL_NC_CONFIG=2``) when
    the value columns split evenly over the programs, else ``()``. Each program
    carries its own value columns of the state through the same chain.
    """
    if not lnc_pair() or vdim % LNC2_PROGRAMS:
        return ()
    return (LNC2_PROGRAMS,)


def can_run_inter_chunk(
    reference: Tensor,
    n_chunks: int,
    chunk: int,
    kdim: int,
    vdim: int,
    gate_peak: float,
    heads: int = 1,
) -> bool:
    """Is the NKI path available *and* admissible for this inter-chunk input?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_inter_admissible` whether this kernel
    accepts these extents and this gate range.
    """
    _require_inter_admissible(n_chunks, chunk, kdim, vdim, gate_peak, heads)
    return can_run_kernel(reference)


def kda_inter_chunk(
    kg: Tensor,
    w: Tensor,
    u: Tensor,
    gk: Tensor,
    q: Tensor,
    aqk: Tensor,
    *,
    state: Tensor | None = None,
    state_only: bool = False,
) -> InterChunkOutputs:
    """Stages 4 and 5 over every chunk of one head or of several, in one kernel dispatch.

    ``kg``, ``w``, ``u`` and ``aqk`` are :func:`kda_intra_chunk`'s returns; ``gk``
    and ``q`` are the same raw inputs it took. There is no raw ``v``.

    Args:
        kg: ``[NC, C, K]`` fp32, the gated key. Every operand may carry a leading
            heads axis instead, ``[H, NC, C, K]`` here; then all of them do, and
            each head is its own sequence.
        w: ``[NC, C, K]`` fp32, the WY row factor.
        u: ``[NC, C, V]`` fp32, the WY value factor. ``v_new`` is derived from
            this inside stage 4.
        gk: ``[NC, C, K]`` fp32, the per-key-channel log gate, not yet
            accumulated; the kernel forms the chunk-local cumulative sum.
        q: ``[NC, C, K]`` fp32, raw. Normalised and scaled inside the kernel.
        aqk: ``[NC, C, C]`` fp32, already carrying the scale and the causal mask.
        state: ``[V, K]`` entering recurrent state (``[H, V, K]`` with heads), in
            ``final_state``'s own orientation, or ``None`` for a zero entry. A
            caller whose tokens arrive across several calls passes the state the
            previous call returned.
        state_only: compute the final state alone, for a caller that needs the
            state a sequence leaves before it needs that sequence's output. The
            chain is the full call's, so ``final_state`` is too; ``q`` and
            ``aqk`` are checked but not read, and ``o`` and ``v_new`` are
            ``None``.

    Returns:
        :class:`InterChunkOutputs`, whose ``final_state`` is ``[V, K]``
        (``[H, V, K]`` with heads). With heads, head ``h`` of every field is what
        a one-head call on head ``h``'s operands returns: the heads are separate
        chains, and each program runs each head's chain with the one-head call's
        instructions (:func:`kda_inter_chunk_kernel`).

    Raises:
        ChunkedRecurrenceError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.

    The state carry cannot overflow with sequence length, by construction rather
    than by a wider bound. A carry written over the whole sequence's cumulative
    gate would exponentiate a quantity that grows with the token count. This one
    decays by ``exp(gc[C - 1])`` where ``gc`` is re-referenced from zero inside
    every chunk, which is upstream's own convention, so the exponent is bounded by
    one chunk's gate sum however long the sequence is, and
    :data:`GATE_CUMSUM_ABS_LIMIT` is checked against exactly that quantity's
    largest value (:func:`_require_inter_admissible`).
    """
    with_heads = kg.dim() == 4
    rank = 4 if with_heads else 3
    ranks = (("kg", kg), ("w", w), ("u", u), ("gk", gk), ("q", q), ("aqk", aqk))
    for name, tensor in ranks:
        if tensor.dim() != rank:
            raise ChunkedRecurrenceError(
                f"{name} must be {rank}-D like kg, got shape {tuple(tensor.shape)}"
            )
    lead = (int(kg.shape[0]),) if with_heads else ()
    heads = lead[0] if with_heads else 1
    n_chunks, chunk, kdim = (int(x) for x in kg.shape[-3:])
    vdim = int(u.shape[-1])
    for name, tensor in (("w", w), ("gk", gk), ("q", q)):
        if tuple(tensor.shape) != tuple(kg.shape):
            raise ChunkedRecurrenceError(
                f"{name} {tuple(tensor.shape)} must match kg {tuple(kg.shape)}"
            )
    if tuple(u.shape)[:-1] != lead + (n_chunks, chunk):
        raise ChunkedRecurrenceError(
            f"u {tuple(u.shape)} must agree with kg's leading dimensions "
            f"{lead + (n_chunks, chunk)}"
        )
    if tuple(aqk.shape) != lead + (n_chunks, chunk, chunk):
        raise ChunkedRecurrenceError(
            f"aqk {tuple(aqk.shape)} must be {lead + (n_chunks, chunk, chunk)}"
        )

    # The entering state is refused on the same footing as every other operand,
    # and before the route is chosen, so a caller sees one contract whichever
    # path serves it.
    if state is not None:
        if tuple(state.shape) != lead + (vdim, kdim):
            raise ChunkedRecurrenceError(
                f"state {tuple(state.shape)} must be {lead + (vdim, kdim)} -- the "
                f"orientation final_state is returned in"
            )
        if state.dtype != kg.dtype:
            raise ChunkedRecurrenceError(
                f"state dtype {state.dtype} must be kg's {kg.dtype}; the entering "
                f"state is combined with these operands and is not cast here"
            )

    # Read on an eager call and 0.0 under a graph build, for the reason
    # `kda_intra_chunk` gives above.
    gate_peak = (
        float(gk.float().cumsum(dim=-2).max().item())
        if values_are_readable(gk)
        else 0.0
    )
    if not can_run_inter_chunk(q, n_chunks, chunk, kdim, vdim, gate_peak, heads):
        _count_inter_torch_fallback()
        logger.debug(
            "kda_inter_chunk: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_inter_chunk_torch_oracle(
            kg, w, u, gk, q, aqk, state=state, state_only=state_only
        )

    def chunks_of_all_heads(tensor: Tensor) -> Tensor:
        """The kernel's ``[H * NC, C, X]`` layout of a ``[H, NC, C, X]`` operand."""
        return tensor.reshape((heads * n_chunks,) + tuple(tensor.shape[-2:]))

    if with_heads:
        kg, w, u, gk, q, aqk = (chunks_of_all_heads(t) for t in (kg, w, u, gk, q, aqk))
        if state is not None:
            state = state.reshape(heads * vdim, kdim)
    consts = inter_chunk_constants(
        chunk, kdim, vdim, heads=heads, device=kg.device, dtype=kg.dtype
    )
    # One head with every output passes neither keyword, so that call is the
    # one-head call.
    form = {}
    if heads != 1:
        form["HEADS"] = heads
    if state_only:
        form["STATE_ONLY"] = True
    _count_inter_nki_dispatch()
    call = wrap_nki(kda_inter_chunk_kernel)
    grid = inter_chunk_grid(vdim)
    if grid:
        call = call[grid]
    produced = call(
        kg_hbm=kg,
        w_hbm=w,
        u_hbm=u,
        gk_hbm=gk,
        q_hbm=q,
        aqk_hbm=aqk,
        triu_hbm=consts.triu_ones,
        last_col_hbm=consts.last_col,
        state_init_hbm=consts.state_init if state is None else state.contiguous(),
        **form,
    )
    o = v_new = None
    if state_only:
        final_state = produced
    else:
        o, final_state, v_new = produced
    if with_heads:
        final_state = final_state.reshape(heads, vdim, kdim)
        if not state_only:
            o = o.reshape(heads, n_chunks, chunk, vdim)
            v_new = v_new.reshape(heads, n_chunks, chunk, vdim)
    return InterChunkOutputs(o=o, final_state=final_state, v_new=v_new)


def kda_inter_chunk_torch_oracle(
    kg: Tensor,
    w: Tensor,
    u: Tensor,
    gk: Tensor,
    q: Tensor,
    aqk: Tensor,
    *,
    state: Tensor | None = None,
    state_only: bool = False,
) -> InterChunkOutputs:
    """Stages 4 and 5 in torch: the fallback path, never the shipped one.

    Present so a host with no device or simulator can exercise this module's
    contract. It follows the same chunked formula the kernel does, so it is not an
    independent reference; :func:`kda_sequential_torch_oracle` is. Operands with a
    leading heads axis run head by head; ``state_only`` returns the final state
    alone, as :func:`kda_inter_chunk` does.
    """
    if kg.dim() == 4:
        per_head = [
            kda_inter_chunk_torch_oracle(
                kg[h], w[h], u[h], gk[h], q[h], aqk[h],
                state=None if state is None else state[h], state_only=state_only,
            )
            for h in range(int(kg.shape[0]))
        ]
        return InterChunkOutputs(
            o=None if state_only else torch.stack([out.o for out in per_head]),
            final_state=torch.stack([out.final_state for out in per_head]),
            v_new=None if state_only else torch.stack([out.v_new for out in per_head]),
        )
    kg32, w32, u32 = kg.float(), w.float(), u.float()
    gk32, q32, aqk32 = gk.float(), q.float(), aqk.float()
    n_chunks, chunk, kdim = kg32.shape
    vdim = u32.shape[2]

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)
    gc = gk32.cumsum(dim=1)

    # ``state`` arrives as ``[V, K]`` and is turned into the ``[K, V]`` this scan
    # carries, the same conversion the kernel emits.
    ht = (
        torch.zeros(kdim, vdim, device=kg32.device, dtype=kg32.dtype)
        if state is None
        else state.float().t().contiguous()
    )
    o = torch.empty(n_chunks, chunk, vdim, device=kg32.device, dtype=kg32.dtype)
    v_new = torch.empty_like(o)
    for c in range(n_chunks):
        vn = u32[c] - w32[c] @ ht
        v_new[c] = vn
        o[c] = (qn[c] * torch.exp(gc[c])) @ ht + aqk32[c] @ vn
        ht = ht * torch.exp(gc[c, -1]).unsqueeze(1) + kg32[c].t() @ vn
    final_state = ht.t().contiguous()
    if state_only:
        return InterChunkOutputs(o=None, final_state=final_state, v_new=None)
    return InterChunkOutputs(o=o, final_state=final_state, v_new=v_new)


def kda_sequential_torch_oracle(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> SequentialOutputs:
    """The sequential delta rule, one token at a time, over flat inputs.

    The independent reference for the chunked path: it materialises no ``A``, forms
    no inverse, computes no ``w`` / ``u`` / ``kg`` / ``Aqk``, and never groups
    tokens. Agreement with the kernel is therefore a statement that the chunking
    is associativity-correct.

    Args:
        q, k, gk: ``[T, K]`` fp32, flat over tokens, with no chunk axis.
        v: ``[T, V]`` fp32.
        beta: ``[T]`` fp32.

    Returns:
        :class:`SequentialOutputs` with ``o`` shaped ``[T, V]`` and
        ``final_state`` shaped ``[V, K]``.

    Three details below are load-bearing for that comparison: the state decays per
    key channel rather than by one scalar, which is the KDA-specific step; the
    L2-normalisation epsilon is :data:`L2_NORM_EPS` and sits inside the square
    root; and ``q`` carries ``K ** -0.5``. So is the step order: the state is
    decayed first, and the delta then reads the decayed state.
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32, gk32 = beta.float(), gk.float()
    tokens, kdim = q32.shape
    vdim = v32.shape[1]

    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)

    state = torch.zeros(vdim, kdim, device=q32.device, dtype=q32.dtype)
    o = torch.empty(tokens, vdim, device=q32.device, dtype=q32.dtype)
    for t in range(tokens):
        state = state * torch.exp(gk32[t]).unsqueeze(0)
        delta = (v32[t] - state @ kn[t]) * beta32[t]
        state = state + torch.outer(delta, kn[t])
        o[t] = state @ qn[t]
    return SequentialOutputs(o=o, final_state=state)


def inter_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the inter-chunk kernel this module authors.

    Read off the unwrapped ``.func``, since ``nki.jit``'s wrapper reports its own
    module either way.
    """
    func = getattr(kda_inter_chunk_kernel, "func", None)
    target = func if func is not None else kda_inter_chunk_kernel
    return target.__module__, target.__qualname__
