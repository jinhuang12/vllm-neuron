# SPDX-License-Identifier: Apache-2.0
"""The sparse-attention indexer's score GEMM, in NKI.

For every query token ``m`` and every key candidate ``n``::

    logits[m, n] = sum over h of  weights[m, h] * ReLU( dot( q[m, h, :], k[n, :] ) )

``q`` is the per-head indexer query, ``k`` the single key per candidate (the indexer
is multi-query attention, so ``k`` carries no head axis), and ``weights`` the
per-token per-head gate, already carrying every scale the caller folds in. The head
axis is reduced away; the output is one score per (token, candidate) pair, in fp32
because ``nc_matmul`` accumulates there anyway and because bf16 collapses distinct
scores badly enough to destroy the ranking the next stage computes.

The ReLU sits inside the head sum, and that order is the function -- rectifying after
the head reduction computes something else. It also rules out the free PSUM
accumulation a plain contraction would get: a nonlinearity and a per-token scale
stand between each head's matmul and the sum, so every head pays its own matmul,
rectify, scale and one add into an SBUF accumulator.

That head sum is the kernel's cost, and it costs per partition: the vector engine adds one fp32
element per lane per cycle, so it wants all 128 lanes busy. Two kernels keep them busy. With a
full token tile (``tokens >= TOKEN_TILE``) the tokens fill the partitions, and
:func:`_score_gemm_nki` sums with the tokens there. With fewer tokens -- 16 rows per rank on the
row-sharded prefill indexer -- the tokens would leave most lanes idle, so
:func:`_score_gemm_cand_major_nki` puts 128 candidates on the partitions instead and turns the
sums back to rows on the PE at the end. Both add the same fp32 values in the same head order, so
the two produce the same bits.

``nisa.nc_matmul`` contracts the partition axis of both operands, so the head
dimension has to arrive there. The kernel makes that turn itself, with DMA transposes
from HBM straight into SBUF. A DMA moves each 2-byte element as a bit pattern, so the
turn is exact for every value, NaN and Inf included: a non-finite element reaches only
the scores its own query row or key column takes part in, as in the reference. Nothing
scans for finiteness here: neither the rotation nor the normalisation the callers apply
removes a non-finite value, and a per-call scan would cost more than the transport.

Quantisation, masking, top-k and head padding all belong to neighbouring stages. In
particular there is no MX path: ``nc_matmul_mx`` needs NeuronCore-v4 and this target
is v3. Dropping the upstream Hadamard rotation along with the quantisation is exact
rather than merely cheaper, since ``q @ H @ H.T @ k.T == q @ k.T``.
"""

import logging

import torch
from dataclasses import dataclass
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

INDEX_HEAD_DIM = 128
"""The indexer head dimension, and the length of every dot product this kernel contracts.

It is pinned rather than tiled, and the reason is a coincidence worth stating: 128 is also the
partition maximum, so the contraction axis fits in exactly one tile and there is no contraction loop
at all. A head dimension other than 128 would need that loop, so the gate refuses it and the torch
reference serves it instead.
"""

INDEX_N_HEADS = 32
"""``index_n_heads`` as the model configuration declares it, recorded for the reader; not a limit.

The head axis is an ordinary loop bound here, so any positive head count runs. Two upstream
comments disagree with this value (``attention.py:233`` says 64, ``attention.py:384`` says 16),
but both are prose rather than configuration; the value the model loads at run time is 32.
"""

TOKEN_TILE = 128
"""Query tokens per matmul, bounded by the stationary free size, which is 128.

Declared as a literal and asserted against ``nl.tile_size`` by the test rather than read from it at
import. Reading the tile extents at import is not safe on this image: ``nl.tile_size.psum_num_banks``
raises ``RuntimeError: No backend set`` outside an activated backend, so a module that read its
neighbours would import fine here and break somewhere else.
"""

CAND_TILE = 512
"""Key candidates per matmul, bounded by the moving free size, which is 512 on NeuronCore-v2 and v3."""

CONTRACTION_TILE = 128
"""The contraction extent, bounded by the partition maximum. Equal to ``INDEX_HEAD_DIM`` by design."""

_GROUP_TILES = 4
"""Candidate tiles in one score group: the columns one head's matmuls fill before one rectify.

A candidate tile of fp32 scores is one PSUM bank, so a group of four takes half of the eight banks
of NeuronCore-v2 and v3 and the PE fills the next head's group in the other half while this one
drains. The scalar engine's rectify and the vector engine's weighted add each run once per group,
over ``_GROUP_TILES * CAND_TILE`` columns, which spreads their fixed per-instruction cost four
times wider than one instruction per matmul would.
"""

_BUFFERS = 2
"""Copies kept of each streamed SBUF tile: a token tile's query and weights, and a group's sum.

With two, the next token tile's query and weights land while the current tile computes, and a
finished group's sum drains to HBM while the next group accumulates into the other copy.
"""

_LNC2_PROGRAMS = 2
"""Programs on an LNC2 launch: one per physical core of the logical core."""

CAND_BLOCK = 128
"""Candidates per matmul of the candidate-major kernel, bounded by the stationary free size (128).

The block's candidates become the PSUM partitions, so one block fills all 128 vector lanes.
"""

_PSUM_BANK_COLUMNS = 512
"""fp32 columns of one PSUM bank (2 KiB of each partition). A matmul writes inside one bank."""

_SCORE_GROUP_BANKS = 2
"""PSUM banks one rectify of the candidate-major kernel drains: two, so two groups in flight
take four of the eight banks and the row turns at the end of a sweep have the rest."""

_QUERY_TURN_ALIGN = 16
"""Token count granule of the candidate-major kernel: 16 bf16 columns are 32 bytes.

The kernel turns each head's ``[tokens, head_dim]`` query rows into SBUF columns ``h * tokens``
onward with a DMA transpose, whose destination must start 32-byte aligned (the compiler's
verifier refuses anything else), so ``tokens`` must be a multiple of 16. Other counts take the
row-major kernel.
"""

_SWEEP_ELEMENTS = 16384
"""fp32 elements per partition of one sweep's rectified head scores: 64 KiB of SBUF.

A sweep is the run of candidate blocks one head-sum instruction spans. The head sum issues one
vector instruction per head per sweep, over ``blocks * tokens`` elements, so a wider sweep spreads
the instruction's fixed cost thinner: at 16 tokens and 32 heads a sweep holds 32 blocks (4096
candidates) and each add spans 512 elements. A call whose ``heads * tokens`` exceeds this takes the
row-major kernel.
"""

_SUPPORTED_Q_DTYPES = (torch.bfloat16,)
"""Query and key dtypes that take the NKI route.

bf16 is the indexer path's dtype and it is also what the upstream reference computes in: it
casts both fp8 operands up to bf16 before the matmul (``rocm_aiter_mla_sparse.py:714-715``) and
accumulates in fp32. ``nc_matmul`` does the same natively, so the two paths agree numerically
rather than approximately.
"""

_SUPPORTED_W_DTYPES = (torch.float32,)
"""Weight dtypes that take the NKI route. Upstream declares ``weights`` fp32
(``rocm_aiter_mla_sparse.py:703``) and the per-token scale is applied in fp32 here, so a bf16
weight would quietly lose precision in the fold."""


class ScoreGemmError(ValueError):
    """A malformed call: wrong rank, a head or feature mismatch, or an unsupported head dimension."""


@dataclass
class _ScoreGemmDispatchCounters:
    """Per-process record of how this module's entry point was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _ScoreGemmDispatchCounters()


def reset_score_gemm_dispatch_counters() -> None:
    """Zero this module's dispatch counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def score_gemm_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def score_gemm_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
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
# Device
# ---------------------------------------------------------------------------------------------


def _transpose_tile(dst_sb, src_hbm):
    """Write ``src_hbm``, a ``[rows, cols]`` HBM slab, into ``dst_sb`` as ``[cols, rows]``.

    The turn is a DMA transpose, HBM to SBUF, which takes ``cols <= 128`` and a 2-byte dtype. No
    compute engine spends cycles on it, and the source and destination are in different
    memories, so they cannot alias.
    """
    nisa.dma_transpose(dst=dst_sb, src=src_hbm)


def _keys_transposed(k_hbm, head_dim, cands):
    """``k`` as stored, ``[cands, head_dim]`` -> ``[head_dim, cands]`` SBUF.

    One transpose per ``CAND_TILE`` candidates, so the first matmuls wait only for their own
    columns rather than for all of ``k``.
    """
    kt = nl.ndarray((head_dim, cands), dtype=k_hbm.dtype, buffer=nl.sbuf)
    for n0 in range(0, cands, CAND_TILE):
        nw = min(CAND_TILE, cands - n0)
        _transpose_tile(kt[0:head_dim, n0:n0 + nw], k_hbm[n0:n0 + nw, 0:head_dim])
    return kt


def _load_token_tile(q_hbm, w_hbm, qt, w_sb, m0, mw, slot):
    """Token rows ``m0 .. m0 + mw`` into copy ``slot`` of the streamed query and weight tiles.

    ``qt`` is ``[head_dim, _BUFFERS, heads, TOKEN_TILE]``: each head's ``[mw, head_dim]`` rows are
    turned to ``[head_dim, mw]``, the stationary layout. ``w_sb`` is
    ``[TOKEN_TILE, _BUFFERS, heads]``, read as stored: column ``h`` of a copy is head ``h``'s
    per-token scale, the per-partition operand the vector engine's scale takes.
    """
    heads = q_hbm.shape[1]
    head_dim = q_hbm.shape[2]
    for h in range(heads):
        _transpose_tile(qt[0:head_dim, slot, h, 0:mw], q_hbm[m0:m0 + mw, h, 0:head_dim])
    nisa.dma_copy(dst=w_sb[0:mw, slot, 0:heads], src=w_hbm[m0:m0 + mw, 0:heads])


@nki.jit
def _score_gemm_nki(q_hbm, k_hbm, w_hbm):
    """Rectified per-head scores, weighted and summed over heads.

    Args:
        q_hbm: ``[tokens, heads, head_dim]`` bf16 -- the query as stored; the kernel puts the head
            dimension on the partition axis itself.
        k_hbm: ``[cands, head_dim]`` bf16 -- the key as stored, with no head axis (MQA).
        w_hbm: ``[tokens, heads]`` fp32 -- the per-token per-head gate, fully pre-scaled.

    Returns:
        ``[tokens, cands]`` fp32.

    Launch: plain, or on an SPMD grid of two (LNC2, one program per physical core). The
    programs take the ``TOKEN_TILE`` row tiles in turn, program ``p`` the tiles ``p, p + 2, ...``;
    each turns all of ``k`` for itself, and each writes only its own output rows.

    ``k`` is turned once per program into ``[head_dim, cands]`` (at ``cands`` 16384, 32 KiB of
    each SBUF partition). Each token tile's query rows are turned once and serve every candidate;
    the next tile's rows are loaded into the other of ``_BUFFERS`` copies while this one computes.
    A ragged tile needs no padding: every operation takes any extent up to its tile bound.

    Per (token tile, group of ``_GROUP_TILES`` candidate tiles, head) the engines split the work
    three ways, which is what lets them overlap across heads: the PE writes this head's raw
    scores to PSUM, one matmul per candidate tile, the scalar engine rectifies the whole group on
    the way out of PSUM, and the vector engine folds the weight and the add into the group's SBUF
    sum in one instruction, ``acc = rect * w[:, h] + acc``. The first head initialises ``acc``
    with ``rect * w[:, 0] + 0.0``; the ``+ 0.0`` is the ``0 + x`` of a zeroed sum, which turns a
    ``-0.0`` product into ``+0.0`` exactly as the add would. Each step rounds once in fp32, in the
    reference order (dot, rectify, weight, then the head sum in head order), so the bits are those
    of the four-instruction form. A finished sum leaves on the Sync engine's DMA queue, which no
    compute stage waits behind.

    The matmul tiles are loaded without a widening cast: ``nc_matmul`` admits bf16
    operands and accumulates in fp32 regardless, so casting up first would buy no
    precision and would cost four times the cycles by this ISA's own cost model.
    """
    tokens = q_hbm.shape[0]
    heads = q_hbm.shape[1]
    head_dim = q_hbm.shape[2]
    cands = k_hbm.shape[0]
    programs = nl.num_programs(axes=0)
    program = nl.program_id(0)
    group = _GROUP_TILES * CAND_TILE
    groups = (cands + group - 1) // group
    # This program's token tiles start at ``first`` and step by ``stride``.
    first = program * TOKEN_TILE
    stride = programs * TOKEN_TILE
    tiles = (tokens - first + stride - 1) // stride

    out = nl.ndarray((tokens, cands), dtype=nl.float32, buffer=nl.shared_hbm)
    kt = _keys_transposed(k_hbm, head_dim, cands)
    qt = nl.ndarray((head_dim, _BUFFERS, heads, TOKEN_TILE), dtype=q_hbm.dtype, buffer=nl.sbuf)
    w_sb = nl.ndarray((TOKEN_TILE, _BUFFERS, heads), dtype=nl.float32, buffer=nl.sbuf)
    acc = nl.ndarray((TOKEN_TILE, _BUFFERS, group), dtype=nl.float32, buffer=nl.sbuf)
    if tiles > 0:
        _load_token_tile(q_hbm, w_hbm, qt, w_sb, first, min(TOKEN_TILE, tokens - first), 0)

    for i in range(tiles):
        m0 = first + i * stride
        mw = min(TOKEN_TILE, tokens - m0)
        if i + 1 < tiles:
            _load_token_tile(q_hbm, w_hbm, qt, w_sb, m0 + stride,
                             min(TOKEN_TILE, tokens - m0 - stride), (i + 1) % _BUFFERS)
        q_tile = qt[0:head_dim, i % _BUFFERS, 0:heads, 0:mw]
        w_tile = w_sb[0:mw, i % _BUFFERS, 0:heads]

        for g in range(groups):
            n0 = g * group
            gw = min(group, cands - n0)
            # The copies alternate over every group of the program, across token tiles too.
            total = acc[0:mw, (i * groups + g) % _BUFFERS, 0:gw]
            for h in range(heads):
                # dst = stationary.T @ moving = [mw, jw] per candidate tile: head_dim contracts
                # on the partitions, and each tile lands in its own PSUM bank of the group.
                ps = nl.ndarray((mw, gw), dtype=nl.float32, buffer=nl.psum)
                for j0 in range(0, gw, CAND_TILE):
                    jw = min(CAND_TILE, gw - j0)
                    nisa.nc_matmul(dst=ps[0:mw, j0:j0 + jw], stationary=q_tile[0:head_dim, h, 0:mw],
                                   moving=kt[0:head_dim, n0 + j0:n0 + j0 + jw], accumulate=False)
                # Rectify per head, before the weight and before the sum.
                rect = nl.ndarray((mw, gw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=ps)
                if h == 0:
                    nisa.tensor_scalar(dst=total, data=rect, op0=nl.multiply,
                                       operand0=w_tile[0:mw, 0:1], op1=nl.add, operand1=0.0,
                                       engine=nisa.vector_engine)
                else:
                    nisa.scalar_tensor_tensor(dst=total, data=rect, op0=nl.multiply,
                                              operand0=w_tile[0:mw, h:h + 1], op1=nl.add,
                                              operand1=total)
            nisa.dma_copy(dst=out[m0:m0 + mw, n0:n0 + gw], src=total,
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    return out


def _sweep_plan(cands, cols, programs):
    """The candidate-major kernel's sweeps, as ``(first candidate, blocks, block width)``.

    Whole ``CAND_BLOCK`` blocks go in sweeps of up to ``_SWEEP_ELEMENTS // cols`` blocks, fewer when
    that would leave a program without a sweep; a ragged last block (``cands % CAND_BLOCK``
    candidates) is a sweep of its own, so every sweep has one partition extent.
    """
    full = cands // CAND_BLOCK
    tail = cands - full * CAND_BLOCK
    per_program = (full + programs - 1) // programs
    blocks = max(1, min(_SWEEP_ELEMENTS // cols, per_program))
    plan = []
    for b0 in range(0, full, blocks):
        plan.append((b0 * CAND_BLOCK, min(blocks, full - b0), CAND_BLOCK))
    if tail > 0:
        plan.append((full * CAND_BLOCK, 1, tail))
    return plan


def _psum_slot(columns):
    """PSUM columns reserved per matmul: the power of two at or above ``columns`` (<= one bank).

    A power of two divides the bank, so slot ``i`` of a bank-aligned tile never straddles two banks.
    """
    slot = 1
    while slot < columns:
        slot *= 2
    return slot


def _score_sweep_rectify(k_hbm, qt, n0, blocks, width, tokens, heads, head_dim):
    """Rectified scores of candidates ``n0 .. n0 + blocks * width``: ``[width, blocks, (head,
    token)]`` fp32 in SBUF. The first of a sweep's three stages (:data:`_SUM_LAG`).

    ``qt`` is the query turned to ``[head_dim, (head, token)]``, token fastest. Per block the PE
    writes ``[width, (head, token)]``: candidate on the partition, so the head sum
    (:func:`_score_sweep_sum`) runs on ``width`` lanes. One rectify per ``_SCORE_GROUP_BANKS``
    banks.
    """
    cols = heads * tokens
    span = blocks * width
    head_step = max(1, min(heads, CAND_TILE // tokens))
    slot = _psum_slot(head_step * tokens)
    pieces = (heads + head_step - 1) // head_step
    group = max(1, _SCORE_GROUP_BANKS * _PSUM_BANK_COLUMNS // (pieces * slot))
    group_cols = max(_PSUM_BANK_COLUMNS, group * pieces * slot)

    kt = nl.ndarray((head_dim, span), dtype=k_hbm.dtype, buffer=nl.sbuf)
    for c0 in range(0, span, CAND_TILE):
        cw = min(CAND_TILE, span - c0)
        _transpose_tile(kt[0:head_dim, c0:c0 + cw], k_hbm[n0 + c0:n0 + c0 + cw, 0:head_dim])

    rect = nl.ndarray((width, blocks, cols), dtype=nl.float32, buffer=nl.sbuf)
    for b0 in range(0, blocks, group):
        bn = min(group, blocks - b0)
        ps = nl.ndarray((width, group_cols), dtype=nl.float32, buffer=nl.psum)
        for u in range(bn):
            stationary = kt[0:head_dim, (b0 + u) * width:(b0 + u + 1) * width]
            for piece in range(pieces):
                h0 = piece * head_step
                hn = min(head_step, heads - h0)
                at = (u * pieces + piece) * slot
                # dst = stationary.T @ moving = [width candidates, hn * tokens]: the same dot
                # products as the row-major kernel's, with the roles of q and k swapped.
                nisa.nc_matmul(dst=ps[0:width, at:at + hn * tokens], stationary=stationary,
                               moving=qt[0:head_dim, h0 * tokens:(h0 + hn) * tokens],
                               accumulate=False)
        # Rectify per head, before the weight and before the sum.
        for piece in range(pieces):
            h0 = piece * head_step
            hn = min(head_step, heads - h0)
            data = ps.ap(pattern=[[group_cols, width], [pieces * slot, bn], [1, hn * tokens]],
                         offset=piece * slot)
            nisa.activation(dst=rect[0:width, b0:b0 + bn, h0 * tokens:(h0 + hn) * tokens],
                            op=nl.relu, data=data)
    return rect


def _score_sweep_sum(rect, w_cols, blocks, width, tokens, heads):
    """The weighted head sum of one sweep's rectified scores ``rect``: ``[width, blocks, tokens]``.

    ``w_cols`` is the gate on every partition as ``[128, (head, token)]``. One multiply by the gate
    for the whole sweep, then one add per head in head order.
    """
    cols = heads * tokens
    # The weight, every (block, head, token) of the sweep in one instruction: rect * w.
    nisa.tensor_tensor(dst=rect, data1=rect,
                       data2=w_cols.ap(pattern=[[cols, width], [0, blocks], [1, cols]], offset=0),
                       op=nl.multiply, engine=nisa.vector_engine)
    # The head sum in head order. ``+ 0.0`` first, as the row-major kernel's first head does.
    acc = nl.ndarray((width, blocks, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=acc, data=rect[0:width, 0:blocks, 0:tokens], op0=nl.add,
                       operand0=0.0, engine=nisa.vector_engine)
    for h in range(1, heads):
        nisa.tensor_tensor(dst=acc, data1=rect[0:width, 0:blocks, h * tokens:(h + 1) * tokens],
                           data2=acc, op=nl.add, engine=nisa.vector_engine)
    return acc


def _score_sweep_store(out, acc, n0, blocks, width, tokens):
    """One sweep's sums ``acc`` (:func:`_score_sweep_sum`) back to rows, into ``out`` at candidate
    ``n0``: one PE turn per block, up to a bank of them per drain and per store."""
    per_bank = max(1, _PSUM_BANK_COLUMNS // width)
    for b0 in range(0, blocks, per_bank):
        bn = min(per_bank, blocks - b0)
        o_ps = nl.ndarray((tokens, per_bank * width), dtype=nl.float32, buffer=nl.psum)
        for u in range(bn):
            nisa.nc_transpose(dst=o_ps[0:tokens, u * width:(u + 1) * width],
                              data=acc[0:width, b0 + u, 0:tokens])
        o_sb = nl.ndarray((tokens, per_bank * width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb[0:tokens, 0:bn * width], src=o_ps[0:tokens, 0:bn * width],
                         engine=nisa.scalar_engine)
        c0 = n0 + b0 * width
        nisa.dma_copy(dst=out[0:tokens, c0:c0 + bn * width], src=o_sb[0:tokens, 0:bn * width],
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


_SUM_LAG = 1
"""Sweeps a program's head sum is issued after its rectify (see :func:`_score_gemm_cand_major_nki`).

The tile allocator places SBUF buffers by their live ranges in program order, not by when the
engines run. Issued one sweep late, a sweep's rectified scores are still live when the next
sweep's are written, so the two get separate SBUF and the scalar engine rectifies one sweep while
the vector engine sums the other. Issued in sweep order, the allocator gave both the same
address, and every rectify waited for the previous sweep's last add (seen in a device
profile): the sweep ran at about 1.5 times the vector engine's busy time. Two sweeps of
``_SWEEP_ELEMENTS`` fp32 take 128 KiB of a partition's SBUF."""

_STORE_LAG = 3
"""Sweeps a program's store is issued after its rectify.

The PE runs its instructions in program order, and a store's PE turns wait for the sweep's last
add. At this lag that sweep's head sum was issued two sweeps earlier, so the turns do not hold
the next rectify's matmuls behind a head sum the vector engine is still running."""


@nki.jit
def _score_gemm_cand_major_nki(q_hbm, k_hbm, w_hbm):
    """:func:`_score_gemm_nki` for fewer than ``TOKEN_TILE`` tokens, with the head sum on 128 lanes.

    Args and returns as :func:`_score_gemm_nki`. Requires (:func:`_cand_major`) ``tokens <
    TOKEN_TILE``, ``tokens`` a multiple of ``_QUERY_TURN_ALIGN`` and ``heads * tokens <=
    _SWEEP_ELEMENTS``.

    The row-major kernel puts the tokens on the partitions, so with 16 tokens its head sum runs on
    16 of the vector engine's 128 lanes. Here the matmul's roles are swapped: ``CAND_BLOCK`` keys
    are stationary and the whole query, ``[head_dim, (head, token)]``, moves, so the PSUM tile is
    ``[128 candidates, (head, token)]`` and every later stage runs on 128 lanes.

    The arithmetic is the row-major kernel's, value for value. Each dot product contracts the same
    128 bf16 pairs on the PE, whichever operand is stationary. The rectify is the same scalar-engine
    ReLU. The weight is one fp32 multiply, ``rect * w`` (``w`` now varies along the free axis, so it
    is a tensor operand rather than a per-partition scalar, and the multiply is its own
    instruction), and the head sum is the same chain of fp32 adds in head order, starting from
    ``prod_0 + 0.0``. Each step rounds once in fp32 in the same order, so the sums are the same
    bits; the PE turns them back to ``[tokens, candidates]`` (bit-accurate for fp32 from
    NeuronCore-v3) and they leave as whole rows.

    Launch: plain, or on an SPMD grid of two (LNC2). The programs take the sweeps of
    :func:`_sweep_plan` in turn, program ``p`` the sweeps ``p, p + 2, ...``; each loads the query
    and the gate for itself and writes only its own output columns.
    """
    tokens = q_hbm.shape[0]
    heads = q_hbm.shape[1]
    head_dim = q_hbm.shape[2]
    cands = k_hbm.shape[0]
    programs = nl.num_programs(axes=0)
    program = nl.program_id(0)
    cols = heads * tokens
    partitions = nl.tile_size.pmax

    out = nl.ndarray((tokens, cands), dtype=nl.float32, buffer=nl.shared_hbm)
    # The query, [head_dim, (head, token)]: one turn per head.
    qt = nl.ndarray((head_dim, cols), dtype=q_hbm.dtype, buffer=nl.sbuf)
    for h in range(heads):
        _transpose_tile(qt[0:head_dim, h * tokens:(h + 1) * tokens], q_hbm[0:tokens, h, 0:head_dim])
    # The gate as stored, [(token, head)], on every partition, then reordered to [(head, token)].
    w_rows = nl.ndarray((partitions, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_rows, src=w_hbm.ap(pattern=[[0, partitions], [1, cols]], offset=0))
    w_cols = nl.ndarray((partitions, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_cols,
                     src=w_rows.ap(pattern=[[cols, partitions], [1, heads], [heads, tokens]],
                                   offset=0),
                     engine=nisa.vector_engine)

    # Three stages per sweep, issued apart: the rectify of sweep i, then the head sum of sweep
    # i - _SUM_LAG, then the store of sweep i - _STORE_LAG (see the two constants).
    plan = _sweep_plan(cands, cols, programs)
    rectified = []
    summed = []
    for s in range(program, len(plan), programs):
        n0, blocks, width = plan[s]
        rect = _score_sweep_rectify(k_hbm, qt, n0, blocks, width, tokens, heads, head_dim)
        rectified.append((rect, n0, blocks, width))
        if len(rectified) > _SUM_LAG:
            ready = rectified.pop(0)
            acc = _score_sweep_sum(ready[0], w_cols, ready[2], ready[3], tokens, heads)
            summed.append((acc, ready[1], ready[2], ready[3]))
        if len(summed) > _STORE_LAG - _SUM_LAG:
            done = summed.pop(0)
            _score_sweep_store(out, done[0], done[1], done[2], done[3], tokens)
    for ready in rectified:
        acc = _score_sweep_sum(ready[0], w_cols, ready[2], ready[3], tokens, heads)
        summed.append((acc, ready[1], ready[2], ready[3]))
    for done in summed:
        _score_sweep_store(out, done[0], done[1], done[2], done[3], tokens)

    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(tokens: int, cands: int, heads: int, head_dim: int,
                         programs: int, cand_major: bool) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    The dispatch branch is traced under ``fullgraph=True``, so a host call Dynamo
    refuses would break it. A folded helper may take ints, strings and dtypes only:
    Dynamo converts every non-tensor argument into a constant at trace time, and an
    ``@nki.jit`` object is a frozen dataclass it cannot reconstruct. The kernel is
    therefore read as a module global instead of passed in, chosen by ``cand_major``.
    """
    kernel = _score_gemm_cand_major_nki if cand_major else _score_gemm_nki
    _COUNTERS.last_kernel = _kernel_identity_of(kernel)
    logger.info(
        "[dsa-score-gemm] kernel=nki layout=%s tokens=%d cands=%d heads=%d head_dim=%d "
        "programs=%d",
        "cand_major" if cand_major else "row_major",
        tokens,
        cands,
        heads,
        head_dim,
        programs,
    )


def _cand_major(tokens: int, heads: int) -> bool:
    """Whether the call takes :func:`_score_gemm_cand_major_nki`: fewer tokens than a token tile,
    a whole number of ``_QUERY_TURN_ALIGN`` granules, and a ``(head, token)`` row that fits one
    sweep."""
    return (tokens < TOKEN_TILE and tokens % _QUERY_TURN_ALIGN == 0
            and heads * tokens <= _SWEEP_ELEMENTS)


def _programs(tokens: int, cand_major: bool) -> int:
    """SPMD programs for a launch: two on an LNC2 runtime when the work splits in two.

    The row-major kernel's programs split the token tiles, so a call with one token tile runs as
    one program; the candidate-major kernel's split the candidate sweeps, so it always takes both.
    """
    if lnc_pair() and (cand_major or tokens > TOKEN_TILE):
        return _LNC2_PROGRAMS
    return 1


def _validate(q: Tensor, k: Tensor, weights: Tensor) -> tuple[int, int, int, int]:
    """Host-side shape validation. Returns ``(tokens, heads, head_dim, cands)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor value, so nothing here forces
    a device-to-host synchronisation or a data-dependent trace.
    """
    if q.ndim != 3:
        raise ScoreGemmError(
            f"q must be 3-D [tokens, heads, head_dim]; got shape {tuple(q.shape)}"
        )
    if k.ndim != 2:
        raise ScoreGemmError(
            f"k must be 2-D [cands, head_dim] -- the indexer is MQA, so k carries no head axis; "
            f"got shape {tuple(k.shape)}"
        )
    tokens, heads, head_dim = (int(d) for d in q.shape)
    cands = int(k.shape[0])
    if int(k.shape[1]) != head_dim:
        raise ScoreGemmError(
            f"k's feature width must match q's head_dim {head_dim}; got {int(k.shape[1])}"
        )
    if weights.ndim != 2 or tuple(weights.shape) != (tokens, heads):
        raise ScoreGemmError(
            f"weights must be [tokens, heads] = {(tokens, heads)}; got {tuple(weights.shape)}"
        )
    if tokens <= 0 or cands <= 0 or heads <= 0:
        raise ScoreGemmError(
            f"tokens, cands and heads must all be positive; got {(tokens, cands, heads)}"
        )
    if head_dim != INDEX_HEAD_DIM:
        raise ScoreGemmError(
            f"the score GEMM contracts exactly {INDEX_HEAD_DIM} features in one tile; got head_dim "
            f"{head_dim}"
        )
    return tokens, heads, head_dim, cands


def can_run_dsa_score_gemm(q: Tensor, k: Tensor, weights: Tensor) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch path."""
    if not can_run_kernel():
        return False
    if q.dtype not in _SUPPORTED_Q_DTYPES or k.dtype not in _SUPPORTED_Q_DTYPES:
        return False
    if weights.dtype not in _SUPPORTED_W_DTYPES:
        return False
    if q.ndim != 3 or k.ndim != 2 or weights.ndim != 2:
        return False
    if int(q.shape[2]) != INDEX_HEAD_DIM or int(k.shape[1]) != INDEX_HEAD_DIM:
        return False
    return tuple(weights.shape) == (int(q.shape[0]), int(q.shape[1]))


def dsa_score_gemm(q: Tensor, k: Tensor, weights: Tensor) -> Tensor:
    """Rectified per-head query-key scores, weighted and summed over heads.

    Args:
        q: ``[tokens, heads, head_dim]`` -- the indexer query, one vector per head.
            ``bfloat16`` takes the NKI route; any other dtype is served by the torch path.
        k: ``[cands, head_dim]`` -- one key per candidate, shared across heads (MQA). The candidate
            axis is pool-granular on this checkpoint rather than token-granular, which is why the
            next stage selects pools.
        weights: ``[tokens, heads]`` fp32 -- the per-token per-head gate, already carrying every
            scale the caller folds in, including the query scale, ``head_dim ** -0.5`` and
            ``n_head ** -0.5``. This seam applies no scale of its own.

    Returns:
        ``[tokens, cands]`` fp32.

    Raises:
        ScoreGemmError: for a malformed call -- a non-3D ``q``, a ``k`` whose feature width does not
            match, a ``weights`` of the wrong shape, or a head dimension that is not 128.
    """
    tokens, heads, head_dim, cands = _validate(q, k, weights)

    if not can_run_dsa_score_gemm(q, k, weights):
        _count_torch_fallback()
        return _dsa_score_gemm_torch(q, k, weights)

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter store
    # inside the trace becomes a value guard that fails on the first call after warmup.
    cand_major = _cand_major(tokens, heads)
    programs = _programs(tokens, cand_major)
    _record_nki_dispatch(tokens, cands, heads, head_dim, programs, cand_major)
    launch = wrap_nki(_score_gemm_cand_major_nki if cand_major else _score_gemm_nki)
    if programs > 1:
        launch = launch[programs]
    # Same layout in, same layout out: a contiguous caller gets its own storage back, a strided
    # one gets a plain copy. Neither is a transposing relayout; the kernel turns the operands.
    return launch(q.contiguous(), k.contiguous(), weights.contiguous())


# ---------------------------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------------------------


def _dsa_score_gemm_torch(q: Tensor, k: Tensor, weights: Tensor) -> Tensor:
    """The CPU reference the NKI route is measured against, and the path a refused call takes.

    Keeps the reference order -- dot, rectify, weight, sum -- and runs everything in
    fp32, which is what ``nc_matmul`` does with bf16 operands and the only basis on
    which the two can be compared. There is no per-key dequant scale because a bf16
    route has nothing quantised to rescale.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head.clamp(min=0.0) * weights.float().unsqueeze(-1)).sum(dim=1)
