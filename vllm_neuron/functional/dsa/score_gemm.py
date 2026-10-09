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
import os

import torch
from dataclasses import dataclass
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

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


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(tokens: int, cands: int, heads: int, head_dim: int,
                         programs: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    The dispatch branch is traced under ``fullgraph=True``, so a host call Dynamo
    refuses would break it. A folded helper may take ints, strings and dtypes only:
    Dynamo converts every non-tensor argument into a constant at trace time, and an
    ``@nki.jit`` object is a frozen dataclass it cannot reconstruct. The kernel is
    therefore read as a module global instead of passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_score_gemm_nki)
    logger.info(
        "[dsa-score-gemm] kernel=nki tokens=%d cands=%d heads=%d head_dim=%d programs=%d",
        tokens,
        cands,
        heads,
        head_dim,
        programs,
    )


def _programs(tokens: int) -> int:
    """SPMD programs for a launch: two on an LNC2 runtime once there is a token tile for each.

    The programs split the token tiles, so a call with one token tile runs as one program.
    """
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == str(_LNC2_PROGRAMS) and tokens > TOKEN_TILE:
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
    programs = _programs(tokens)
    _record_nki_dispatch(tokens, cands, heads, head_dim, programs)
    launch = wrap_nki(_score_gemm_nki)
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
