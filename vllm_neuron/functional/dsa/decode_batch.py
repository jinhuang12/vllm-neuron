# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's decode step for ``B`` requests in one launch per stage.

The one-request decode chain (``Glm5NextDSAIndexer.forward``) launches the ring step,
the candidate read, the score GEMM and the causal bound once per request, because each
request reads its own pooled keys. This module serves a whole decode batch:

* :func:`dsa_decode_ring_step` -- every request's ring advances by its own token, and the
  pool a completing token closes is compressed. One request per partition, the
  one-request kernel's instruction sequence on each.
* :func:`dsa_decode_scores` -- every request scores the candidate pools of its own slot
  of the pooled-key bank against its own query, and the causal bound is applied with
  its own length. The result feeds the selector, which already takes a row per request.

Both read the runner's side-cache banks directly: ``[slots, ...]`` tensors addressed by
a ``[B]`` slot vector, so no request's state is copied or stacked first and no request
can read another's rows. The writes (the advanced ring, the completed pool) go back to
the banks at the call site, and nothing here depends on whether they landed: the score
stage takes this step's completed pool from ``pooled`` and stands it in at its own
candidate column, as the decode attention stands in its own latent row.

Layout of the score stage. Candidates ride the partitions (128 per tile) and the 32
indexer heads ride the free axis, so one matmul per candidate tile computes every head
(``stationary`` = 128 keys transposed onto the head dimension, ``moving`` = the
request's query transposed). The rectify and the head weight are one op per PSUM bank
and the head sum is one free-axis reduce. The one-request kernel puts the single token
on the partitions instead and pays a matmul, a rectify, a scale and an add per head per
512 candidates, which is what made it slow at one token.

The candidate axis is walked in chunks of at most ``CHUNK_TILES`` tiles (16,384
candidates, 65,536 tokens of context at pool 4): a chunk's tile count rides the
partitions of the output transpose, and a chunk's keys sit in SBUF twice (as read and
turned). Each chunk carries this step's pool as one more tile, so a chunk is the
whole-axis kernel on a column range, and an axis of one chunk runs exactly the
instructions it ran before chunking.

Both physical cores of an LNC2 core split the requests of the score stage (``[2]``
grid at ``B >= 2``). The chunks of one request run on its program, one after another,
so ``B = 1`` runs on one core. The ring step is a few vector ops on ``B`` partitions and
runs on one program.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.decode_tail_update import TAIL_HALVES, _compress_pool_torch
from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    INDEX_HEAD_DIM,
    _fwht128_inplace,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Partition extent: requests per ring-step tile, candidates per score tile.
PARTITIONS = 128
#: bf16 ``[128, 128]`` key transposes gathered into one PSUM tile before the copy out.
TRANSPOSE_GROUP = 4
#: fp32 columns of one PSUM bank; the head scores of ``PSUM_FP32 // heads`` candidate
#: tiles share one bank, so the rectify, the weight and the head sum run once per bank.
PSUM_FP32 = 512
#: Candidate tiles per chunk of the score stage: the output transpose puts a chunk's
#: tile count on the partitions. Wider candidate axes are walked chunk by chunk.
CHUNK_TILES = PARTITIONS
#: Candidates per chunk (16,384): the whole axis the kernel served before chunking.
CHUNK_CANDIDATES = CHUNK_TILES * PARTITIONS
#: fp32 columns of a [P, 1] scratch tile: one whole 32-byte line, as ``mla_decode``
#: declares its scalars.
_LINE = 8

#: This file's content digest, handed to both kernels as a trace-time int: the compiled
#: kernel cache keys on a kernel's own source and its arguments, and both kernels call
#: helpers whose edits it would otherwise not see.
SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)


class DecodeBatchError(ValueError):
    """A malformed batched decode call: a shape, dtype or geometry this module refuses."""


@dataclass
class _DecodeBatchCounters:
    nki_dispatch: int = 0
    torch_fallback: int = 0
    ring_dispatch: int = 0
    scores_dispatch: int = 0
    two_program_dispatch: int = 0


_COUNTERS = _DecodeBatchCounters()


def reset_decode_batch_dispatch_counters() -> None:
    """Zero every count this module keeps."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.ring_dispatch = 0
    _COUNTERS.scores_dispatch = 0
    _COUNTERS.two_program_dispatch = 0


def decode_batch_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` over both entry points, the family form."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def decode_batch_route_counts() -> tuple[int, int, int]:
    """``(ring_dispatch, scores_dispatch, two_program_dispatch)`` since the last reset."""
    return (_COUNTERS.ring_dispatch, _COUNTERS.scores_dispatch,
            _COUNTERS.two_program_dispatch)


@torch._dynamo.assume_constant_result
def _count_dispatch(entry: str, programs: int, batch: int) -> None:
    """Count one kernel dispatch, off the traced graph."""
    _COUNTERS.nki_dispatch += 1
    if entry == "ring":
        _COUNTERS.ring_dispatch += 1
    else:
        _COUNTERS.scores_dispatch += 1
    if programs == 2:
        _COUNTERS.two_program_dispatch += 1
    logger.info("[dsa-decode-batch] kernel=nki entry=%s batch=%d programs=%d",
                entry, batch, programs)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


# ---------------------------------------------------------------------------------------
# Device helpers. Positional arguments only: the NKI front end drops keyword defaults.
# ---------------------------------------------------------------------------------------


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    """A ``[parts, 1]`` view of a tile one 32-byte line wide."""
    return nl.ndarray((parts, _LINE), dtype=dtype, buffer=nl.sbuf)[:, 0:1]


def _log2(value):
    shift = 0
    while (1 << shift) < value:
        shift += 1
    return shift


# ---------------------------------------------------------------------------------------
# Ring step
# ---------------------------------------------------------------------------------------


@nki.jit
def dsa_decode_ring_step_kernel(tail_hbm, slots_hbm, key_hbm, score_hbm, ape_hbm, pos_hbm,
                                pool_size, source_digest):
    """Advance ``B`` rings by one token each and compress the pool each would close.

    Args:
        tail_hbm: ``[slots, 2 * pool_size * head_dim]`` the ring bank, one flattened ring
            per slot: keys in rows ``0 .. pool_size - 1``, gate scores after them.
        slots_hbm: ``[B, 1]`` int32, each request's slot.
        key_hbm / score_hbm: ``[B, head_dim]`` this step's indexer key and gate score.
        ape_hbm: ``[pool_size, head_dim]`` fp32, the per-slot additive bias.
        pos_hbm: ``[B, 1]`` int32, each request's absolute position.
        pool_size: tokens per pool, a power of two.
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``(pooled, rings)``: ``[B, head_dim]`` the compressed pool that ring rows
        ``0 .. pool_size - 2`` and this token make, which is the completed pool exactly
        when ``position % pool_size == pool_size - 1`` (and meaningless otherwise; the
        caller writes it to its slot's trash row then), and ``[B, 2 * pool_size *
        head_dim]`` the advanced rings, this token stashed at row ``position %
        pool_size`` of each half.

    The compression is the one-request kernel's instruction sequence at its last slot,
    one request per partition: the same loads, the same order of adds, the same two
    bf16 round trips and the same butterfly, so the two agree bit for bit.
    """
    batch = key_hbm.shape[0]
    head_dim = key_hbm.shape[1]
    width = tail_hbm.shape[1]
    last = pool_size - 1
    pooled_hbm = nl.ndarray((batch, head_dim), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    rings_hbm = nl.ndarray((batch, width), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    for r0 in range(0, batch, PARTITIONS):
        h = min(PARTITIONS, batch - r0)
        ring = _sb((h, width), tail_hbm.dtype)
        if tail_hbm.shape[0] == 1:
            # One slot: the only valid slot is 0, so the read is static. (Tracing in CPU
            # simulation fills int operands with ones, and slot 1 is past this bank.)
            nisa.dma_copy(dst=ring, src=tail_hbm.ap(pattern=[[0, h], [1, width]], offset=0))
        else:
            slot_t = _col(h, nl.int32)
            nisa.dma_copy(dst=slot_t, src=slots_hbm.ap(pattern=[[1, h], [1, 1]], offset=r0))
            nisa.dma_copy(dst=ring, src=tail_hbm.ap(pattern=[[width, h], [1, width]],
                                                    vector_offset=slot_t, indirect_dim=0))
        key_bf = _sb((h, head_dim), key_hbm.dtype)
        nisa.dma_copy(dst=key_bf, src=key_hbm.ap(pattern=[[head_dim, h], [1, head_dim]],
                                                 offset=r0 * head_dim))
        score_bf = _sb((h, head_dim), score_hbm.dtype)
        nisa.dma_copy(dst=score_bf, src=score_hbm.ap(pattern=[[head_dim, h], [1, head_dim]],
                                                     offset=r0 * head_dim))
        key_f = _sb((h, head_dim), nl.float32)
        nisa.tensor_copy(dst=key_f, src=key_bf)
        score_f = _sb((h, head_dim), nl.float32)
        nisa.tensor_copy(dst=score_f, src=score_bf)
        pos_i = _col(h, nl.int32)
        nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[1, h], [1, 1]], offset=r0))
        at_i = _col(h, nl.int32)
        nisa.tensor_scalar(dst=at_i, data=pos_i, op0=nl.bitwise_and, operand0=pool_size - 1)
        at_f = _col(h, nl.float32)
        nisa.tensor_copy(dst=at_f, src=at_i)

        # ---- the pool this token would close: members 0 .. last-1 from the ring ----
        totals = []
        running_max = _sb((h, head_dim), nl.float32)
        for member in range(pool_size):
            if member == last:
                score_src = score_f
            else:
                score_src = _sb((h, head_dim), nl.float32)
                row0 = (pool_size + member) * head_dim
                nisa.tensor_copy(dst=score_src, src=ring[:, row0:row0 + head_dim])
            bias = _sb((h, head_dim), nl.float32)
            nisa.dma_copy(dst=bias, src=ape_hbm.ap(pattern=[[0, h], [1, head_dim]],
                                                   offset=member * head_dim))
            total = _sb((h, head_dim), nl.float32)
            nisa.tensor_tensor(dst=total, data1=score_src, data2=bias, op=nl.add)
            totals.append(total)
            if member == 0:
                nisa.tensor_copy(dst=running_max, src=total)
            else:
                nisa.tensor_tensor(dst=running_max, data1=running_max, data2=total,
                                   op=nl.maximum)
        acc = _sb((h, head_dim), nl.float32)
        denom = _sb((h, head_dim), nl.float32)
        nisa.memset(dst=acc, value=0.0)
        nisa.memset(dst=denom, value=0.0)
        for member in range(pool_size):
            shifted = _sb((h, head_dim), nl.float32)
            nisa.tensor_tensor(dst=shifted, data1=totals[member], data2=running_max,
                               op=nl.subtract)
            weight = _sb((h, head_dim), nl.float32)
            nisa.activation(dst=weight, op=nl.exp, data=shifted)
            nisa.tensor_tensor(dst=denom, data1=denom, data2=weight, op=nl.add)
            if member == last:
                key_src = key_f
            else:
                key_src = _sb((h, head_dim), nl.float32)
                nisa.tensor_copy(dst=key_src,
                                 src=ring[:, member * head_dim:(member + 1) * head_dim])
            weighted = _sb((h, head_dim), nl.float32)
            nisa.tensor_tensor(dst=weighted, data1=weight, data2=key_src, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=weighted, op=nl.add)
        inv = _sb((h, head_dim), nl.float32)
        nisa.reciprocal(dst=inv, data=denom)
        pooled = _sb((h, head_dim), nl.float32)
        nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)
        pooled_bf = _sb((h, head_dim), tail_hbm.dtype)
        nisa.tensor_copy(dst=pooled_bf, src=pooled)
        pooled_rt = _sb((h, head_dim), nl.float32)
        nisa.tensor_copy(dst=pooled_rt, src=pooled_bf)
        scratch = _sb((h, head_dim), nl.float32)
        rotated = _fwht128_inplace(pooled_rt, scratch, head_dim)
        scaled = _sb((h, head_dim), nl.float32)
        nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
        result = _sb((h, head_dim), tail_hbm.dtype)
        nisa.tensor_copy(dst=result, src=scaled)
        nisa.dma_copy(dst=pooled_hbm.ap(pattern=[[head_dim, h], [1, head_dim]],
                                        offset=r0 * head_dim), src=result)

        # ---- the stash: this token at row position % pool_size of each half ---------
        ones = _sb((h, head_dim), nl.float32)
        nisa.memset(dst=ones, value=1.0)
        for row in range(pool_size):
            hit = _col(h, nl.float32)
            nisa.tensor_scalar(dst=hit, data=at_f, op0=nl.equal, operand0=float(row))
            mask = _sb((h, head_dim), nl.uint8)
            nisa.tensor_scalar(dst=mask, data=ones, op0=nl.multiply, operand0=hit)
            for half in range(TAIL_HALVES):
                row0 = (half * pool_size + row) * head_dim
                nisa.tensor_copy_predicated(dst=ring[:, row0:row0 + head_dim],
                                            src=key_bf if half == 0 else score_bf,
                                            predicate=mask)
        nisa.dma_copy(dst=rings_hbm.ap(pattern=[[width, h], [1, width]], offset=r0 * width),
                      src=ring)
    return pooled_hbm, rings_hbm


# ---------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------


@nki.jit
def dsa_decode_scores_kernel(q_hbm, w_hbm, bank_hbm, slots_hbm, lens_hbm, pos_hbm,
                             pooled_hbm, candidates, pool_size, source_digest):
    """Bounded candidate scores for ``B`` decode requests, each on its own slot.

    Args:
        q_hbm: ``[B, heads, head_dim]`` bf16, the rotated indexer query.
        w_hbm: ``[B, heads]`` fp32, the head weights, every scale folded in.
        bank_hbm: ``[slots, rows, head_dim]`` bf16, the pooled-key bank.
        slots_hbm / lens_hbm / pos_hbm: ``[B, 1]`` int32, each request's slot, length
            (this step's token included) and position.
        pooled_hbm: ``[B, head_dim]`` bf16, this step's compressed pool, which stands in
            at candidate ``position // pool_size`` where this step closes that pool.
        candidates: python int, the candidate axis (``max_seq_len // pool_size``).
        pool_size: python int, a power of two.
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``[B, candidates]`` fp32: ``sum_h w[h] * relu(q[h] . k[c])`` where pool ``c`` is
        complete for the request (``(c + 1) * pool_size <= length``) and ``BOUND_FILL``
        where it is not.

    Request ``b`` reads rows ``slot[b] * rows + [0, candidates)`` of the bank and nothing
    else. Candidate ``c`` sits on partition ``c % 128`` of tile ``c // 128``. The tiles
    are walked in chunks of ``CHUNK_TILES``; one extra tile per chunk carries
    ``pooled[b]`` on every partition, so its score is computed by the very instructions
    that score a bank row, and it is copied into the column it stands for.
    """
    batch = q_hbm.shape[0]
    heads = q_hbm.shape[1]
    head_dim = q_hbm.shape[2]
    out = nl.ndarray((batch, candidates), dtype=nl.float32, buffer=nl.shared_hbm)
    full = candidates // PARTITIONS
    rem = candidates - full * PARTITIONS
    n_tiles = full
    if rem > 0:
        n_tiles = full + 1
    chunk_max = min(CHUNK_TILES, n_tiles)
    shift = _log2(pool_size)
    kv_dtype = bank_hbm.dtype
    one_slot = bank_hbm.shape[0] == 1

    # Request-independent: candidate index and the first token past its pool.
    cand_f = _sb((PARTITIONS, n_tiles), nl.float32)
    nisa.iota(dst=cand_f, pattern=[[PARTITIONS, n_tiles]], offset=0, channel_multiplier=1)
    end_f = _sb((PARTITIONS, n_tiles), nl.float32)
    nisa.tensor_scalar(dst=end_f, data=cand_f, op0=nl.multiply, operand0=float(pool_size),
                       op1=nl.add, operand1=float(pool_size))
    fill = _sb((PARTITIONS, chunk_max), nl.float32)
    nisa.memset(dst=fill, value=BOUND_FILL)

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for b in nl.affine_range(prg, batch, n_prgs):
        # ---- this request's slot, length and position --------------------------------
        slot_t = _sb((1, _LINE), nl.int32)
        nisa.dma_copy(dst=slot_t[0:1, 0:1], src=slots_hbm.ap(pattern=[[1, 1], [1, 1]],
                                                             offset=b))
        len_i = _col(PARTITIONS, nl.int32)
        nisa.dma_copy(dst=len_i, src=lens_hbm.ap(pattern=[[0, PARTITIONS], [1, 1]], offset=b))
        len_f = _col(PARTITIONS, nl.float32)
        nisa.tensor_copy(dst=len_f, src=len_i)
        neg_len = _col(PARTITIONS, nl.float32)
        nisa.tensor_scalar(dst=neg_len, data=len_f, op0=nl.multiply, operand0=-1.0)
        pos_i = _col(PARTITIONS, nl.int32)
        nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[0, PARTITIONS], [1, 1]], offset=b))
        pool_i = _col(PARTITIONS, nl.int32)
        nisa.tensor_scalar(dst=pool_i, data=pos_i, op0=nl.right_shift, operand0=shift)
        ring_i = _col(PARTITIONS, nl.int32)
        nisa.tensor_scalar(dst=ring_i, data=pos_i, op0=nl.bitwise_and,
                           operand0=pool_size - 1)
        pool_f = _col(PARTITIONS, nl.float32)
        nisa.tensor_copy(dst=pool_f, src=pool_i)
        ring_f = _col(PARTITIONS, nl.float32)
        nisa.tensor_copy(dst=ring_f, src=ring_i)
        # at = position // pool_size where this step closes that pool, else -1.
        closes = _col(PARTITIONS, nl.float32)
        nisa.tensor_scalar(dst=closes, data=ring_f, op0=nl.equal,
                           operand0=float(pool_size - 1))
        at_f = _col(PARTITIONS, nl.float32)
        nisa.tensor_scalar(dst=at_f, data=pool_f, op0=nl.add, operand0=1.0)
        nisa.tensor_tensor(dst=at_f, data1=at_f, data2=closes, op=nl.multiply)
        nisa.tensor_scalar(dst=at_f, data=at_f, op0=nl.add, operand0=-1.0)

        # ---- the query onto the head dimension, the weights onto every partition ------
        # A PE transpose writes PSUM in its input's dtype and PSUM writes are whole
        # 4-byte words, so the query is transposed in fp32 (exact) and rounded back,
        # losslessly, to bf16.
        q_nat = _sb((heads, head_dim), q_hbm.dtype)
        nisa.dma_copy(dst=q_nat, src=q_hbm.ap(pattern=[[head_dim, heads], [1, head_dim]],
                                              offset=b * heads * head_dim))
        q_f = _sb((heads, head_dim), nl.float32)
        nisa.tensor_copy(dst=q_f, src=q_nat)
        qt_ps = nl.ndarray((head_dim, heads), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qt_ps, data=q_f)
        q_t = _sb((head_dim, heads), q_hbm.dtype)
        nisa.tensor_copy(dst=q_t, src=qt_ps)
        w_rep = _sb((PARTITIONS, heads), nl.float32)
        nisa.dma_copy(dst=w_rep, src=w_hbm.ap(pattern=[[0, PARTITIONS], [1, heads]],
                                              offset=b * heads))

        for t0 in range(0, n_tiles, CHUNK_TILES):
            # ---- one chunk: tiles t0 .. t0 + ct - 1, the last one ragged at the end ---
            ct = min(CHUNK_TILES, n_tiles - t0)
            c_rem = 0
            if t0 + ct == n_tiles:
                c_rem = rem
            c_full = ct
            if c_rem > 0:
                c_full = ct - 1
            virt = ct
            n_all = ct + 1
            per_bank = PSUM_FP32 // heads
            if per_bank > n_all:
                per_bank = n_all

            # ---- this chunk's keys of the request's slot, and this step's pool -------
            # One slot: the only valid slot is 0, so the reads are static. (Tracing in
            # CPU simulation fills int operands with ones, and slot 1 is past this bank.)
            k_rows = _sb((PARTITIONS, n_all, head_dim), kv_dtype)
            if c_full > 0:
                if one_slot:
                    nisa.dma_copy(
                        dst=k_rows[:, 0:c_full, :],
                        src=bank_hbm.ap(pattern=[[head_dim, PARTITIONS],
                                                 [PARTITIONS * head_dim, c_full],
                                                 [1, head_dim]],
                                        offset=t0 * PARTITIONS * head_dim))
                else:
                    nisa.dma_copy(
                        dst=k_rows[:, 0:c_full, :],
                        src=bank_hbm.ap(pattern=[[head_dim, PARTITIONS],
                                                 [PARTITIONS * head_dim, c_full],
                                                 [1, head_dim]],
                                        offset=t0 * PARTITIONS * head_dim,
                                        scalar_offset=slot_t[0:1, 0:1], indirect_dim=0))
            if c_rem > 0:
                nisa.memset(dst=k_rows[:, c_full, :], value=0.0)
                if one_slot:
                    nisa.dma_copy(
                        dst=k_rows[0:c_rem, c_full, :],
                        src=bank_hbm.ap(pattern=[[head_dim, c_rem], [1, head_dim]],
                                        offset=full * PARTITIONS * head_dim))
                else:
                    nisa.dma_copy(
                        dst=k_rows[0:c_rem, c_full, :],
                        src=bank_hbm.ap(pattern=[[head_dim, c_rem], [1, head_dim]],
                                        offset=full * PARTITIONS * head_dim,
                                        scalar_offset=slot_t[0:1, 0:1], indirect_dim=0))
            nisa.dma_copy(dst=k_rows[:, virt, :],
                          src=pooled_hbm.ap(pattern=[[0, PARTITIONS], [1, head_dim]],
                                            offset=b * head_dim))

            # ---- keys onto the head dimension -----------------------------------------
            k_t = _sb((head_dim, n_all * PARTITIONS), kv_dtype)
            for g0 in range(0, n_all, TRANSPOSE_GROUP):
                gn = min(TRANSPOSE_GROUP, n_all - g0)
                t_ps = nl.ndarray((head_dim, TRANSPOSE_GROUP * PARTITIONS), dtype=kv_dtype,
                                  buffer=nl.psum)
                for i in range(gn):
                    nisa.nc_transpose(dst=t_ps[:, i * PARTITIONS:(i + 1) * PARTITIONS],
                                      data=k_rows[:, g0 + i, :])
                nisa.tensor_copy(dst=k_t[:, g0 * PARTITIONS:(g0 + gn) * PARTITIONS],
                                 src=t_ps[:, 0:gn * PARTITIONS])

            # ---- per-head scores, rectified, weighted, summed over heads ---------------
            scores = _sb((PARTITIONS, n_all), nl.float32)
            for s0 in range(0, n_all, per_bank):
                sn = min(per_bank, n_all - s0)
                s_ps = nl.ndarray((PARTITIONS, per_bank, heads), dtype=nl.float32,
                                  buffer=nl.psum)
                for i in range(sn):
                    t = s0 + i
                    nisa.nc_matmul(dst=s_ps[:, i, :],
                                   stationary=k_t[:, t * PARTITIONS:(t + 1) * PARTITIONS],
                                   moving=q_t, accumulate=False)
                prod = _sb((PARTITIONS, per_bank, heads), nl.float32)
                nisa.scalar_tensor_tensor(
                    dst=prod[:, 0:sn, :], data=s_ps[:, 0:sn, :], op0=nl.maximum,
                    operand0=0.0, op1=nl.multiply,
                    operand1=w_rep.ap(pattern=[[heads, PARTITIONS], [0, sn], [1, heads]]))
                nisa.tensor_reduce(dst=scores[:, s0:s0 + sn], op=nl.add,
                                   data=prod[:, 0:sn, :], axis=2)

            # ---- this step's pool at its own column, then the causal bound ------------
            cand_c = cand_f[:, t0:t0 + ct]
            stand_in = _sb((PARTITIONS, ct), nl.float32)
            nisa.tensor_scalar(dst=stand_in, data=cand_c, op0=nl.multiply, operand0=0.0,
                               op1=nl.add, operand1=scores[:, virt:virt + 1])
            hit = _sb((PARTITIONS, ct), nl.uint8)
            nisa.tensor_scalar(dst=hit, data=cand_c, op0=nl.equal, operand0=at_f)
            nisa.tensor_copy_predicated(dst=scores[:, 0:ct], src=stand_in, predicate=hit)
            room = _sb((PARTITIONS, ct), nl.float32)
            nisa.tensor_scalar(dst=room, data=end_f[:, t0:t0 + ct], op0=nl.add,
                               operand0=neg_len)
            bounded = _sb((PARTITIONS, ct), nl.uint8)
            nisa.tensor_scalar(dst=bounded, data=room, op0=nl.greater, operand0=0.0)
            nisa.tensor_copy_predicated(dst=scores[:, 0:ct], src=fill[:, 0:ct],
                                        predicate=bounded)

            # ---- candidate-major out: tiles onto the partitions, rows stored whole -----
            o_ps = nl.ndarray((ct, PARTITIONS), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=o_ps, data=scores[:, 0:ct])
            o_sb = _sb((ct, PARTITIONS), nl.float32)
            nisa.tensor_copy(dst=o_sb, src=o_ps)
            if c_full > 0:
                nisa.dma_copy(dst=out.ap(pattern=[[PARTITIONS, c_full], [1, PARTITIONS]],
                                         offset=b * candidates + t0 * PARTITIONS),
                              src=o_sb[0:c_full, :])
            if c_rem > 0:
                nisa.dma_copy(dst=out.ap(pattern=[[c_rem, 1], [1, c_rem]],
                                         offset=b * candidates + full * PARTITIONS),
                              src=o_sb[c_full:c_full + 1, 0:c_rem])
    return out


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def _programs(batch: int) -> int:
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and batch >= 2:
        return 2
    return 1


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and value & (value - 1) == 0


def _require_vector(name: str, value: Tensor, batch: int) -> None:
    if not torch.is_tensor(value) or tuple(value.shape) != (batch,):
        raise DecodeBatchError(f"{name} must be a [{batch}] tensor, one entry per request; "
                               f"got {value!r}")
    if value.dtype not in (torch.int32, torch.int64):
        raise DecodeBatchError(f"{name} must be int32 or int64; got {value.dtype}")


def _require_ring(tail_bank, slots, key, score, ape, position) -> tuple[int, int, int]:
    if tail_bank.ndim != 4 or int(tail_bank.shape[1]) != TAIL_HALVES:
        raise DecodeBatchError(
            f"tail must be the ring bank [slots, {TAIL_HALVES}, pool_size, head_dim]; got "
            f"{tuple(tail_bank.shape)}")
    pool, head_dim = int(tail_bank.shape[2]), int(tail_bank.shape[3])
    if not _is_power_of_two(pool):
        raise DecodeBatchError(f"pool_size must be a power of two; got {pool}")
    if head_dim != INDEX_HEAD_DIM:
        raise DecodeBatchError(f"the rotation is a {INDEX_HEAD_DIM}-point transform; got "
                               f"head_dim {head_dim}")
    if key.ndim != 2 or int(key.shape[1]) != head_dim or tuple(score.shape) != tuple(key.shape):
        raise DecodeBatchError(f"key and score must both be [B, {head_dim}]; got "
                               f"{tuple(key.shape)} and {tuple(score.shape)}")
    batch = int(key.shape[0])
    _require_vector("slots", slots, batch)
    _require_vector("position", position, batch)
    if tuple(ape.shape) != (pool, head_dim):
        raise DecodeBatchError(f"ape must be [pool_size, head_dim] = {(pool, head_dim)}; got "
                               f"{tuple(ape.shape)}")
    return batch, pool, head_dim


def dsa_decode_ring_step(tail_bank: Tensor, slots: Tensor, key: Tensor, score: Tensor,
                         ape: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
    """Advance each request's ring by its own token. Nothing is written in place.

    Args:
        tail_bank: ``[slots, 2, pool_size, head_dim]`` bf16, the ring bank (keys in half
            0, gate scores in half 1), read at ``slots`` only.
        slots: ``[B]`` int, each request's slot; distinct.
        key / score: ``[B, head_dim]`` bf16, this step's key and gate score per request.
        ape: ``[pool_size, head_dim]`` fp32.
        position: ``[B]`` int, each request's absolute position.

    Returns:
        ``(pooled, rings)``: ``[B, head_dim]`` the pool each request closes when
        ``position % pool_size == pool_size - 1`` (meaningless otherwise), and
        ``[B, 2, pool_size, head_dim]`` the advanced rings, for the caller to write back at
        ``slots``.
    """
    batch, pool, head_dim = _require_ring(tail_bank, slots, key, score, ape, position)
    usable = (can_run_kernel(key) and tail_bank.dtype == torch.bfloat16
              and key.dtype == torch.bfloat16 and score.dtype == torch.bfloat16)
    if not usable:
        _count_torch_fallback()
        return dsa_decode_ring_step_torch_oracle(tail_bank, slots, key, score, ape, position)
    _count_dispatch("ring", 1, batch)
    width = TAIL_HALVES * pool * head_dim
    pooled, rings = wrap_nki(dsa_decode_ring_step_kernel)(
        tail_bank.reshape(int(tail_bank.shape[0]), width).contiguous(),
        slots.reshape(batch, 1).to(torch.int32).contiguous(),
        key.contiguous(), score.contiguous(), ape.to(torch.float32).contiguous(),
        position.reshape(batch, 1).to(torch.int32).contiguous(), pool, SOURCE_DIGEST)
    return pooled, rings.reshape(batch, TAIL_HALVES, pool, head_dim)


def _require_scores(query, weights, pool_bank, slots, seq_lens, position, pooled,
                    candidates, pool_size) -> tuple[int, int]:
    if not isinstance(pool_size, int) or not _is_power_of_two(pool_size):
        raise DecodeBatchError(f"pool_size must be a power-of-two python int; got "
                               f"{pool_size!r}")
    if query.ndim != 3 or int(query.shape[2]) != INDEX_HEAD_DIM:
        raise DecodeBatchError(f"query must be [B, heads, {INDEX_HEAD_DIM}]; got "
                               f"{tuple(query.shape)}")
    batch, heads = int(query.shape[0]), int(query.shape[1])
    if not 1 <= heads <= PARTITIONS:
        raise DecodeBatchError(f"heads must lie in [1, {PARTITIONS}]; got {heads}")
    if tuple(weights.shape) != (batch, heads):
        raise DecodeBatchError(f"weights must be [B, heads] = {(batch, heads)}; got "
                               f"{tuple(weights.shape)}")
    if pool_bank.ndim != 3 or int(pool_bank.shape[2]) != INDEX_HEAD_DIM:
        raise DecodeBatchError(f"pool_bank must be [slots, rows, {INDEX_HEAD_DIM}]; got "
                               f"{tuple(pool_bank.shape)}")
    if not isinstance(candidates, int) or not 1 <= candidates < int(pool_bank.shape[1]):
        raise DecodeBatchError(
            f"candidates must be a python int below the bank's {int(pool_bank.shape[1])} "
            f"rows per slot (the last row is the write trash); got {candidates!r}")
    for name, value in (("slots", slots), ("seq_lens", seq_lens), ("position", position)):
        _require_vector(name, value, batch)
    if tuple(pooled.shape) != (batch, INDEX_HEAD_DIM):
        raise DecodeBatchError(f"pooled must be [B, {INDEX_HEAD_DIM}]; got "
                               f"{tuple(pooled.shape)}")
    return batch, heads


def dsa_decode_scores(query: Tensor, weights: Tensor, pool_bank: Tensor, slots: Tensor,
                      seq_lens: Tensor, position: Tensor, pooled: Tensor, *,
                      candidates: int, pool_size: int) -> Tensor:
    """Each request's bounded candidate scores, read from its own slot. ``[B, C]`` fp32.

    Args:
        query: ``[B, heads, head_dim]`` bf16, the rotated indexer query.
        weights: ``[B, heads]`` fp32, scale folded in.
        pool_bank: ``[slots, rows, head_dim]`` bf16, ``rows > candidates``.
        slots: ``[B]`` int, each request's slot.
        seq_lens: ``[B]`` int, each request's length with this step's token.
        position: ``[B]`` int, this step's position per request.
        pooled: ``[B, head_dim]``, this step's compressed pool per request
            (:func:`dsa_decode_ring_step`), used where this step closes a pool.
        candidates: the shared candidate axis, ``max_seq_len // pool_size``.
        pool_size: tokens per pool.

    Returns:
        ``dsa_causal_bound(dsa_score_gemm(...))`` per request, without a launch per
        request.
    """
    batch, heads = _require_scores(query, weights, pool_bank, slots, seq_lens, position,
                                   pooled, candidates, pool_size)
    usable = (can_run_kernel(query) and query.dtype == torch.bfloat16
              and pool_bank.dtype == torch.bfloat16 and weights.dtype == torch.float32)
    if not usable:
        _count_torch_fallback()
        return dsa_decode_scores_torch_oracle(query, weights, pool_bank, slots, seq_lens,
                                              position, pooled, candidates=candidates,
                                              pool_size=pool_size)
    programs = _programs(batch)
    _count_dispatch("scores", programs, batch)
    call = wrap_nki(dsa_decode_scores_kernel)
    if programs == 2:
        call = call[2]

    def column(value):
        return value.reshape(batch, 1).to(torch.int32).contiguous()

    return call(query.contiguous(), weights.contiguous(), pool_bank.contiguous(),
                column(slots), column(seq_lens), column(position),
                pooled.to(pool_bank.dtype).contiguous(), int(candidates), int(pool_size),
                SOURCE_DIGEST)


def decode_pool_destinations(slots: Tensor, position: Tensor, *, rows: int,
                             pool_size: int) -> tuple[Tensor, Tensor]:
    """Where each request's ``pooled`` row is written: ``(slot, row)`` int64, ``[B]`` each.

    ``row`` is ``position // pool_size`` where this step closes that pool and the slot's
    trash row ``rows - 1`` otherwise, so every request writes once and a step that closes
    nothing lands where no candidate read looks.
    """
    at = position.to(torch.int64)
    closes = torch.remainder(at, pool_size) == pool_size - 1
    row = torch.where(closes, torch.div(at, pool_size, rounding_mode="floor"),
                      torch.full_like(at, int(rows) - 1))
    return slots.to(torch.int64), row


# ---------------------------------------------------------------------------------------
# Torch references, also the route a refused call takes
# ---------------------------------------------------------------------------------------


def dsa_decode_ring_step_torch_oracle(tail_bank, slots, key, score, ape, position):
    """The ring step in torch: the one-request recompute, row by row."""
    rings = tail_bank[slots.to(torch.int64)].clone()
    pool = int(rings.shape[2])
    members_k = rings[:, 0].clone()
    members_s = rings[:, 1].clone()
    members_k[:, pool - 1] = key.to(rings.dtype)
    members_s[:, pool - 1] = score.to(rings.dtype)
    pooled = torch.cat([_compress_pool_torch(members_k[b], members_s[b], ape.float())
                        for b in range(int(rings.shape[0]))])
    at = torch.remainder(position.to(torch.int64), pool)
    hit = (torch.arange(pool, device=rings.device)[None, :] == at[:, None])[:, :, None]
    rings[:, 0] = torch.where(hit, key.to(rings.dtype)[:, None, :], rings[:, 0])
    rings[:, 1] = torch.where(hit, score.to(rings.dtype)[:, None, :], rings[:, 1])
    return pooled, rings


def dsa_decode_scores_torch_oracle(query, weights, pool_bank, slots, seq_lens, position,
                                   pooled, *, candidates, pool_size):
    """The score stage in torch: gather each slot, stand this step's pool in, score, bound."""
    keys = pool_bank[slots.to(torch.int64), :candidates].float()
    at = position.to(torch.int64)
    closes = torch.remainder(at, pool_size) == pool_size - 1
    column = torch.arange(candidates, device=keys.device)
    stand = (column[None, :] == torch.div(at, pool_size, rounding_mode="floor")[:, None])
    stand = (stand & closes[:, None])[:, :, None]
    keys = torch.where(stand, pooled.float()[:, None, :], keys)
    per_head = torch.einsum("bhd,bcd->bhc", query.float(), keys).clamp(min=0.0)
    scores = (per_head * weights.float()[:, :, None]).sum(dim=1)
    bounded = (column[None, :] + 1) * pool_size > seq_lens.to(torch.int64)[:, None]
    return scores.masked_fill(bounded, BOUND_FILL)
