# SPDX-License-Identifier: Apache-2.0
"""DSA indexer candidate scores for ``T`` query rows per request in one step.

A speculative verify step hands the indexer ``T = 1 + k`` tokens per request at positions
``start .. start + T - 1`` in one call. :mod:`decode_batch` scores one row per request
and stands one pool in -- the one that row closes -- because every earlier pool is in the
store. With ``T`` rows the pools rows ``0 .. t - 1`` close are not in the store yet when
row ``t`` is scored (the write lands after the step), so :func:`dsa_decode_scores_rows`
stands every pool a row ``t' <= t`` closes in from the ring step's ``pooled`` rows and
bounds each row at its own length ``start + t + 1``. The candidate rows of a request's
slot are read once per request; each row runs the one-row kernel's matmul shapes, so the
``T``-row step equals ``T`` sequential one-row steps bit for bit on the simulator.

The ring advance for ``T`` rows is :func:`decode_tail_update.dsa_decode_ring_rows`, whose
``pooled`` output (``[B * T, head_dim]``, request-major) is this kernel's ``pooled``
operand.
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
from vllm_neuron.utils.neuron_utils import can_run_kernel, values_are_readable

logger = logging.getLogger(__name__)

PARTITIONS = 128
"""SBUF partitions: candidates ride them, ``PARTITIONS`` per tile."""
TRANSPOSE_GROUP = 4
"""Candidate tiles transposed into one PSUM tile before the copy out."""
PSUM_FP32 = 512
"""fp32 columns of one PSUM bank; the head scores of ``PSUM_FP32 // heads`` candidate
tiles share one."""
_LINE = 8
"""Elements of the one-partition line a scalar slot is read into."""

SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)
"""A digest of this file, passed to the kernel so the kernel cache keys on its source."""


class DecodeTrowError(ValueError):
    """A malformed ``T``-row call: wrong rank, rows that do not split over the requests,
    a position past the window, or a candidate axis without a trash row."""


@dataclass
class _DecodeTrowCounters:
    nki_dispatch: int = 0
    torch_fallback: int = 0
    one_program: int = 0
    two_programs: int = 0


_COUNTERS = _DecodeTrowCounters()


def reset_decode_trow_dispatch_counters() -> None:
    """Zero this module's counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.one_program = 0
    _COUNTERS.two_programs = 0


def decode_trow_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def decode_trow_route_counts() -> tuple[int, int]:
    """``(one_program, two_programs)`` launches since the last reset."""
    return (_COUNTERS.one_program, _COUNTERS.two_programs)


@torch._dynamo.assume_constant_result
def _count_dispatch(programs: int, batch: int, rows: int) -> None:
    _COUNTERS.nki_dispatch += 1
    if programs == 2:
        _COUNTERS.two_programs += 1
    else:
        _COUNTERS.one_program += 1
    logger.info("[dsa-decode-trow] kernel=nki entry=scores batch=%d rows=%d programs=%d",
                batch, rows, programs)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    _COUNTERS.torch_fallback += 1


def indexer_ring_depth(index_kpool: int, num_speculative_tokens: int | None) -> int:
    """The indexer ring's depth for a server that drafts ``num_speculative_tokens``.

    The one site this number is derived at: the runner sizes ``tail`` and ``pad_tail``
    as ``[slots, 2, depth, index_head_dim]`` with it, and the decode legs read the depth
    back off that shape. A verify step hands the ring ``1 + num_speculative_tokens``
    rows, and a ring of depth ``R`` takes ``R - index_kpool + 2`` rows back on rollback
    (:func:`decode_tail_update.max_rows_for`), so the depth is the smallest power of two
    at or above ``max(index_kpool, (1 + num_speculative_tokens) + index_kpool - 2)`` --
    ``max(4, num_speculative_tokens + 3)`` at this checkpoint's pool of 4. A server that drafts
    nothing (``0`` or ``None``) gets ``index_kpool``: today's ring, shape and kernels
    untouched.
    """
    from vllm_neuron.functional.dsa.decode_tail_update import ring_depth_for

    drafts = 0 if num_speculative_tokens is None else int(num_speculative_tokens)
    if drafts < 0:
        raise DecodeTrowError(
            f"num_speculative_tokens is a count of drafted tokens; got {drafts}"
        )
    return ring_depth_for(int(index_kpool), 1 + drafts)


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    return nl.ndarray((parts, 1), dtype=dtype, buffer=nl.sbuf)


def _log2(value):
    shift = 0
    while (1 << shift) < value:
        shift += 1
    return shift


@nki.jit
def dsa_decode_scores_rows_kernel(q_hbm, w_hbm, bank_hbm, slots_hbm, pos_hbm, pooled_hbm,
                                  candidates, pool_size, rows, source_digest):
    """Bounded candidate scores for ``B`` requests x ``rows`` query rows, each request on
    its own slot.

    Args:
        q_hbm: ``[B * rows, heads, head_dim]`` bf16, the rotated indexer queries,
            request-major: request ``b``'s row ``t`` is row ``b * rows + t``.
        w_hbm: ``[B * rows, heads]`` fp32, the head weights, every scale folded in.
        bank_hbm: ``[slots, store_rows, head_dim]`` bf16, the pooled-key bank.
        slots_hbm / pos_hbm: ``[B, 1]`` int32, each request's slot and the position of
            its row 0.
        pooled_hbm: ``[B * rows, head_dim]`` bf16, the pool each row closes (the ring
            step's output); row ``b * rows + t`` stands in at candidate
            ``(pos[b] + t) // pool_size`` for every row ``t'' >= t`` of request ``b``
            where position ``pos[b] + t`` closes a pool.
        candidates: python int, the candidate axis (``max_seq_len // pool_size``).
        pool_size: python int, a power of two.
        rows: python int, query rows per request.
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``[B * rows, candidates]`` fp32: row ``b * rows + t`` is ``sum_h w[h] * relu(q[h]
        . k[c])`` where pool ``c`` is complete at length ``pos[b] + t + 1``
        (``(c + 1) * pool_size <= length``) and ``BOUND_FILL`` where it is not.

    Request ``b`` reads rows ``slot[b] * store_rows + [0, candidates)`` of the bank once
    and nothing else. Candidate ``c`` sits on partition ``c % 128`` of tile ``c // 128``;
    ``rows`` more tiles carry the request's ``pooled`` rows on every partition, so each is
    scored by the very instructions that score a bank row (the one-row kernel's shapes)
    and copied into the column it stands for.
    """
    batch = slots_hbm.shape[0]
    heads = q_hbm.shape[1]
    head_dim = q_hbm.shape[2]
    out = nl.ndarray((batch * rows, candidates), dtype=nl.float32, buffer=nl.shared_hbm)
    full = candidates // PARTITIONS
    rem = candidates - full * PARTITIONS
    n_tiles = full
    if rem > 0:
        n_tiles = full + 1
    virt = n_tiles
    n_all = n_tiles + rows
    per_bank = PSUM_FP32 // heads
    if per_bank > n_all:
        per_bank = n_all
    shift = _log2(pool_size)
    kv_dtype = bank_hbm.dtype
    one_slot = bank_hbm.shape[0] == 1

    # Request-independent: candidate index and the first token past its pool.
    cand_f = _sb((PARTITIONS, n_tiles), nl.float32)
    nisa.iota(dst=cand_f, pattern=[[PARTITIONS, n_tiles]], offset=0, channel_multiplier=1)
    end_f = _sb((PARTITIONS, n_tiles), nl.float32)
    nisa.tensor_scalar(dst=end_f, data=cand_f, op0=nl.multiply, operand0=float(pool_size),
                       op1=nl.add, operand1=float(pool_size))
    fill = _sb((PARTITIONS, n_tiles), nl.float32)
    nisa.memset(dst=fill, value=BOUND_FILL)

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for b in nl.affine_range(prg, batch, n_prgs):
        # ---- this request's slot and row-0 position ------------------------------------
        slot_t = _sb((1, _LINE), nl.int32)
        nisa.dma_copy(dst=slot_t[0:1, 0:1], src=slots_hbm.ap(pattern=[[1, 1], [1, 1]],
                                                             offset=b))
        pos_i = _col(PARTITIONS, nl.int32)
        nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[0, PARTITIONS], [1, 1]], offset=b))
        pos_f = _col(PARTITIONS, nl.float32)
        nisa.tensor_copy(dst=pos_f, src=pos_i)
        # Per row: the candidate its pool stands in at (position // pool_size where the
        # row closes a pool, else -1) and minus its length.
        stand_at = []
        neg_len = []
        for t in range(rows):
            pos_t = _col(PARTITIONS, nl.int32)
            nisa.tensor_scalar(dst=pos_t, data=pos_i, op0=nl.add, operand0=t)
            pool_i = _col(PARTITIONS, nl.int32)
            nisa.tensor_scalar(dst=pool_i, data=pos_t, op0=nl.right_shift, operand0=shift)
            ring_i = _col(PARTITIONS, nl.int32)
            nisa.tensor_scalar(dst=ring_i, data=pos_t, op0=nl.bitwise_and,
                               operand0=pool_size - 1)
            pool_f = _col(PARTITIONS, nl.float32)
            nisa.tensor_copy(dst=pool_f, src=pool_i)
            ring_f = _col(PARTITIONS, nl.float32)
            nisa.tensor_copy(dst=ring_f, src=ring_i)
            closes = _col(PARTITIONS, nl.float32)
            nisa.tensor_scalar(dst=closes, data=ring_f, op0=nl.equal,
                               operand0=float(pool_size - 1))
            at_f = _col(PARTITIONS, nl.float32)
            nisa.tensor_scalar(dst=at_f, data=pool_f, op0=nl.add, operand0=1.0)
            nisa.tensor_tensor(dst=at_f, data1=at_f, data2=closes, op=nl.multiply)
            nisa.tensor_scalar(dst=at_f, data=at_f, op0=nl.add, operand0=-1.0)
            stand_at.append(at_f)
            minus = _col(PARTITIONS, nl.float32)
            nisa.tensor_scalar(dst=minus, data=pos_f, op0=nl.add, operand0=float(t + 1),
                               op1=nl.multiply, operand1=-1.0)
            neg_len.append(minus)

        # ---- the keys of this request's slot once, and its rows' pools as more tiles ---
        # One slot: the only valid slot is 0, so the reads are static. (Tracing in CPU
        # simulation fills int operands with ones, and slot 1 is past this bank.)
        k_rows = _sb((PARTITIONS, n_all, head_dim), kv_dtype)
        if full > 0:
            if one_slot:
                nisa.dma_copy(
                    dst=k_rows[:, 0:full, :],
                    src=bank_hbm.ap(pattern=[[head_dim, PARTITIONS],
                                             [PARTITIONS * head_dim, full], [1, head_dim]],
                                    offset=0))
            else:
                nisa.dma_copy(
                    dst=k_rows[:, 0:full, :],
                    src=bank_hbm.ap(pattern=[[head_dim, PARTITIONS],
                                             [PARTITIONS * head_dim, full], [1, head_dim]],
                                    offset=0, scalar_offset=slot_t[0:1, 0:1], indirect_dim=0))
        if rem > 0:
            nisa.memset(dst=k_rows[:, full, :], value=0.0)
            if one_slot:
                nisa.dma_copy(
                    dst=k_rows[0:rem, full, :],
                    src=bank_hbm.ap(pattern=[[head_dim, rem], [1, head_dim]],
                                    offset=full * PARTITIONS * head_dim))
            else:
                nisa.dma_copy(
                    dst=k_rows[0:rem, full, :],
                    src=bank_hbm.ap(pattern=[[head_dim, rem], [1, head_dim]],
                                    offset=full * PARTITIONS * head_dim,
                                    scalar_offset=slot_t[0:1, 0:1], indirect_dim=0))
        for t in range(rows):
            nisa.dma_copy(dst=k_rows[:, virt + t, :],
                          src=pooled_hbm.ap(pattern=[[0, PARTITIONS], [1, head_dim]],
                                            offset=(b * rows + t) * head_dim))

        # ---- keys onto the head dimension ---------------------------------------------
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

        for t in range(rows):
            row = b * rows + t
            # ---- this row's query onto the head dimension, its weights on every
            # partition. A PE transpose writes PSUM in its input's dtype and PSUM writes
            # are whole 4-byte words, so the query is transposed in fp32 (exact) and
            # rounded back, losslessly, to bf16.
            q_nat = _sb((heads, head_dim), q_hbm.dtype)
            nisa.dma_copy(dst=q_nat, src=q_hbm.ap(pattern=[[head_dim, heads], [1, head_dim]],
                                                  offset=row * heads * head_dim))
            q_f = _sb((heads, head_dim), nl.float32)
            nisa.tensor_copy(dst=q_f, src=q_nat)
            qt_ps = nl.ndarray((head_dim, heads), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=qt_ps, data=q_f)
            q_t = _sb((head_dim, heads), q_hbm.dtype)
            nisa.tensor_copy(dst=q_t, src=qt_ps)
            w_rep = _sb((PARTITIONS, heads), nl.float32)
            nisa.dma_copy(dst=w_rep, src=w_hbm.ap(pattern=[[0, PARTITIONS], [1, heads]],
                                                  offset=row * heads))

            # ---- per-head scores, rectified, weighted, summed over heads ---------------
            scores = _sb((PARTITIONS, n_all), nl.float32)
            for s0 in range(0, n_all, per_bank):
                sn = min(per_bank, n_all - s0)
                s_ps = nl.ndarray((PARTITIONS, per_bank, heads), dtype=nl.float32,
                                  buffer=nl.psum)
                for i in range(sn):
                    tile = s0 + i
                    nisa.nc_matmul(dst=s_ps[:, i, :],
                                   stationary=k_t[:, tile * PARTITIONS:(tile + 1) * PARTITIONS],
                                   moving=q_t, accumulate=False)
                prod = _sb((PARTITIONS, per_bank, heads), nl.float32)
                nisa.scalar_tensor_tensor(
                    dst=prod[:, 0:sn, :], data=s_ps[:, 0:sn, :], op0=nl.maximum, operand0=0.0,
                    op1=nl.multiply,
                    operand1=w_rep.ap(pattern=[[heads, PARTITIONS], [0, sn], [1, heads]]))
                nisa.tensor_reduce(dst=scores[:, s0:s0 + sn], op=nl.add,
                                   data=prod[:, 0:sn, :], axis=2)

            # ---- the pools this step closed up to this row, at their own columns -------
            for tp in range(t + 1):
                stand_in = _sb((PARTITIONS, n_tiles), nl.float32)
                nisa.tensor_scalar(dst=stand_in, data=cand_f, op0=nl.multiply, operand0=0.0,
                                   op1=nl.add, operand1=scores[:, virt + tp:virt + tp + 1])
                hit = _sb((PARTITIONS, n_tiles), nl.uint8)
                nisa.tensor_scalar(dst=hit, data=cand_f, op0=nl.equal, operand0=stand_at[tp])
                nisa.tensor_copy_predicated(dst=scores[:, 0:n_tiles], src=stand_in,
                                            predicate=hit)
            # ---- this row's causal bound ----------------------------------------------
            room = _sb((PARTITIONS, n_tiles), nl.float32)
            nisa.tensor_scalar(dst=room, data=end_f, op0=nl.add, operand0=neg_len[t])
            bounded = _sb((PARTITIONS, n_tiles), nl.uint8)
            nisa.tensor_scalar(dst=bounded, data=room, op0=nl.greater, operand0=0.0)
            nisa.tensor_copy_predicated(dst=scores[:, 0:n_tiles], src=fill, predicate=bounded)

            # ---- candidate-major out: tiles onto the partitions, rows stored whole -----
            o_ps = nl.ndarray((n_tiles, PARTITIONS), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=o_ps, data=scores[:, 0:n_tiles])
            o_sb = _sb((n_tiles, PARTITIONS), nl.float32)
            nisa.tensor_copy(dst=o_sb, src=o_ps)
            if full > 0:
                nisa.dma_copy(dst=out.ap(pattern=[[PARTITIONS, full], [1, PARTITIONS]],
                                         offset=row * candidates),
                              src=o_sb[0:full, :])
            if rem > 0:
                nisa.dma_copy(dst=out.ap(pattern=[[rem, 1], [1, rem]],
                                         offset=row * candidates + full * PARTITIONS),
                              src=o_sb[full:full + 1, 0:rem])
    return out


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def _programs(batch: int) -> int:
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and batch >= 2:
        return 2
    return 1


def _require(query, weights, pool_bank, slots, position, pooled, candidates,
             pool_size) -> tuple[int, int]:
    """Check a :func:`dsa_decode_scores_rows` call; return ``(batch, rows)``."""
    if pool_bank.ndim != 3:
        raise DecodeTrowError(f"pool_bank must be [slots, rows, head_dim]; got "
                              f"{tuple(pool_bank.shape)}")
    head_dim = int(pool_bank.shape[2])
    if query.ndim != 3 or int(query.shape[2]) != head_dim:
        raise DecodeTrowError(f"query must be [B * rows, heads, {head_dim}]; got "
                              f"{tuple(query.shape)}")
    if not torch.is_tensor(slots) or slots.ndim != 1 or int(slots.shape[0]) < 1:
        raise DecodeTrowError(f"slots must be a [B] tensor; got {slots!r}")
    batch = int(slots.shape[0])
    total, heads = int(query.shape[0]), int(query.shape[1])
    if total < batch or total % batch:
        raise DecodeTrowError(
            f"query must hold a whole number of rows per request: {total} rows do not "
            f"split over {batch} requests")
    rows = total // batch
    if tuple(weights.shape) != (total, heads):
        raise DecodeTrowError(f"weights must be [B * rows, heads] = {(total, heads)}; got "
                              f"{tuple(weights.shape)}")
    if tuple(pooled.shape) != (total, head_dim):
        raise DecodeTrowError(f"pooled must be [B * rows, head_dim] = {(total, head_dim)}, "
                              f"one closed pool per query row; got {tuple(pooled.shape)}")
    if not torch.is_tensor(position) or tuple(position.shape) != (batch,):
        raise DecodeTrowError(f"position must be [{batch}], row 0's position per request; "
                              f"got {position!r}")
    for name, value in (("slots", slots), ("position", position)):
        if value.dtype not in (torch.int32, torch.int64):
            raise DecodeTrowError(f"{name} must be int32 or int64; got {value.dtype}")
    pool = int(pool_size)
    if pool < 2 or pool & (pool - 1):
        raise DecodeTrowError(f"pool_size must be a power of two >= 2; got {pool_size}")
    cands = int(candidates)
    if cands < 1 or cands >= int(pool_bank.shape[1]):
        raise DecodeTrowError(
            f"candidates must be at least 1 and fewer than the bank's {int(pool_bank.shape[1])} "
            f"rows per slot (the last row is the write trash); got {candidates}")
    if 1 <= heads and heads > PSUM_FP32:
        raise DecodeTrowError(f"heads must fit one PSUM bank ({PSUM_FP32}); got {heads}")
    if values_are_readable(position):
        lo, hi = int(position.min()), int(position.max())
        if lo < 0 or hi + rows - 1 >= cands * pool:
            raise DecodeTrowError(
                f"each request's {rows} rows must lie inside the window of {cands} pools x "
                f"{pool} tokens: positions [{lo}, {hi}] + {rows - 1}")
    return batch, rows


def dsa_decode_scores_rows(query: Tensor, weights: Tensor, pool_bank: Tensor, slots: Tensor,
                           position: Tensor, pooled: Tensor, *, candidates: int,
                           pool_size: int) -> Tensor:
    """Bounded candidate scores for ``B`` requests x ``rows`` query rows in one launch.

    Args:
        query: ``[B * rows, heads, head_dim]`` bf16, request-major.
        weights: ``[B * rows, heads]`` fp32, every scale folded in.
        pool_bank: ``[slots, store_rows, head_dim]`` bf16, read at ``slots`` only,
            candidates ``0 .. candidates - 1`` of each.
        slots: ``[B]`` int, each request's slot.
        position: ``[B]`` int, each request's position of row 0.
        pooled: ``[B * rows, head_dim]`` bf16, the ring step's output: the pool each row
            closes (meaningless where it closes none; not read there).
        candidates: the candidate axis (``max_seq_len // pool_size``), below the bank's
            rows per slot.
        pool_size: a power of two.

    Returns:
        ``[B * rows, candidates]`` fp32, row ``b * rows + t`` the scores of request
        ``b``'s row ``t``, ``BOUND_FILL`` past its length ``position[b] + t + 1``.
    """
    batch, rows = _require(query, weights, pool_bank, slots, position, pooled, candidates,
                           pool_size)
    usable = (can_run_kernel(query) and query.dtype == torch.bfloat16
              and pool_bank.dtype == torch.bfloat16 and pooled.dtype == torch.bfloat16)
    if not usable:
        _count_torch_fallback()
        return dsa_decode_scores_rows_torch_oracle(
            query, weights, pool_bank, slots, position, pooled, candidates=candidates,
            pool_size=pool_size)
    programs = _programs(batch)
    _count_dispatch(programs, batch, rows)
    call = wrap_nki(dsa_decode_scores_rows_kernel)
    if programs == 2:
        call = call[2]
    return call(query.contiguous(), weights.to(torch.float32).contiguous(),
                pool_bank.contiguous(), slots.reshape(batch, 1).to(torch.int32).contiguous(),
                position.reshape(batch, 1).to(torch.int32).contiguous(), pooled.contiguous(),
                int(candidates), int(pool_size), rows, SOURCE_DIGEST)


def dsa_decode_scores_rows_torch_oracle(query, weights, pool_bank, slots, position, pooled, *,
                                        candidates, pool_size):
    """The ``T``-row score stage in torch: gather each slot, stand in every pool this step
    closes up to the row, score, bound at the row's own length."""
    batch, rows = _require(query, weights, pool_bank, slots, position, pooled, candidates,
                           pool_size)
    cands, pool = int(candidates), int(pool_size)
    out = torch.empty(batch * rows, cands, dtype=torch.float32)
    column = torch.arange(cands)
    for b in range(batch):
        keys = pool_bank[int(slots[b]), :cands].float().clone()
        start = int(position[b])
        for t in range(rows):
            pos = start + t
            if pos % pool == pool - 1:
                keys[pos // pool] = pooled[b * rows + t].float()
            per_head = (query[b * rows + t].float() @ keys.t()).clamp(min=0.0)
            scores = (per_head * weights[b * rows + t].float()[:, None]).sum(dim=0)
            bounded = (column + 1) * pool > pos + 1
            out[b * rows + t] = scores.masked_fill(bounded, BOUND_FILL)
    return out
