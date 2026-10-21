# SPDX-License-Identifier: Apache-2.0
"""DSA indexer candidate scores for ``T`` query rows per request above 16384 candidates.

:func:`decode_trow.dsa_decode_scores_rows` moves each row's scores out of SBUF with one
Tensor Engine transpose of the whole ``[128, n_tiles]`` score tile, and a transpose takes
``[128, 128]`` at most. So it serves up to :data:`MAX_DECODE_SCORES_ROWS_CANDIDATES`
(16384) candidates: 128 tiles. :func:`dsa_decode_scores_rows_wide` is that kernel with
the transpose, the copy and the two stores done per group of :data:`OUTPUT_GROUP_TILES`
tiles. Every other statement of the kernel body is the old body's (a test parses both
and holds them equal), so at one group it gives the old kernel's bits, and above it each
row is scored, stood in and bounded by the very same instructions.

A 1M context under DCP 8 gives each rank 131072 tokens, 32768 candidates over pools of
4: two output groups. :data:`MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES` is that axis, the
widest built clean at one and four rows (the score and key tiles fill SBUF above it).
:func:`check_decode_scores_rows_context` refuses a wider form at start, before any graph
compiles.

The model takes this kernel only for a static candidate axis above 16384
(:func:`decode_scores_rows_for`), so a form whose axis fits one group runs the old kernel
and keeps its compiled graphs. The two bodies are one kernel kept twice for that reason.
Retirement: at the next planned re-key of the forms the old kernel serves, route every
axis here (:func:`decode_scores_rows_for` returns :func:`dsa_decode_scores_rows_wide`) and
delete the old body.
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

from vllm_neuron.functional.dsa import decode_trow as _decode_trow
from vllm_neuron.functional.dsa import kpool_hadamard as _kpool_hadamard
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.decode_trow import (
    _LINE,
    PARTITIONS,
    PSUM_FP32,
    TRANSPOSE_GROUP,
    DecodeTrowError,
    _col,
    _hadamard_signs,
    _log2,
    _programs,
    _require,
    _sb,
    dsa_decode_scores_rows,
    dsa_decode_scores_rows_torch_oracle,
)
from vllm_neuron.functional.dsa.kpool_hadamard import HADAMARD_SCALE
from vllm_neuron.source_digest import source_digest
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

OUTPUT_GROUP_TILES = PARTITIONS
"""Candidate tiles one output transpose moves: a transpose takes ``[128, 128]`` at most."""
MAX_DECODE_SCORES_ROWS_CANDIDATES = PARTITIONS * PARTITIONS
"""The old kernel's axis: one output transpose of 128 tiles (16384 candidates)."""
MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES = 2 * OUTPUT_GROUP_TILES * PARTITIONS
"""This kernel's axis: two output groups (32768 candidates, the 1M / DCP 8 rank axis)."""

SOURCE_DIGEST = source_digest(__file__, _decode_trow.__file__, _kpool_hadamard.__file__)
"""A digest of this file, of ``decode_trow.py`` (whose helpers and tile constants the
kernel reads) and of ``kpool_hadamard.py`` (the rotation's size and scale), passed to the
kernel so the kernel cache keys on all three."""


@dataclass
class _DecodeTrowWideCounters:
    nki_dispatch: int = 0
    torch_fallback: int = 0
    one_program: int = 0
    two_programs: int = 0


_COUNTERS = _DecodeTrowWideCounters()


def reset_decode_trow_wide_dispatch_counters() -> None:
    """Zero this module's counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.one_program = 0
    _COUNTERS.two_programs = 0


def decode_trow_wide_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def decode_trow_wide_route_counts() -> tuple[int, int]:
    """``(one_program, two_programs)`` launches since the last reset."""
    return (_COUNTERS.one_program, _COUNTERS.two_programs)


@torch._dynamo.assume_constant_result
def _count_dispatch(programs: int, batch: int, rows: int) -> None:
    _COUNTERS.nki_dispatch += 1
    if programs == 2:
        _COUNTERS.two_programs += 1
    else:
        _COUNTERS.one_program += 1
    logger.info("[dsa-decode-trow-wide] kernel=nki entry=scores batch=%d rows=%d programs=%d",
                batch, rows, programs)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    _COUNTERS.torch_fallback += 1


@nki.jit
def dsa_decode_scores_rows_wide_kernel(q_hbm, w_hbm, bank_hbm, slots_hbm, pos_hbm,
                                       pooled_hbm, candidates, pool_size, rows, rotate,
                                       source_digest):
    """:func:`decode_trow.dsa_decode_scores_rows_kernel` with the candidate-major output
    moved per group of :data:`OUTPUT_GROUP_TILES` tiles.

    Arguments, return and layout are the old kernel's: ``q_hbm`` ``[B * rows, heads,
    head_dim]`` bf16, ``w_hbm`` ``[B * rows, heads]`` fp32, ``bank_hbm`` ``[slots,
    store_rows, head_dim]`` bf16, ``slots_hbm`` ``[B, 1]`` int32, ``pos_hbm`` ``[B, 1]`` or
    ``[B * rows, 1]`` int32, ``pooled_hbm`` ``[B * rows, head_dim]`` bf16; ``candidates``,
    ``pool_size``, ``rows``, ``rotate`` and ``source_digest`` (:data:`SOURCE_DIGEST`)
    static. Returns ``[B * rows, candidates]`` fp32.

    Each row's scores stay ``[128, n_tiles]`` in SBUF (candidate ``c`` on partition
    ``c % 128`` of tile ``c // 128``). Group ``g`` transposes tiles ``128 g .. 128 g + 127``
    into a ``[tiles, 128]`` PSUM tile, at most 512 B a partition, and stores its whole
    tiles, then the ragged last tile if the group holds it, at the row's own columns.
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
    signs = None
    if rotate:
        signs = _hadamard_signs(q_hbm.dtype)

    # The per-row form: one position per row (``[B * rows, 1]``) instead of row 0's.
    per_row = pos_hbm.shape[0] != batch

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for b in nl.affine_range(prg, batch, n_prgs):
        # ---- this request's slot and row-0 position ------------------------------------
        slot_t = _sb((1, _LINE), nl.int32)
        nisa.dma_copy(dst=slot_t[0:1, 0:1], src=slots_hbm.ap(pattern=[[1, 1], [1, 1]],
                                                             offset=b))
        if not per_row:
            pos_i = _col(PARTITIONS, nl.int32)
            nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[0, PARTITIONS], [1, 1]],
                                                    offset=b))
            pos_f = _col(PARTITIONS, nl.float32)
            nisa.tensor_copy(dst=pos_f, src=pos_i)
        # Per row: the candidate its pool stands in at (position // pool_size where the
        # row closes a pool, else -1) and minus its length.
        stand_at = []
        neg_len = []
        for t in range(rows):
            pos_t = _col(PARTITIONS, nl.int32)
            if per_row:
                nisa.dma_copy(dst=pos_t, src=pos_hbm.ap(pattern=[[0, PARTITIONS], [1, 1]],
                                                        offset=b * rows + t))
            else:
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
            if per_row:
                row_f = _col(PARTITIONS, nl.float32)
                nisa.tensor_copy(dst=row_f, src=pos_t)
                nisa.tensor_scalar(dst=minus, data=row_f, op0=nl.add, operand0=1.0,
                                   op1=nl.multiply, operand1=-1.0)
            else:
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
            if rotate:
                # ``q @ H`` with the head dimension contracted on the partitions, as
                # ``dsa_hadamard128`` contracts it; scaled and rounded once on the way out.
                rot_ps = nl.ndarray((head_dim, heads), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=rot_ps, stationary=signs, moving=q_t)
                q_t = _sb((head_dim, heads), q_hbm.dtype)
                nisa.activation(dst=q_t, op=nl.copy, data=rot_ps, scale=HADAMARD_SCALE)
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

            # ---- candidate-major out, per group of tiles: tiles onto the partitions,
            # rows stored whole, the ragged last tile by the group that holds it --------
            for o0 in range(0, n_tiles, OUTPUT_GROUP_TILES):
                on = min(OUTPUT_GROUP_TILES, n_tiles - o0)
                o_full = min(on, full - o0)
                o_ps = nl.ndarray((on, PARTITIONS), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=o_ps, data=scores[:, o0:o0 + on])
                o_sb = _sb((on, PARTITIONS), nl.float32)
                nisa.tensor_copy(dst=o_sb, src=o_ps)
                if o_full > 0:
                    nisa.dma_copy(dst=out.ap(pattern=[[PARTITIONS, o_full], [1, PARTITIONS]],
                                             offset=row * candidates + o0 * PARTITIONS),
                                  src=o_sb[0:o_full, :])
                if o_full < on:
                    nisa.dma_copy(dst=out.ap(pattern=[[rem, 1], [1, rem]],
                                             offset=row * candidates + full * PARTITIONS),
                                  src=o_sb[o_full:o_full + 1, 0:rem])
    return out


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def _ceiling_problem(candidates: int) -> str | None:
    """Why this kernel cannot take ``candidates``, or ``None``."""
    if int(candidates) <= MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES:
        return None
    return (f"{int(candidates)} candidates are above MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES="
            f"{MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES} of the T-row scores kernel ({__file__}: "
            f"two output groups; its score and key tiles fill SBUF above them); lower "
            f"max_model_len or raise decode_context_parallel_size")


def dsa_decode_scores_rows_wide(query: Tensor, weights: Tensor, pool_bank: Tensor,
                                slots: Tensor, position: Tensor, pooled: Tensor, *,
                                candidates: int, pool_size: int,
                                rotate_query: bool = False) -> Tensor:
    """:func:`decode_trow.dsa_decode_scores_rows` up to
    :data:`MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES` candidates: the same arguments, checks,
    torch fallback (the old oracle), launch grid and result.

    Raises:
        DecodeTrowError: an operand :func:`decode_trow.dsa_decode_scores_rows` refuses, or
            ``candidates`` above :data:`MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES`.
    """
    batch, rows = _require(query, weights, pool_bank, slots, position, pooled, candidates,
                           pool_size, rotate_query)
    problem = _ceiling_problem(candidates)
    if problem is not None:
        raise DecodeTrowError(problem)
    usable = (can_run_kernel(query) and query.dtype == torch.bfloat16
              and pool_bank.dtype == torch.bfloat16 and pooled.dtype == torch.bfloat16)
    if not usable:
        _count_torch_fallback()
        return dsa_decode_scores_rows_torch_oracle(
            query, weights, pool_bank, slots, position, pooled, candidates=candidates,
            pool_size=pool_size, rotate_query=rotate_query)
    programs = _programs(batch)
    _count_dispatch(programs, batch, rows)
    call = wrap_nki(dsa_decode_scores_rows_wide_kernel)
    if programs == 2:
        call = call[2]
    positions = int(position.shape[0])
    return call(query.contiguous(), weights.to(torch.float32).contiguous(),
                pool_bank.contiguous(), slots.reshape(batch, 1).to(torch.int32).contiguous(),
                position.reshape(positions, 1).to(torch.int32).contiguous(), pooled.contiguous(),
                int(candidates), int(pool_size), rows, bool(rotate_query), SOURCE_DIGEST)


def decode_scores_rows_for(candidates: int):
    """The ``T``-row scores host function for a static candidate axis:
    :func:`decode_trow.dsa_decode_scores_rows` up to
    :data:`MAX_DECODE_SCORES_ROWS_CANDIDATES`, :func:`dsa_decode_scores_rows_wide` above.

    The choice is a Python test on an int, so it adds no graph node.
    """
    if int(candidates) <= MAX_DECODE_SCORES_ROWS_CANDIDATES:
        return dsa_decode_scores_rows
    return dsa_decode_scores_rows_wide


def check_decode_scores_rows_context(rank_context: int, pool_size: int | None,
                                     speculative_tokens: int | None) -> None:
    """Refuse a drafting form whose per-rank candidate axis is past this kernel's.

    A server that drafts ``k >= 1`` tokens runs verify steps of ``1 + k`` rows, which the
    ``T``-row kernels score; without drafts the one-token kernel
    (:mod:`decode_batch`) scores and has no such ceiling, and without an indexer
    (``pool_size`` ``None``) nothing scores. ``rank_context`` is the tokens one rank's
    indexer covers (``max_model_len``, or a DCP rank's share).

    Raises:
        DecodeTrowError: ``rank_context // pool_size`` above
            :data:`MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES`, naming both and the fix.
    """
    if pool_size is None or not speculative_tokens or int(speculative_tokens) < 1:
        return
    candidates = int(rank_context) // int(pool_size)
    problem = _ceiling_problem(candidates)
    if problem is not None:
        raise DecodeTrowError(
            f"a rank context of {int(rank_context)} tokens over pools of {int(pool_size)} "
            f"gives {candidates} candidates per verify row: {problem}")
