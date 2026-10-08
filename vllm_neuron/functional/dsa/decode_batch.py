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

Under an LNC2 launch both physical cores run each stage (``[2]`` grid) whenever it has two
units of work. At ``B >= 2`` they split the requests. At ``B = 1`` the score stage splits
the request's candidate blocks over the two cores (``score_blocks``) from two candidate
tiles on, and the ring step runs the pool compression on one core and the token stash on
the other. A request scores its candidates ``BLOCK_TILES`` tiles at a time, so SBUF does
not bound the candidate axis. The one bound is ``FP32_EXACT_INTEGERS``: a candidate's token
positions, carried in fp32, must be exact (``candidates * pool_size <= 2**24`` tokens).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron import envs
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.decode_tail_update import TAIL_HALVES, _compress_pool_torch
from vllm_neuron.functional.dsa import kpool_hadamard as _kpool_hadamard
from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    INDEX_HEAD_DIM,
    _fwht128_blocks,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: SBUF partitions of a NeuronCore (``nl.tile_size.pmax``), the partition extent of every
#: tile in the DSA decode kernels: requests per ring-step tile, candidates per score tile.
PARTITIONS = nl.tile_size.pmax
#: bf16 ``[128, 128]`` key transposes gathered into one PSUM tile before the copy out.
TRANSPOSE_GROUP = 4
#: fp32 columns of one PSUM bank; the head scores of ``PSUM_FP32 // heads`` candidate
#: tiles share one bank, so the rectify, the weight and the head sum run once per bank.
PSUM_FP32 = 512
#: Candidate tiles (of 128) per score block: a block's keys, their transpose and its
#: output transpose (the block's tiles onto the partitions) are what a request holds at
#: once, so no candidate axis is too wide. 4096 candidates, 8 KiB of keys per partition.
BLOCK_TILES = 32
#: Uniform score blocks a program emits inline per request; past that, whole groups of
#: this many run as one device loop, so the instruction stream and the compile time stop
#: growing with the candidate axis. Four blocks inline is 16384 candidates per request at
#: B >= 2 (32768 at B = 1, split over two cores).
UNROLL_BLOCKS = 4
#: fp32 columns of a [P, 1] scratch tile: one whole 32-byte line, as ``mla_decode``
#: declares its scalars.
_LINE = 8
#: The score kernel computes candidate and token numbers in fp32, which holds every
#: integer below ``2 ** 24`` exactly: a request's candidate axis times its pool size stays
#: within it (a context of 16 Mi tokens), or a bound would compare rounded positions.
FP32_EXACT_INTEGERS = 1 << 24

#: The content digest of this file and of ``kpool_hadamard.py`` (whose butterfly the ring
#: step calls), handed to both kernels as a trace-time int: the compiled kernel cache keys
#: on a kernel's own source and its arguments, and both kernels call helpers whose edits it
#: would otherwise not see.
SOURCE_DIGEST = int(hashlib.sha256(
    Path(__file__).read_bytes() + Path(_kpool_hadamard.__file__).read_bytes()
).hexdigest()[:7], 16)


class DecodeBatchError(ValueError):
    """A malformed batched decode call: a shape, dtype or geometry this module refuses."""


@dataclass
class _DecodeBatchCounters:
    nki_dispatch: int = 0
    torch_fallback: int = 0
    ring_dispatch: int = 0
    scores_dispatch: int = 0
    two_program_dispatch: int = 0
    select_dispatch: int = 0


_COUNTERS = _DecodeBatchCounters()


def reset_decode_batch_dispatch_counters() -> None:
    """Zero every count this module keeps."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.ring_dispatch = 0
    _COUNTERS.scores_dispatch = 0
    _COUNTERS.two_program_dispatch = 0
    _COUNTERS.select_dispatch = 0


def decode_batch_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` over the batched decode stages, the family form:
    the ring step and the scores here, and the selection
    (:func:`vllm_neuron.functional.dsa.decode_select.dsa_decode_select`), which counts in
    this family."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def decode_batch_route_counts() -> tuple[int, int, int, int]:
    """``(ring_dispatch, scores_dispatch, two_program_dispatch, select_dispatch)`` since
    the last reset."""
    return (_COUNTERS.ring_dispatch, _COUNTERS.scores_dispatch,
            _COUNTERS.two_program_dispatch, _COUNTERS.select_dispatch)


@torch._dynamo.assume_constant_result
def _count_dispatch(entry: str, programs: int, batch: int) -> None:
    """Count one kernel dispatch, off the traced graph."""
    _COUNTERS.nki_dispatch += 1
    if entry == "ring":
        _COUNTERS.ring_dispatch += 1
    elif entry == "select":
        _COUNTERS.select_dispatch += 1
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
    bf16 round trips and the same butterfly arithmetic (each stage as two instructions
    over every block, :func:`kpool_hadamard._fwht128_blocks`), so the two agree bit for
    bit. The requests split evenly over the programs of the launch grid; a launch of
    fewer requests than programs (one request on both cores) splits the work instead:
    program 0 compresses the pool, program 1 stashes the token.
    """
    batch = key_hbm.shape[0]
    head_dim = key_hbm.shape[1]
    width = tail_hbm.shape[1]
    last = pool_size - 1
    pooled_hbm = nl.ndarray((batch, head_dim), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    rings_hbm = nl.ndarray((batch, width), dtype=tail_hbm.dtype, buffer=nl.shared_hbm)
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    do_pool = True
    do_stash = True
    if batch < n_prgs:
        lo = 0
        hi = batch
        do_pool = prg == 0
        do_stash = prg == 1
    else:
        share = (batch + n_prgs - 1) // n_prgs
        lo = prg * share
        hi = min(batch, lo + share)
    for r0 in range(lo, hi, PARTITIONS):
        h = min(PARTITIONS, hi - r0)
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
        if do_pool:
            _ring_pool(ring, key_bf, score_bf, ape_hbm, pooled_hbm, r0, h, head_dim,
                       pool_size, last, tail_hbm.dtype)
        if do_stash:
            _ring_stash(ring, key_bf, score_bf, pos_hbm, rings_hbm, r0, h, head_dim, width,
                        pool_size)
    return pooled_hbm, rings_hbm


def _ring_pool(ring, key_bf, score_bf, ape_hbm, pooled_hbm, r0, h, head_dim, pool_size,
               last, dtype):
    """The pool this token would close: members ``0 .. last - 1`` from the ring, this
    token as the last; compressed, rotated, scaled, to ``pooled_hbm`` rows ``r0 ..``."""
    key_f = _sb((h, head_dim), nl.float32)
    nisa.tensor_copy(dst=key_f, src=key_bf)
    score_f = _sb((h, head_dim), nl.float32)
    nisa.tensor_copy(dst=score_f, src=score_bf)
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
    pooled_bf = _sb((h, head_dim), dtype)
    nisa.tensor_copy(dst=pooled_bf, src=pooled)
    pooled_rt = _sb((h, head_dim), nl.float32)
    nisa.tensor_copy(dst=pooled_rt, src=pooled_bf)
    scratch = _sb((h, head_dim), nl.float32)
    rotated = _fwht128_blocks(pooled_rt, scratch, h, 1)
    scaled = _sb((h, head_dim), nl.float32)
    nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
    result = _sb((h, head_dim), dtype)
    nisa.tensor_copy(dst=result, src=scaled)
    nisa.dma_copy(dst=pooled_hbm.ap(pattern=[[head_dim, h], [1, head_dim]],
                                    offset=r0 * head_dim), src=result)


def _ring_stash(ring, key_bf, score_bf, pos_hbm, rings_hbm, r0, h, head_dim, width,
                pool_size):
    """This token at row ``position % pool_size`` of each half; the rings out."""
    pos_i = _col(h, nl.int32)
    nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[1, h], [1, 1]], offset=r0))
    at_i = _col(h, nl.int32)
    nisa.tensor_scalar(dst=at_i, data=pos_i, op0=nl.bitwise_and, operand0=pool_size - 1)
    at_f = _col(h, nl.float32)
    nisa.tensor_copy(dst=at_f, src=at_i)
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


# ---------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------


@nki.jit
def dsa_decode_scores_kernel(q_hbm, w_hbm, bank_hbm, slots_hbm, lens_hbm, pos_hbm,
                             pooled_hbm, candidates, pool_size, block_tiles, unroll_blocks,
                             source_digest):
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
        block_tiles: python int, candidate tiles of 128 per block (``<= 128``).
        unroll_blocks: python int, the most uniform blocks a program emits inline; past
            it whole groups of ``max(1, unroll_blocks)`` blocks run as one device loop
            (:data:`UNROLL_BLOCKS`).
        source_digest: :data:`SOURCE_DIGEST`; it only keys the kernel cache.

    Returns:
        ``[B, candidates]`` fp32: ``sum_h w[h] * relu(q[h] . k[c])`` where pool ``c`` is
        complete for the request (``(c + 1) * pool_size <= length``) and ``BOUND_FILL``
        where it is not.

    Request ``b`` reads rows ``slot[b] * rows + [0, candidates)`` of the bank and nothing
    else. Candidate ``c`` sits on partition ``c % 128`` of tile ``c // 128``, and the
    tiles go through in blocks of ``block_tiles``: a block's keys are read, transposed
    onto the head dimension, scored, bounded and written out, so neither the SBUF a
    request needs nor the output transpose (a block's tiles onto the partitions) grows
    with the candidate axis.

    Up to ``unroll_blocks`` uniform blocks (whole tiles) on a program, the walk is inline
    and runs as one pass: the first block's keys are read before the
    query work, this step's pool rides that block as one more tile carrying ``pooled[b]``
    on every partition (scored by the very instructions that score a bank tile, then
    copied into the column it stands for), and the blocks' candidate numbers are computed
    once for every request. Past it, whole groups of uniform blocks run as one device
    loop (``nl.fori_loop``) whose body walks every request of the program, each block
    placed by registers: neither the instruction stream nor the compile time grows with
    the candidate axis, and the requests of one iteration overlap as the inline walk's
    do. The uniform blocks past the last whole group and the last, shorter block stay
    inline, and this step's pool is scored on a tile of its own first.

    The programs of the launch grid split the requests; a launch of fewer requests than
    programs (one request on both cores) splits each request's blocks instead, in equal
    runs of uniform blocks, so both cores run the same device loop or none.
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
    n_blocks = (n_tiles + block_tiles - 1) // block_tiles
    per_bank = PSUM_FP32 // heads
    shift = _log2(pool_size)
    kv_dtype = bank_hbm.dtype
    one_slot = bank_hbm.shape[0] == 1
    span = block_tiles * PARTITIONS

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    # Blocks of block_tiles whole tiles are uniform; the last block (number n_uniform), if
    # any, is shorter or ends on the ragged tile. Requests b_lo, b_lo + b_step, .. on this
    # program; of each, the run of uniform blocks u_lo .. u_hi - 1, then inline the
    # uniform blocks x_lo .. n_uniform - 1 and, with takes_last, the last block.
    n_uniform = full // block_tiles
    b_lo = prg
    b_step = n_prgs
    u_lo = 0
    u_hi = n_uniform
    x_lo = n_uniform
    takes_last = n_uniform < n_blocks
    if batch < n_prgs:
        # One request on every program: equal runs, so every program runs the same device
        # loop or none (the backend needs one control flow on all cores); program 0 takes
        # the uniform blocks left over, the last program the last block.
        per = n_uniform // n_prgs
        b_lo = 0
        b_step = 1
        u_lo = prg * per
        u_hi = u_lo + per
        if prg == 0:
            x_lo = n_prgs * per
        takes_last = takes_last and prg == n_prgs - 1
    # Past unroll_blocks uniform blocks in the run, its whole groups of `group` blocks
    # run as one device loop.
    group = max(1, unroll_blocks)
    n_groups = 0
    if u_hi - u_lo > unroll_blocks:
        n_groups = (u_hi - u_lo) // group
    loop_hi = u_lo + n_groups * group
    # The inline blocks: first tile, tiles, whole tiles.
    in_t0 = []
    in_tn = []
    in_tf = []
    for blk in range(loop_hi, u_hi):
        in_t0.append(blk * block_tiles)
        in_tn.append(block_tiles)
        in_tf.append(block_tiles)
    for blk in range(x_lo, n_uniform):
        in_t0.append(blk * block_tiles)
        in_tn.append(block_tiles)
        in_tf.append(block_tiles)
    if takes_last:
        in_t0.append(n_uniform * block_tiles)
        in_tn.append(n_tiles - n_uniform * block_tiles)
        in_tf.append(full - n_uniform * block_tiles)
    n_inline = len(in_t0)
    # Without the loop, this step's pool rides the first inline block as one more tile.
    pool_tile = n_groups == 0

    # ---- request-independent: candidate numbers, the first token past each pool -------
    fill = _sb((PARTITIONS, block_tiles), nl.float32)
    nisa.memset(dst=fill, value=BOUND_FILL)
    in_cand = []
    in_end = []
    for i in range(n_inline):
        cand_f = _sb((PARTITIONS, in_tn[i]), nl.float32)
        nisa.iota(dst=cand_f, pattern=[[PARTITIONS, in_tn[i]]], offset=in_t0[i] * PARTITIONS,
                  channel_multiplier=1)
        end_f = _sb((PARTITIONS, in_tn[i]), nl.float32)
        nisa.tensor_scalar(dst=end_f, data=cand_f, op0=nl.multiply,
                           operand0=float(pool_size), op1=nl.add, operand1=float(pool_size))
        in_cand.append(cand_f)
        in_end.append(end_f)
    if n_groups > 0:
        bank_rows = bank_hbm.shape[1]
        bank_flat = bank_hbm.reshape((bank_hbm.shape[0] * bank_rows, head_dim))
        out_flat = out.reshape((batch * candidates, 1))
        # Group g's first candidate as int32 (bank row and out offset) and as fp32 on
        # every partition, and a group's candidates less its first (block u, tile t,
        # partition p: u * span + 128 t + p).
        grp_rows = _sb((1, n_groups), nl.int32)
        nisa.iota(dst=grp_rows, pattern=[[group * span, n_groups]], offset=u_lo * span,
                  channel_multiplier=0)
        grp_base = _sb((PARTITIONS, n_groups), nl.float32)
        nisa.iota(dst=grp_base, pattern=[[group * span, n_groups]], offset=u_lo * span,
                  channel_multiplier=0)
        grp_rel = _sb((PARTITIONS, group * block_tiles), nl.float32)
        nisa.iota(dst=grp_rel, pattern=[[PARTITIONS, group * block_tiles]], offset=0,
                  channel_multiplier=1)

    # What the device loop needs of each request: its number, bank rows per group, query,
    # weights, this step's pool score, the pool it closes and its length.
    lp_b = []
    lp_rows = []
    lp_q = []
    lp_w = []
    lp_virt = []
    lp_at = []
    lp_neg = []
    for b in range(b_lo, batch, b_step):
        # ---- this request's slot, then the first inline block's keys ------------------
        slot_t = _sb((1, _LINE), nl.int32)
        nisa.dma_copy(dst=slot_t[0:1, 0:1], src=slots_hbm.ap(pattern=[[1, 1], [1, 1]],
                                                             offset=b))
        if n_inline > 0:
            k_first = _block_keys(bank_hbm, pooled_hbm, slot_t, b, in_t0[0], in_tn[0],
                                  in_tf[0], pool_tile, rem, full, head_dim, kv_dtype,
                                  one_slot)

        # ---- its length and position ----------------------------------------------------
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
        if n_groups > 0:
            virt = _pool_score(pooled_hbm, q_t, w_rep, b, heads, head_dim, per_bank,
                               kv_dtype)

        # ---- the inline blocks ----------------------------------------------------------
        for i in range(n_inline):
            n_all = in_tn[i]
            if i == 0:
                k_rows = k_first
                if pool_tile:
                    n_all = in_tn[i] + 1
            else:
                k_rows = _block_keys(bank_hbm, pooled_hbm, slot_t, b, in_t0[i], in_tn[i],
                                     in_tf[i], False, rem, full, head_dim, kv_dtype,
                                     one_slot)
            scores = _score_tiles(k_rows, n_all, q_t, w_rep, heads, head_dim, per_bank,
                                  kv_dtype)
            if i == 0 and pool_tile:
                virt = scores[:, in_tn[i]:in_tn[i] + 1]
            _bound_store(scores, in_tn[i], in_tf[i], in_cand[i], in_end[i], fill, virt,
                         at_f, neg_len, out, b * candidates + in_t0[i] * PARTITIONS,
                         b * candidates + full * PARTITIONS, rem, False, 0)

        if n_groups > 0:
            row_tbl = _sb((1, n_groups), nl.int32)
            slot_rows = _sb((1, _LINE), nl.int32)
            if one_slot:
                nisa.memset(dst=slot_rows[0:1, 0:1], value=0)
            else:
                # int32 * int32 with a scalar operand: exact address arithmetic.
                nisa.tensor_scalar(dst=slot_rows[0:1, 0:1], data=slot_t[0:1, 0:1],
                                   op0=nl.multiply, operand0=bank_rows)
            # All-int32 operands: integer arithmetic on GpSimd, exact past 2**24.
            nisa.tensor_tensor(dst=row_tbl, data1=grp_rows,
                               data2=slot_rows.ap(pattern=[[_LINE, 1], [0, n_groups]]),
                               op=nl.add)
            lp_b.append(b)
            lp_rows.append(row_tbl)
            lp_q.append(q_t)
            lp_w.append(w_rep)
            lp_virt.append(virt)
            lp_at.append(at_f)
            lp_neg.append(neg_len)

    # ---- the device loop: group g of every request ------------------------------------
    if n_groups > 0:
        def uniform_group(g):
            o_cell = _sb((1, _LINE), nl.int32)
            nisa.tensor_copy(dst=o_cell[0:1, 0:1],
                             src=grp_rows.ap(pattern=[[n_groups, 1], [1, 1]],
                                             scalar_offset=g, indirect_dim=1))
            out_reg = nisa.register_alloc()
            nisa.register_load(dst=out_reg, src=o_cell[0:1, 0:1])
            base_f = _col(PARTITIONS, nl.float32)
            nisa.tensor_copy(dst=base_f,
                             src=grp_base.ap(pattern=[[n_groups, PARTITIONS], [1, 1]],
                                             scalar_offset=g, indirect_dim=1))
            cand_g = _sb((PARTITIONS, group * block_tiles), nl.float32)
            nisa.tensor_scalar(dst=cand_g, data=grp_rel, op0=nl.add, operand0=base_f)
            end_g = _sb((PARTITIONS, group * block_tiles), nl.float32)
            nisa.tensor_scalar(dst=end_g, data=cand_g, op0=nl.multiply,
                               operand0=float(pool_size), op1=nl.add,
                               operand1=float(pool_size))
            for i in range(len(lp_b)):
                cell = _sb((1, _LINE), nl.int32)
                nisa.tensor_copy(dst=cell[0:1, 0:1],
                                 src=lp_rows[i].ap(pattern=[[n_groups, 1], [1, 1]],
                                                   scalar_offset=g, indirect_dim=1))
                row_reg = nisa.register_alloc()
                nisa.register_load(dst=row_reg, src=cell[0:1, 0:1])
                for u in range(group):
                    k_rows = _sb((PARTITIONS, block_tiles, head_dim), kv_dtype)
                    nisa.dma_copy(dst=k_rows,
                                  src=bank_flat.ap(pattern=[[head_dim, PARTITIONS],
                                                            [PARTITIONS * head_dim,
                                                             block_tiles],
                                                            [1, head_dim]],
                                                   offset=u * span * head_dim,
                                                   scalar_offset=row_reg, indirect_dim=0))
                    scores = _score_tiles(k_rows, block_tiles, lp_q[i], lp_w[i], heads,
                                          head_dim, per_bank, kv_dtype)
                    c0 = u * block_tiles
                    _bound_store(scores, block_tiles, block_tiles,
                                 cand_g[:, c0:c0 + block_tiles], end_g[:, c0:c0 + block_tiles],
                                 fill, lp_virt[i], lp_at[i], lp_neg[i], out_flat,
                                 lp_b[i] * candidates + u * span, 0, rem, True, out_reg)

        nl.fori_loop(0, n_groups, uniform_group)
    return out


def _block_keys(bank_hbm, pooled_hbm, slot_t, b, t0, tn, tf, pool_tile, rem, full, head_dim,
                kv_dtype, one_slot):
    """The keys of one inline block of request ``b`` ``[128, tn (+ 1), head_dim]``: ``tf``
    whole tiles from tile ``t0``, the ragged tile (``rem`` rows, zeros below) when
    ``tf < tn``, and with ``pool_tile`` this step's pool on every partition of one more
    tile.

    One slot: the only valid slot is 0, so the reads are static. (Tracing in CPU
    simulation fills int operands with ones, and slot 1 is past this bank.)
    """
    n_all = tn
    if pool_tile:
        n_all = tn + 1
    k_rows = _sb((PARTITIONS, n_all, head_dim), kv_dtype)
    if tf > 0:
        src_ap = bank_hbm.ap(pattern=[[head_dim, PARTITIONS],
                                      [PARTITIONS * head_dim, tf], [1, head_dim]],
                             offset=t0 * PARTITIONS * head_dim)
        if not one_slot:
            src_ap = bank_hbm.ap(pattern=[[head_dim, PARTITIONS],
                                          [PARTITIONS * head_dim, tf], [1, head_dim]],
                                 offset=t0 * PARTITIONS * head_dim,
                                 scalar_offset=slot_t[0:1, 0:1], indirect_dim=0)
        nisa.dma_copy(dst=k_rows[:, 0:tf, :], src=src_ap)
    if tf < tn:
        nisa.memset(dst=k_rows[:, tf, :], value=0.0)
        src_ap = bank_hbm.ap(pattern=[[head_dim, rem], [1, head_dim]],
                             offset=full * PARTITIONS * head_dim)
        if not one_slot:
            src_ap = bank_hbm.ap(pattern=[[head_dim, rem], [1, head_dim]],
                                 offset=full * PARTITIONS * head_dim,
                                 scalar_offset=slot_t[0:1, 0:1], indirect_dim=0)
        nisa.dma_copy(dst=k_rows[0:rem, tf, :], src=src_ap)
    if pool_tile:
        nisa.dma_copy(dst=k_rows[:, tn, :],
                      src=pooled_hbm.ap(pattern=[[0, PARTITIONS], [1, head_dim]],
                                        offset=b * head_dim))
    return k_rows


def _pool_score(pooled_hbm, q_t, w_rep, b, heads, head_dim, per_bank, kv_dtype):
    """This step's pool of request ``b`` scored as one tile on every partition: the
    ``[128, 1]`` column the loop's blocks copy in where this step closes that pool."""
    v_rows = _sb((PARTITIONS, head_dim), kv_dtype)
    nisa.dma_copy(dst=v_rows, src=pooled_hbm.ap(pattern=[[0, PARTITIONS], [1, head_dim]],
                                                offset=b * head_dim))
    v_ps = nl.ndarray((head_dim, PARTITIONS), dtype=kv_dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=v_ps, data=v_rows)
    v_t = _sb((head_dim, PARTITIONS), kv_dtype)
    nisa.tensor_copy(dst=v_t, src=v_ps)
    vs_ps = nl.ndarray((PARTITIONS, per_bank, heads), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vs_ps[:, 0, :], stationary=v_t, moving=q_t, accumulate=False)
    v_prod = _sb((PARTITIONS, per_bank, heads), nl.float32)
    nisa.scalar_tensor_tensor(
        dst=v_prod[:, 0:1, :], data=vs_ps[:, 0:1, :], op0=nl.maximum, operand0=0.0,
        op1=nl.multiply,
        operand1=w_rep.ap(pattern=[[heads, PARTITIONS], [0, 1], [1, heads]]))
    virt = _col(PARTITIONS, nl.float32)
    nisa.tensor_reduce(dst=virt, op=nl.add, data=v_prod[:, 0:1, :], axis=2)
    return virt


def _score_tiles(k_rows, n_all, q_t, w_rep, heads, head_dim, per_bank, kv_dtype):
    """``[128, n_all]`` fp32: per tile of ``k_rows``, ``sum_h w[h] * relu(q[h] . k)``.
    The keys go onto the head dimension in PSUM groups of :data:`TRANSPOSE_GROUP`; the
    head scores of up to ``per_bank`` tiles share one PSUM bank, so the rectify, the
    weight and the head sum run once per bank."""
    banked = per_bank
    if banked > n_all:
        banked = n_all
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
    scores = _sb((PARTITIONS, n_all), nl.float32)
    for s0 in range(0, n_all, banked):
        sn = min(banked, n_all - s0)
        s_ps = nl.ndarray((PARTITIONS, banked, heads), dtype=nl.float32, buffer=nl.psum)
        for i in range(sn):
            t = s0 + i
            nisa.nc_matmul(dst=s_ps[:, i, :],
                           stationary=k_t[:, t * PARTITIONS:(t + 1) * PARTITIONS],
                           moving=q_t, accumulate=False)
        prod = _sb((PARTITIONS, banked, heads), nl.float32)
        nisa.scalar_tensor_tensor(
            dst=prod[:, 0:sn, :], data=s_ps[:, 0:sn, :], op0=nl.maximum, operand0=0.0,
            op1=nl.multiply,
            operand1=w_rep.ap(pattern=[[heads, PARTITIONS], [0, sn], [1, heads]]))
        nisa.tensor_reduce(dst=scores[:, s0:s0 + sn], op=nl.add, data=prod[:, 0:sn, :],
                           axis=2)
    return scores


def _bound_store(scores, tn, tf, cand_f, end_f, fill, virt, at_f, neg_len, out, o_off, r_off,
                 rem, dynamic, out_reg):
    """This step's pool at its own column, the causal bound, then the block out
    candidate-major: ``tf`` whole tiles at element ``o_off`` of ``out`` (plus ``out_reg``
    when ``dynamic``), the ragged tile's ``rem`` rows at ``r_off`` when ``tf < tn``.

    ``cand_f`` / ``end_f``: ``[128, tn]`` fp32, the block's candidate numbers and the
    first token past each of its pools."""
    stand_in = _sb((PARTITIONS, tn), nl.float32)
    nisa.tensor_scalar(dst=stand_in, data=cand_f, op0=nl.multiply, operand0=0.0,
                       op1=nl.add, operand1=virt)
    hit = _sb((PARTITIONS, tn), nl.uint8)
    nisa.tensor_scalar(dst=hit, data=cand_f, op0=nl.equal, operand0=at_f)
    nisa.tensor_copy_predicated(dst=scores[:, 0:tn], src=stand_in, predicate=hit)
    # The first token past the pool, less the length: > 0 where it is not complete.
    room = _sb((PARTITIONS, tn), nl.float32)
    nisa.tensor_scalar(dst=room, data=end_f, op0=nl.add, operand0=neg_len)
    bounded = _sb((PARTITIONS, tn), nl.uint8)
    nisa.tensor_scalar(dst=bounded, data=room, op0=nl.greater, operand0=0.0)
    nisa.tensor_copy_predicated(dst=scores[:, 0:tn], src=fill[:, 0:tn], predicate=bounded)

    # ---- candidate-major out: tiles onto the partitions, rows stored whole ------------
    o_ps = nl.ndarray((tn, PARTITIONS), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=o_ps, data=scores[:, 0:tn])
    o_sb = _sb((tn, PARTITIONS), nl.float32)
    nisa.tensor_copy(dst=o_sb, src=o_ps)
    if dynamic:
        nisa.dma_copy(dst=out.ap(pattern=[[PARTITIONS, tf], [1, PARTITIONS]], offset=o_off,
                                 scalar_offset=out_reg, indirect_dim=0),
                      src=o_sb[0:tf, :])
    elif tf > 0:
        nisa.dma_copy(dst=out.ap(pattern=[[PARTITIONS, tf], [1, PARTITIONS]], offset=o_off),
                      src=o_sb[0:tf, :])
    if tf < tn:
        nisa.dma_copy(dst=out.ap(pattern=[[rem, 1], [1, rem]], offset=r_off),
                      src=o_sb[tf:tf + 1, 0:rem])


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def _programs(batch: int, units: int) -> int:
    """Both cores of an LNC2 core when there are two units of work to split: two
    requests, or one request with two (ring step: the pool and the stash; scores: two
    candidate tiles)."""
    if envs.NEURON_LOGICAL_NC_CONFIG == 2 and (batch >= 2 or units >= 2):
        return 2
    return 1


def score_blocks(batch: int, candidates: int, programs: int) -> int:
    """Tiles per score block: :data:`BLOCK_TILES`, or fewer so that one request on two
    programs has a block for each."""
    tiles = (int(candidates) + PARTITIONS - 1) // PARTITIONS
    if int(batch) < int(programs):
        return max(1, min(BLOCK_TILES, (tiles + programs - 1) // programs))
    return min(BLOCK_TILES, tiles)


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
    programs = _programs(batch, 2)  # one request: the pool compression and the stash
    _count_dispatch("ring", programs, batch)
    width = TAIL_HALVES * pool * head_dim
    call = wrap_nki(dsa_decode_ring_step_kernel)
    if programs == 2:
        call = call[2]
    pooled, rings = call(
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
    if candidates * pool_size > FP32_EXACT_INTEGERS:
        raise DecodeBatchError(
            f"candidates * pool_size = {candidates * pool_size} tokens exceeds "
            f"{FP32_EXACT_INTEGERS}, the integers fp32 holds exactly; the kernel's token "
            f"positions would round")
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
    tiles = (int(candidates) + PARTITIONS - 1) // PARTITIONS
    programs = _programs(batch, tiles)
    block_tiles = score_blocks(batch, candidates, programs)
    _count_dispatch("scores", programs, batch)
    call = wrap_nki(dsa_decode_scores_kernel)
    if programs == 2:
        call = call[2]

    def column(value):
        return value.reshape(batch, 1).to(torch.int32).contiguous()

    return call(query.contiguous(), weights.contiguous(), pool_bank.contiguous(),
                column(slots), column(seq_lens), column(position),
                pooled.to(pool_bank.dtype).contiguous(), int(candidates), int(pool_size),
                int(block_tiles), int(UNROLL_BLOCKS), SOURCE_DIGEST)


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
