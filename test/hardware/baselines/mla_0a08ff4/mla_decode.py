# SPDX-License-Identifier: Apache-2.0
"""Batched MLA decode attention over a paged latent bank: dense or selected rows.

One query row per request. Request ``b`` owns the window its block-table row
names (``pages * page_size`` rows of the bank, in table order), and this step's
own latent row ``written[b]`` sits at window row ``position[b]``. Two modes:

* **dense** -- attend window rows ``0 .. position[b]``. This is the DSA layer's
  short-context regime: while ``seq_len <= index_topk + index_kpool - 1`` the
  indexer's selection keeps every token, so attending the causal prefix directly
  gives the same row set without scoring or selecting anything.
* **selected** -- attend the window rows ``topk_indices[b]`` names; ``-1`` is a
  column that carries no token. Duplicates count once per column, as the sparse
  kernel counts them.

Either way ``written[b]`` stands in for the bank row at ``position[b]``, so the
result does not depend on whether this step's cache write has landed.

What changes against ``mla_sparse``: no window is staged. Dense rows are read
page by page straight from the bank and selected rows are gathered from the bank
by translating each window row through the request's own block-table row, so the
cost follows the rows attended, and each request reads its own table. The query
and the cache are 2-byte floats, so MM1 runs single-pass (exact products, fp32
accumulation). MM2 runs as two single-pass matmuls on a bf16 hi/lo split of the
fp32 probabilities, which carries about 16 significand bits.

Both physical cores of an LNC2 core split the requests (``[2]`` grid). A single
request runs on one: splitting one request's keys would need a cross-core
softmax merge.
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

from vllm_neuron.utils.neuron_utils import values_are_readable

logger = logging.getLogger(__name__)

#: Partition extent: the latent is tiled to it for MM1 and the keys for MM2.
LATENT_TILE = 128
KEY_CHUNK = 128
#: Moving free-axis extent of one matmul: MM1 scores 512 keys per instruction.
MOVING_MAX = 512
#: Stationary free-axis extent: the head count rides it in both matmuls.
HEAD_MAX = 128
#: The selector's "no token" column, the only negative index admitted.
SENTINEL_INDEX = -1
#: The score a column that carries no token is given, by predicated copy rather than
#: by adding a bias, so whatever the bank holds in that row cannot reach the softmax:
#: ``exp(softmax_scale * (MASKED - max))`` is exactly 0.0 in fp32 once the max is at
#: least :data:`_MAX_FLOOR`.
MASKED_SCORE = -1.0e30
#: The softmax max is floored here. A request whose every column is masked would
#: otherwise take MASKED_SCORE as its max and give each masked column exp(0) = 1; with
#: the floor they stay exp(-huge) = 0. No real score comes near it.
_MAX_FLOOR = -1.0e20
#: The denominator floor: a request that attends nothing returns exact zeros.
_TOTAL_FLOOR = 1.0e-30
#: fp32 columns of a [P, 1] scratch tile: one whole 32-byte line, as ``mla_sparse``
#: declares its scalars so the allocator keeps later tiles on the line.
_LINE = 8


#: This file's content digest, handed to both kernels as a trace-time int. The
#: compiled-kernel cache keys on a kernel's own source and its arguments, and both
#: entry points below are thin wrappers over :func:`_decode_body`: without this, an
#: edit to the body or its helpers would be served the previous build.
SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)


class MlaDecodeAttentionError(ValueError):
    """Raised for a call this kernel does not serve; there is no torch fallback."""


@dataclass
class _MlaDecodeDispatchCounters:
    nki_dispatch: int = 0
    dense_dispatch: int = 0
    selected_dispatch: int = 0
    two_program_dispatch: int = 0


_MLA_DECODE_COUNTERS = _MlaDecodeDispatchCounters()


def reset_mla_decode_dispatch_counters() -> None:
    """Zero every count this module keeps."""
    _MLA_DECODE_COUNTERS.nki_dispatch = 0
    _MLA_DECODE_COUNTERS.dense_dispatch = 0
    _MLA_DECODE_COUNTERS.selected_dispatch = 0
    _MLA_DECODE_COUNTERS.two_program_dispatch = 0


def mla_decode_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)``, the family form every kernel seam reports.

    ``torch_fallback`` is always 0: a call this kernel does not serve is refused.
    """
    return (_MLA_DECODE_COUNTERS.nki_dispatch, 0)


def mla_decode_route_counts() -> tuple[int, int, int]:
    """``(dense_dispatch, selected_dispatch, two_program_dispatch)`` since the last reset."""
    return (
        _MLA_DECODE_COUNTERS.dense_dispatch,
        _MLA_DECODE_COUNTERS.selected_dispatch,
        _MLA_DECODE_COUNTERS.two_program_dispatch,
    )


@torch._dynamo.assume_constant_result
def _count_dispatch(dense: bool, programs: int) -> None:
    _MLA_DECODE_COUNTERS.nki_dispatch += 1
    if dense:
        _MLA_DECODE_COUNTERS.dense_dispatch += 1
    else:
        _MLA_DECODE_COUNTERS.selected_dispatch += 1
    if programs == 2:
        _MLA_DECODE_COUNTERS.two_program_dispatch += 1


# Helpers take positional arguments only: the NKI front end drops keyword defaults.
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


def _load_dense_rows(c_rows, bank_hbm, table_hbm, b, page_size):
    """Window rows of request ``b`` into ``c_rows[:, chunk, :]``, page piece by piece."""
    latent = bank_hbm.shape[1]
    pages = table_hbm.shape[1]
    span = page_size if page_size < KEY_CHUNK else KEY_CHUNK
    pieces = page_size // span
    n_pieces = pages * pieces
    banked = bank_hbm.reshape((bank_hbm.shape[0] // span, span, latent))
    # The table row once, the -1 pad clamped onto page 0 (its rows are masked).
    raw = _sb((1, n_pieces), nl.int32)
    nisa.dma_copy(dst=raw.ap(pattern=[[n_pieces, 1], [pieces, pages], [1, 1]]),
                  src=table_hbm.ap(pattern=[[pages, 1], [1, pages], [1, 1]],
                                   offset=b * pages))
    held = _sb((1, n_pieces), nl.int32)
    nisa.tensor_scalar(dst=held, data=raw.ap(pattern=[[n_pieces, 1], [pieces, pages],
                                                      [0, pieces]]),
                       op0=nl.maximum, operand0=0, op1=nl.multiply, operand1=pieces)
    if pieces > 1:
        offs = _sb((1, n_pieces), nl.int32)
        nisa.iota(dst=offs, pattern=[[0, pages], [1, pieces]], offset=0)
        nisa.tensor_tensor(dst=held, data1=held, data2=offs, op=nl.add)
    for piece in range(n_pieces):
        row0 = piece * span
        chunk = row0 // KEY_CHUNK
        part = row0 - chunk * KEY_CHUNK
        nisa.dma_copy(
            dst=c_rows[part:part + span, chunk, :],
            src=banked.ap(pattern=[[latent, span], [1, latent]], offset=0,
                          scalar_offset=held[:, piece:piece + 1], indirect_dim=0),
        )


def _load_selected_rows(c_rows, bank_hbm, table_hbm, topk_hbm, b, page_size):
    """Gather request ``b``'s selected window rows from the bank, via its table row."""
    latent = bank_hbm.shape[1]
    pages = table_hbm.shape[1]
    width = topk_hbm.shape[1]
    n_chunks = width // KEY_CHUNK
    shift = _log2(page_size)
    # idx[r, j] = topk[b, j * 128 + r]: one index per partition per chunk.
    idx = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.dma_copy(dst=idx, src=topk_hbm.ap(pattern=[[1, KEY_CHUNK], [KEY_CHUNK, n_chunks]],
                                           offset=b * width))
    safe = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.tensor_scalar(dst=safe, data=idx, op0=nl.maximum, operand0=0)
    page_no = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.tensor_scalar(dst=page_no, data=safe, op0=nl.right_shift, operand0=shift)
    in_page = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.tensor_scalar(dst=in_page, data=safe, op0=nl.bitwise_and, operand0=page_size - 1)
    # The table viewed one entry per row, so an indirect row index is a flat entry index.
    entries = table_hbm.reshape((table_hbm.shape[0] * pages, 1))
    page_id = _sb((KEY_CHUNK, n_chunks), nl.int32)
    for j in range(n_chunks):
        nisa.dma_copy(dst=page_id[:, j:j + 1],
                      src=entries.ap(pattern=[[1, KEY_CHUNK], [1, 1]], offset=b * pages,
                                     vector_offset=page_no[:, j:j + 1], indirect_dim=0))
    row = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.tensor_scalar(dst=row, data=page_id, op0=nl.maximum, operand0=0,
                       op1=nl.multiply, operand1=page_size)
    nisa.tensor_tensor(dst=row, data1=row, data2=in_page, op=nl.add)
    for j in range(n_chunks):
        nisa.dma_copy(dst=c_rows[:, j, :],
                      src=bank_hbm.ap(pattern=[[latent, KEY_CHUNK], [1, latent]],
                                      vector_offset=row[:, j:j + 1], indirect_dim=0))


def _decode_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                 softmax_scale, page_size, dense, out_hbm):
    batch, heads, latent = q_hbm.shape
    pages = table_hbm.shape[1]
    n_lat = latent // LATENT_TILE
    if dense:
        width = pages * page_size
    else:
        width = topk_hbm.shape[1]
    n_chunks = width // KEY_CHUNK
    kv_dtype = bank_hbm.dtype

    if dense:
        rows_f = _sb((heads, width), nl.float32)
        nisa.iota(dst=rows_f, pattern=[[1, width]], offset=0)

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for b in nl.affine_range(prg, batch, n_prgs):
        # ---- this request's position, query and own row --------------------------
        pos_i = _col(heads, nl.int32)
        nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[0, heads], [1, 1]], offset=b))
        pos_f = _col(heads, nl.float32)
        nisa.tensor_copy(dst=pos_f, src=pos_i)

        q_nat = _sb((heads, latent), q_hbm.dtype)
        nisa.dma_copy(dst=q_nat, src=q_hbm.ap(pattern=[[latent, heads], [1, latent]],
                                              offset=b * heads * latent))
        # A PE transpose writes PSUM in its input's dtype, and a PSUM write must be
        # whole 4-byte words: one head of a 2-byte dtype is a 2-byte column. So the
        # query is transposed in fp32 (exact on the PE) and rounded back, losslessly,
        # to its own dtype.
        q_f = _sb((heads, latent), nl.float32)
        nisa.tensor_copy(dst=q_f, src=q_nat)
        q_t_ps = nl.ndarray((LATENT_TILE, n_lat * heads), dtype=nl.float32,
                            buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_transpose(dst=q_t_ps[:, li * heads:(li + 1) * heads],
                              data=q_f[:, li * LATENT_TILE:(li + 1) * LATENT_TILE])
        q_t = _sb((LATENT_TILE, n_lat * heads), q_hbm.dtype)
        nisa.tensor_copy(dst=q_t, src=q_t_ps)

        own = _sb((heads, latent), written_hbm.dtype)
        nisa.dma_copy(dst=own, src=written_hbm.ap(pattern=[[0, heads], [1, latent]],
                                                  offset=b * latent))
        own_f = _sb((heads, latent), nl.float32)
        nisa.tensor_copy(dst=own_f, src=own)
        prod = _sb((heads, latent), nl.float32)
        nisa.tensor_tensor(dst=prod, data1=q_nat, data2=own, op=nl.multiply)
        s_self = _col(heads, nl.float32)
        nisa.tensor_reduce(dst=s_self, op=nl.add, data=prod, axis=1)

        # ---- the rows, rows on partitions ---------------------------------------
        c_rows = _sb((KEY_CHUNK, n_chunks, latent), kv_dtype)
        if dense:
            _load_dense_rows(c_rows, bank_hbm, table_hbm, b, page_size)
        else:
            _load_selected_rows(c_rows, bank_hbm, table_hbm, topk_hbm, b, page_size)

        # ---- which columns carry a token, and how often this step's own row counts --
        valid = _sb((heads, width), nl.float32)
        count = _col(heads, nl.float32)
        if dense:
            nisa.tensor_scalar(dst=valid, data=rows_f, op0=nl.less, operand0=pos_f)
            nisa.memset(dst=count, value=1.0)
        else:
            idx_i = _sb((heads, width), nl.int32)
            nisa.dma_copy(dst=idx_i, src=topk_hbm.ap(pattern=[[0, heads], [1, width]],
                                                     offset=b * width))
            idx_f = _sb((heads, width), nl.float32)
            nisa.tensor_copy(dst=idx_f, src=idx_i)
            live = _sb((heads, width), nl.float32)
            nisa.tensor_scalar(dst=live, data=idx_f, op0=nl.greater,
                               operand0=float(SENTINEL_INDEX))
            other = _sb((heads, width), nl.float32)
            nisa.tensor_scalar(dst=other, data=idx_f, op0=nl.not_equal, operand0=pos_f)
            nisa.tensor_tensor(dst=valid, data1=live, data2=other, op=nl.multiply)
            is_own = _sb((heads, width), nl.float32)
            nisa.tensor_scalar(dst=is_own, data=idx_f, op0=nl.equal, operand0=pos_f)
            nisa.tensor_reduce(dst=count, op=nl.add, data=is_own, axis=1)
        pred = _sb((heads, width), nl.uint8)
        nisa.tensor_copy(dst=pred, src=valid)

        # ---- MM1: scores[H, width] = q . c, keys transposed onto the latent axis ----
        # Only columns that carry a token are copied out of PSUM; the rest keep
        # MASKED_SCORE, so a stale or never-written row in the window changes nothing.
        scores = _sb((heads, width), nl.float32)
        nisa.memset(dst=scores, value=MASKED_SCORE)
        for t0 in range(0, width, MOVING_MAX):
            tw = min(MOVING_MAX, width - t0)
            c_t = _sb((LATENT_TILE, n_lat, MOVING_MAX), kv_dtype)
            for li in range(n_lat):
                t_ps = nl.ndarray((LATENT_TILE, MOVING_MAX), dtype=kv_dtype, buffer=nl.psum)
                for ck in range(tw // KEY_CHUNK):
                    nisa.nc_transpose(
                        dst=t_ps[:, ck * KEY_CHUNK:(ck + 1) * KEY_CHUNK],
                        data=c_rows[:, t0 // KEY_CHUNK + ck,
                                    li * LATENT_TILE:(li + 1) * LATENT_TILE])
                nisa.tensor_copy(dst=c_t[:, li, 0:tw], src=t_ps[:, 0:tw])
            s_ps = nl.ndarray((heads, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
            for li in range(n_lat):
                nisa.nc_matmul(dst=s_ps[:, 0:tw],
                               stationary=q_t[:, li * heads:(li + 1) * heads],
                               moving=c_t[:, li, 0:tw], accumulate=(li > 0))
            nisa.tensor_copy_predicated(dst=scores[:, t0:t0 + tw], src=s_ps[:, 0:tw],
                                        predicate=pred[:, t0:t0 + tw])

        # ---- one softmax over every column plus this step's own row --------------
        # s_own = s_self where this step's row is attended, else MASKED_SCORE.
        own_on = _col(heads, nl.float32)
        nisa.tensor_scalar(dst=own_on, data=count, op0=nl.minimum, operand0=1.0)
        own_off = _col(heads, nl.float32)
        nisa.tensor_scalar(dst=own_off, data=own_on, op0=nl.subtract, operand0=1.0,
                           op1=nl.multiply, operand1=-MASKED_SCORE)
        s_own = _col(heads, nl.float32)
        nisa.tensor_tensor(dst=s_own, data1=s_self, data2=own_on, op=nl.multiply)
        nisa.tensor_tensor(dst=s_own, data1=s_own, data2=own_off, op=nl.add)
        row_max = _col(heads, nl.float32)
        nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=scores, axis=1)
        top = _col(heads, nl.float32)
        nisa.tensor_tensor(dst=top, data1=row_max, data2=s_own, op=nl.maximum)
        nisa.tensor_scalar(dst=top, data=top, op0=nl.maximum, operand0=_MAX_FLOOR)
        neg_top = _col(heads, nl.float32)
        nisa.tensor_scalar(dst=neg_top, data=top, op0=nl.multiply, operand0=-softmax_scale)
        p = _sb((heads, width), nl.float32)
        col_sum = _col(heads, nl.float32)
        nisa.activation(dst=p, op=nl.exp, data=scores, bias=neg_top, scale=softmax_scale,
                        reduce_op=nl.add, reduce_res=col_sum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        e_own = _col(heads, nl.float32)
        nisa.activation(dst=e_own, op=nl.exp, data=s_own, bias=neg_top, scale=softmax_scale)
        p_own = _col(heads, nl.float32)
        nisa.tensor_tensor(dst=p_own, data1=e_own, data2=count, op=nl.multiply)
        total = _col(heads, nl.float32)
        nisa.tensor_tensor(dst=total, data1=col_sum, data2=p_own, op=nl.add)
        nisa.tensor_scalar(dst=total, data=total, op0=nl.maximum, operand0=_TOTAL_FLOOR)

        # ---- MM2: out[H, L] = p . c over a bf16 hi/lo split of p -----------------
        # p is transposed in fp32 (the same 4-byte rule as the query), then split in
        # the transposed layout: hi = bf16(p), lo = bf16(p - hi).
        p_t = _sb((KEY_CHUNK, n_chunks, heads), nl.float32)
        for g0 in range(0, n_chunks, 4):
            gn = min(4, n_chunks - g0)
            pt_ps = nl.ndarray((KEY_CHUNK, 4 * heads), dtype=nl.float32, buffer=nl.psum)
            for ck in range(gn):
                c0 = (g0 + ck) * KEY_CHUNK
                nisa.nc_transpose(dst=pt_ps[:, ck * heads:(ck + 1) * heads],
                                  data=p[:, c0:c0 + KEY_CHUNK])
            nisa.tensor_copy(dst=p_t[:, g0:g0 + gn, :], src=pt_ps[:, 0:gn * heads])
        p_hi = _sb((KEY_CHUNK, n_chunks, heads), nl.bfloat16)
        nisa.tensor_copy(dst=p_hi, src=p_t)
        p_hi_f = _sb((KEY_CHUNK, n_chunks, heads), nl.float32)
        nisa.tensor_copy(dst=p_hi_f, src=p_hi)
        p_lo = _sb((KEY_CHUNK, n_chunks, heads), nl.bfloat16)
        nisa.tensor_tensor(dst=p_lo, data1=p_t, data2=p_hi_f, op=nl.subtract)
        halves = (p_hi, p_lo)
        pv_ps = nl.ndarray((heads, latent), dtype=nl.float32, buffer=nl.psum)
        for ck in range(n_chunks):
            for half in range(2):
                nisa.nc_matmul(dst=pv_ps, stationary=halves[half][:, ck, :],
                               moving=c_rows[:, ck, :],
                               accumulate=(ck > 0 or half > 0))

        # ---- normalise, add this step's own row, store ----------------------------
        num = _sb((heads, latent), nl.float32)
        nisa.scalar_tensor_tensor(dst=num, data=own_f, op0=nl.multiply, operand0=p_own,
                                  op1=nl.add, operand1=pv_ps)
        recip = _col(heads, nl.float32)
        nisa.reciprocal(dst=recip, data=total)
        res = _sb((heads, latent), nl.float32)
        nisa.tensor_scalar(dst=res, data=num, op0=nl.multiply, operand0=recip)
        nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                                     offset=b * heads * latent),
                      src=res)


@nki.jit
def mla_decode_dense_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                            softmax_scale, page_size, source_digest):
    """Dense decode attention: window rows ``0 .. position[b]`` of each request.

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    out = nl.ndarray((batch, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _decode_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, None,
                 softmax_scale, page_size, True, out)
    return out


@nki.jit
def mla_decode_selected_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                               topk_hbm, softmax_scale, page_size, source_digest):
    """Selected decode attention: the window rows ``topk[b]`` names, ``-1`` masked.

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    out = nl.ndarray((batch, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _decode_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                 softmax_scale, page_size, False, out)
    return out


def _programs(batch: int) -> int:
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and batch >= 2:
        return 2
    return 1


def _require(q_lift, bank, block_table, position, written, page_size, topk_indices,
             softmax_scale) -> None:
    if q_lift.ndim != 3 or bank.ndim != 2:
        raise MlaDecodeAttentionError(
            f"q_lift must be [batch, heads, latent] and bank [slots, latent]; got "
            f"{tuple(q_lift.shape)} and {tuple(bank.shape)}"
        )
    batch, heads, latent = (int(d) for d in q_lift.shape)
    if int(bank.shape[1]) != latent:
        raise MlaDecodeAttentionError(
            f"q_lift and bank must share the latent rank; got {latent} and {int(bank.shape[1])}"
        )
    if latent % LATENT_TILE or latent > MOVING_MAX:
        raise MlaDecodeAttentionError(
            f"the latent rank must be whole {LATENT_TILE}-row tiles and fit one {MOVING_MAX}"
            f"-wide MM2 tile; got {latent}"
        )
    if not 1 <= heads <= HEAD_MAX:
        raise MlaDecodeAttentionError(f"heads must lie in [1, {HEAD_MAX}]; got {heads}")
    if q_lift.dtype not in (torch.bfloat16, torch.float16) or bank.dtype != q_lift.dtype:
        raise MlaDecodeAttentionError(
            f"q_lift and bank must be one 2-byte float dtype (single-pass MM1); got "
            f"{q_lift.dtype} and {bank.dtype}"
        )
    if written.dtype != bank.dtype or tuple(written.shape) != (batch, latent):
        raise MlaDecodeAttentionError(
            f"written must be [batch, latent] = {(batch, latent)} in the bank's dtype; got "
            f"{tuple(written.shape)} {written.dtype}"
        )
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
        raise MlaDecodeAttentionError(
            f"block_table must be [batch, pages] with batch={batch}; got "
            f"{tuple(block_table.shape)}"
        )
    if position.ndim != 1 or int(position.shape[0]) != batch:
        raise MlaDecodeAttentionError(
            f"position must be [batch] = [{batch}]; got {tuple(position.shape)}"
        )
    page = int(page_size)
    if page < 1 or (page % KEY_CHUNK and KEY_CHUNK % page):
        raise MlaDecodeAttentionError(
            f"page_size must be a multiple or a divisor of {KEY_CHUNK}; got {page_size}"
        )
    if int(bank.shape[0]) % page:
        raise MlaDecodeAttentionError(
            f"the bank's {int(bank.shape[0])} rows must be whole pages of {page}"
        )
    window = int(block_table.shape[1]) * page
    if topk_indices is None:
        if window % KEY_CHUNK:
            raise MlaDecodeAttentionError(
                f"a dense window must be whole {KEY_CHUNK}-row chunks; got {window}"
            )
    else:
        if topk_indices.ndim != 2 or int(topk_indices.shape[0]) != batch:
            raise MlaDecodeAttentionError(
                f"topk_indices must be [batch, width]; got {tuple(topk_indices.shape)}"
            )
        if int(topk_indices.shape[1]) % KEY_CHUNK or int(topk_indices.shape[1]) < 1:
            raise MlaDecodeAttentionError(
                f"the selected width must be a positive multiple of {KEY_CHUNK}; got "
                f"{int(topk_indices.shape[1])}"
            )
        if page & (page - 1):
            raise MlaDecodeAttentionError(
                f"selected rows are translated through the table with a shift, so "
                f"page_size must be a power of two; got {page}"
            )
        if values_are_readable(topk_indices):
            lo, hi = int(topk_indices.min()), int(topk_indices.max())
            if lo < SENTINEL_INDEX or hi >= window:
                raise MlaDecodeAttentionError(
                    f"every selected row must index the window or be {SENTINEL_INDEX}; "
                    f"got [{lo}, {hi}] against a window of {window}"
                )
    if values_are_readable(position):
        lo, hi = int(position.min()), int(position.max())
        if lo < 0 or hi >= window:
            raise MlaDecodeAttentionError(
                f"each position must lie inside its window; got [{lo}, {hi}] against "
                f"{window}"
            )
    if not softmax_scale > 0:
        raise MlaDecodeAttentionError(f"softmax_scale must be positive; got {softmax_scale}")


def mla_decode_attention(q_lift: Tensor, bank: Tensor, block_table: Tensor,
                         position: Tensor, written: Tensor, softmax_scale: float,
                         page_size: int, topk_indices: Tensor | None = None) -> Tensor:
    """Decode attention for ``B`` requests, one query row each. ``[B, H, L]`` float32.

    Args:
        q_lift: ``[B, H, L]`` bf16/fp16, the absorbed query.
        bank: ``[slots, L]`` the whole latent bank, same dtype.
        block_table: ``[B, pages]`` int, each request's pages (``-1`` pads).
        position: ``[B]`` int, the window row of this step's own token.
        written: ``[B, L]`` this step's own latent rows.
        softmax_scale: positive.
        page_size: rows per page, a trace-time int.
        topk_indices: ``[B, K]`` int window rows, ``-1`` for "no token", or None
            for the dense causal prefix ``0 .. position``.
    """
    _require(q_lift, bank, block_table, position, written, page_size, topk_indices,
             float(softmax_scale))
    batch = int(q_lift.shape[0])
    programs = _programs(batch)
    dense = topk_indices is None
    _count_dispatch(dense, programs)
    entry = mla_decode_dense_kernel if dense else mla_decode_selected_kernel
    call = wrap_nki(entry)
    if programs == 2:
        call = call[2]
    args = (q_lift.contiguous(), bank.contiguous(),
            block_table.contiguous().to(torch.int32),
            position.contiguous().to(torch.int32), written.contiguous())
    if dense:
        return call(*args, float(softmax_scale), int(page_size), SOURCE_DIGEST)
    return call(*args, topk_indices.contiguous().to(torch.int32), float(softmax_scale),
                int(page_size), SOURCE_DIGEST)


def mla_decode_attention_torch_oracle(q_lift: Tensor, bank: Tensor, block_table: Tensor,
                                      position: Tensor, written: Tensor,
                                      softmax_scale: float, page_size: int,
                                      topk_indices: Tensor | None = None) -> Tensor:
    """CPU reference for tests: assemble each window, overlay, attend in fp32."""
    batch, heads, latent = q_lift.shape
    out = torch.empty(batch, heads, latent, dtype=torch.float32)
    page = int(page_size)
    for b in range(batch):
        table = block_table[b].to(torch.int64).clamp(min=0)
        window = bank.reshape(-1, page, latent)[table].reshape(-1, latent).to(torch.float32)
        window = window.clone()
        pos = int(position[b])
        window[pos] = written[b].to(torch.float32)
        if topk_indices is None:
            rows = torch.arange(pos + 1)
            keep = torch.ones(pos + 1, dtype=torch.bool)
        else:
            idx = topk_indices[b].to(torch.int64)
            keep = idx >= 0
            rows = idx.clamp(min=0)
        gathered = window[rows]
        scores = q_lift[b].to(torch.float32) @ gathered.t()
        scores = scores.masked_fill(~keep, float("-inf")) * softmax_scale
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
        out[b] = weights @ gathered
    return out
