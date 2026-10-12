# SPDX-License-Identifier: Apache-2.0
"""Dense causal MLA latent attention over a paged prefill window, as an NKI kernel.

The DSA layer's prefill short regime. While every query row's causal context holds at
most ``select_k`` complete pools (``seq_len <= index_topk + index_kpool - 1``), the
indexer's top-k keeps every pool and the expansion appends the open tail, so the
selected token set of row ``i`` is exactly window rows ``0 .. seq_lens[i] - 1``. This
kernel attends that set directly::

    scores[r, j] = q_lift[r] . window[j]          for j < seq_lens[r]
    out[r]       = softmax(scores[r] * scale) @ window[:seq_lens[r]]

which is the sparse answer up to the order of the fp32 sums. It replaces the indexer's
query side, scoring, top-k, expansion and the sparse kernel's per-query gathers of the
index columns with one dense pass over the window: queries ride the matmul stationary
axis 128 rows at a time instead of one at a time.

The window is assembled by ``mla_sparse``'s own staging (the block table's pages in
order, this step's own rows overlaid at ``write_offset``), so both paths read the same
rows. Rows past a query's context are masked by predicated copy rather than by a bias,
as ``mla_decode`` does: a stale or never-written window row cannot reach the softmax.

Arithmetic, chosen to match the as-built low-precision sparse body: MM1 multiplies the
stored 2-byte query and cache (exact products) into fp32 PSUM; the softmax runs in fp32
over the whole row with its global maximum; MM2 contracts a bf16 hi/lo split of the
normalised fp32 probabilities (about 16 significand bits) against the stored cache.

Partial mode (:func:`mla_dense_window_attention_partial`) serves decode context
parallelism (DCP). The CP ranks of a group interleave the latent cache in 128-token blocks
(block ``b`` is rank ``b % CP``'s), so rank c's block table names its own pages and its
staged window is its own blocks in order. The global causal set ``t < seq_len`` is then a
prefix of that window, of length ``own_c(seq_len)``, which the caller passes per query. A
non-owner rank in a request's first block has length 0. Every head of the group (``Hq =
CP * H``) is attended, and the kernel returns the partial contract of ``dcp_merge``:
``partial [Hq, S, L]`` float32 normalised over this rank's columns, head-major, and ``lse
[Hq, S]`` float32 ``= softmax_scale * m + ln(l)``; a query with length 0 returns 0 and
``EMPTY_LSE`` exactly.

Masked mode (:func:`mla_masked_window_attention`) serves the sparse regime past the dense
one: it attends the indexer's selection without gathering it. The sparse kernel gathers
each query's selected rows one query at a time, and the gathers are serial on the
software DMA queue. Here the whole window stays resident in SBUF and the selection
becomes a predicate over the window's rows, built once per 128-query block and shared by
its heads. There is no SBUF scatter on this core, so the PE counts each query's selected
pools from one-hot rows of their ids. A row outside the selection reads a large negative
power of two in place of its score, so the softmax gives it exactly 0.0. The softmax and
MM2 keep the arithmetic of the sparse low-precision body: the same MM1 products, the
global row maximum, ``p`` normalised before a bf16 hi/lo split. Only the order of the sums
changes: the row sum and MM2 run over the window in row order rather than over the
selection in its column order. Every window row is read, selected or not, so every row
must be finite (see :func:`mla_masked_window_attention`). :func:`masked_window_serves` decides from
shapes alone whether a call is cheaper here than gathered, and whether the window fits.

This module never decides the regime and holds no model dial: the caller passes the
window and each row's causal length. The call site is
``vllm_neuron/model/glm5_next/dsa_dense_window.py``; its kill switch is
``VLLM_NEURON_MLA_DENSE_WINDOW`` (``vllm_neuron/envs.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as _ms
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest
from vllm_neuron.utils.neuron_utils import values_are_readable
# The partial contract's empty lse lives with the merge that reads it.
from vllm_neuron.functional.attention.dcp_merge import EMPTY_LSE

#: Partition extent. Query rows (queries x heads) are tiled to it for MM1 and the
#: softmax; window rows are chunked to it for MM2; the latent is tiled to it for MM1.
ROW_TILE = 128
KEY_CHUNK = 128
LATENT_TILE = 128
#: Moving free-axis extent of one matmul: MM1 scores 512 keys per instruction.
MOVING_MAX = 512
#: The widest latent one fp32 PSUM bank holds for the MM2 result row (2 KB).
LATENT_MAX = 512
#: PE transposes of 128 fp32 columns that fill one 2 KB PSUM bank.
_TRANSPOSES_PER_BANK = MOVING_MAX // ROW_TILE
#: PSUM bytes per partition of one bank, the extent of one matmul or transpose output.
_PSUM_BANK_BYTES = 2048
#: The most window rows one call stages: 20 chunks of 128. The kernel holds the window
#: in SBUF twice (rows on partitions and transposed) plus fp32 score rows of the window's
#: width: 33 bytes per window row per partition plus 6336 bytes. At 2560 rows that is
#: 90816 of the 229376 bytes of a partition, which leaves room for double buffering.
MAX_KEY_ROWS = 2560
#: The score a masked column holds: ``exp(scale * (MASKED - max))`` is exactly 0.0.
MASKED_SCORE = -1.0e30
#: Floors: a row whose every column is masked returns zeros rather than NaN.
_MAX_FLOOR = -1.0e20
_TOTAL_FLOOR = 1.0e-30
#: fp32 columns of a [P, 1] scratch tile: one whole 32-byte line.
_LINE = 8

#: This file's digest, and the staging helper's (``mla_sparse.py``), as trace-time ints:
#: the compiled-kernel cache keys on the entry's own source and arguments only.
SOURCE_DIGEST = source_digest(__file__)
STAGING_DIGEST = int(_ms.SOURCE_DIGEST)


class MlaDenseWindowError(ValueError):
    """Raised for a call this kernel does not serve. There is no fallback here."""


@dataclass
class _Counters:
    nki_dispatch: int = 0
    two_program_dispatch: int = 0
    masked_dispatch: int = 0


_COUNTERS = _Counters()


def reset_mla_dense_window_dispatch_counters() -> None:
    """Zero the dispatch counters (outside a trace; the counts are host-side)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.two_program_dispatch = 0
    _COUNTERS.masked_dispatch = 0


def mla_dense_window_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)``; the fallback count is always 0."""
    return (_COUNTERS.nki_dispatch, 0)


def mla_dense_window_route_counts() -> tuple[int, int]:
    """``(nki_dispatch, two_program_dispatch)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.two_program_dispatch)


def mla_masked_window_dispatch_count() -> int:
    """Calls of :func:`mla_masked_window_attention` since the last reset."""
    return _COUNTERS.masked_dispatch


@torch._dynamo.assume_constant_result
def _count_dispatch(programs: int) -> None:
    _COUNTERS.nki_dispatch += 1
    if programs == 2:
        _COUNTERS.two_program_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_masked_dispatch() -> None:
    _COUNTERS.masked_dispatch += 1


# Helpers take positional arguments only: the NKI front end drops keyword defaults.
def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    """A ``[parts, 1]`` view of a tile one 32-byte line wide."""
    return nl.ndarray((parts, _LINE), dtype=dtype, buffer=nl.sbuf)[:, 0:1]


def _query_columns(q_hbm, rt, q_stride, q_off, latent):
    """``rt`` query rows transposed onto the latent axis: ``[128, L / 128, 128]``, q's dtype.

    Row ``p`` is read at element ``q_off + p * q_stride`` of ``q_hbm``; column ``p`` of
    latent tile ``li`` holds it. A PE transpose writes PSUM in its input's dtype and a
    PSUM write must be whole 4-byte words, so an odd row count of a 2-byte dtype is
    transposed in fp32 (exact on the PE) and rounded back, losslessly, to its own dtype
    -- mla_decode's rule.
    """
    n_lat = latent // LATENT_TILE
    q_nat = _sb((ROW_TILE, latent), q_hbm.dtype)
    nisa.dma_copy(dst=q_nat[0:rt, :],
                  src=q_hbm.ap(pattern=[[q_stride, rt], [1, latent]], offset=q_off))
    q_f = _sb((ROW_TILE, latent), nl.float32)
    nisa.tensor_copy(dst=q_f[0:rt, :], src=q_nat[0:rt, :])
    q_ps = nl.ndarray((LATENT_TILE, n_lat * ROW_TILE), dtype=nl.float32, buffer=nl.psum)
    for li in range(n_lat):
        nisa.nc_transpose(dst=q_ps[:, li * ROW_TILE:li * ROW_TILE + rt],
                          data=q_f[0:rt, li * LATENT_TILE:(li + 1) * LATENT_TILE])
    q_t = _sb((LATENT_TILE, n_lat, ROW_TILE), q_hbm.dtype)
    for li in range(n_lat):
        nisa.tensor_copy(dst=q_t[:, li, 0:rt], src=q_ps[:, li * ROW_TILE:li * ROW_TILE + rt])
    return q_t


def _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
              key_rows, rt, q_stride, q_off, lim_off, out_off, lse_hbm, lse_off):
    """One tile of ``rt`` query rows: MM1, softmax, MM2, store.

    Row ``p`` of the tile reads its query at element ``q_off + p * q_stride`` of ``q_hbm``
    and its limit at ``lim_off + p`` of ``limits_hbm``, and writes its output at element
    ``out_off + p * L`` of ``out_hbm``. With ``lse_hbm`` (partial mode) it also writes its
    lse at ``lse_off + p``: ``ln(sum) - bias``, and exactly EMPTY_LSE when the limit is 0.
    """
    latent = c_rows.shape[2]
    n_lat = latent // LATENT_TILE
    n_chunks = c_rows.shape[1]
    width = n_chunks * KEY_CHUNK
    kv_dtype = c_rows.dtype
    q_t = _query_columns(q_hbm, rt, q_stride, q_off, latent)

    # ---- which window rows each query row may see: j < min(seq_len, key_rows) ------
    lim_i = _col(ROW_TILE, nl.int32)
    nisa.dma_copy(dst=lim_i[0:rt, :],
                  src=limits_hbm.ap(pattern=[[1, rt], [1, 1]], offset=lim_off))
    lim_f = _col(ROW_TILE, nl.float32)
    nisa.tensor_copy(dst=lim_f[0:rt, :], src=lim_i[0:rt, :])
    nisa.tensor_scalar(dst=lim_f[0:rt, :], data=lim_f[0:rt, :], op0=nl.minimum,
                       operand0=float(key_rows))
    # p_t is free until the transposes below, so it holds the 0/1 mask meanwhile.
    p_t = _sb((KEY_CHUNK, width), nl.float32)
    nisa.tensor_scalar(dst=p_t[0:rt, :], data=cols_f[0:rt, :], op0=nl.less,
                       operand0=lim_f[0:rt, :])
    pred = _sb((ROW_TILE, width), nl.uint8)
    nisa.tensor_copy(dst=pred[0:rt, :], src=p_t[0:rt, :])

    # ---- MM1: scores[rt, width] = q . window, copied out of PSUM by predicate ----------
    scores = _sb((ROW_TILE, width), nl.float32)
    nisa.memset(dst=scores[0:rt, :], value=MASKED_SCORE)
    for t0 in range(0, width, MOVING_MAX):
        tw = min(MOVING_MAX, width - t0)
        s_ps = nl.ndarray((ROW_TILE, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_matmul(dst=s_ps[0:rt, 0:tw], stationary=q_t[:, li, 0:rt],
                           moving=c_t[:, li, t0:t0 + tw], accumulate=(li > 0))
        nisa.tensor_copy_predicated(dst=scores[0:rt, t0:t0 + tw], src=s_ps[0:rt, 0:tw],
                                    predicate=pred[0:rt, t0:t0 + tw])

    # ---- softmax over the whole row, normalised before MM2 ---------------------------
    row_max = _col(ROW_TILE, nl.float32)
    nisa.tensor_reduce(dst=row_max[0:rt, :], op=nl.maximum, data=scores[0:rt, :], axis=1)
    nisa.tensor_scalar(dst=row_max[0:rt, :], data=row_max[0:rt, :], op0=nl.maximum,
                       operand0=_MAX_FLOOR)
    neg_top = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=neg_top[0:rt, :], data=row_max[0:rt, :], op0=nl.multiply,
                       operand0=-softmax_scale)
    row_sum = _col(ROW_TILE, nl.float32)
    nisa.activation(dst=scores[0:rt, :], op=nl.exp, data=scores[0:rt, :],
                    bias=neg_top[0:rt, :], scale=softmax_scale, reduce_op=nl.add,
                    reduce_res=row_sum[0:rt, :], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.tensor_scalar(dst=row_sum[0:rt, :], data=row_sum[0:rt, :], op0=nl.maximum,
                       operand0=_TOTAL_FLOOR)
    if lse_hbm is not None:
        # The floored sum: an empty row's raw sum is 0, whose log the select below would
        # turn into NaN; its floored lse is finite and the select replaces it.
        _lse_rows(lse_hbm, lse_off, row_sum, neg_top, lim_f, rt)
    recip = _col(ROW_TILE, nl.float32)
    nisa.reciprocal(dst=recip[0:rt, :], data=row_sum[0:rt, :])
    nisa.tensor_scalar(dst=scores[0:rt, :], data=scores[0:rt, :], op0=nl.multiply,
                       operand0=recip[0:rt, :])

    # ---- p onto the key partitions, one PSUM bank of chunks at a time, then hi/lo ------
    for g0 in range(0, n_chunks, _TRANSPOSES_PER_BANK):
        gn = min(_TRANSPOSES_PER_BANK, n_chunks - g0)
        pt_ps = nl.ndarray((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), dtype=nl.float32,
                           buffer=nl.psum)
        for ck in range(gn):
            c0 = (g0 + ck) * KEY_CHUNK
            nisa.nc_transpose(dst=pt_ps[:, ck * ROW_TILE:ck * ROW_TILE + rt],
                              data=scores[0:rt, c0:c0 + KEY_CHUNK])
        for ck in range(gn):
            c0 = (g0 + ck) * ROW_TILE
            nisa.tensor_copy(dst=p_t[:, c0:c0 + rt], src=pt_ps[:, ck * ROW_TILE:ck * ROW_TILE + rt])
    p_hi = _sb((KEY_CHUNK, width), kv_dtype)
    nisa.tensor_copy(dst=p_hi, src=p_t)
    # scores is free again: it holds the hi half widened, to form lo.
    nisa.tensor_copy(dst=scores, src=p_hi)
    p_lo = _sb((KEY_CHUNK, width), kv_dtype)
    nisa.tensor_tensor(dst=p_lo, data1=p_t, data2=scores, op=nl.subtract)

    # ---- MM2: out[rt, L] = p . window over both halves, one fp32 PSUM ----------------
    pv_ps = nl.ndarray((ROW_TILE, latent), dtype=nl.float32, buffer=nl.psum)
    for ck in range(n_chunks):
        c0 = ck * ROW_TILE
        nisa.nc_matmul(dst=pv_ps[0:rt, :], stationary=p_hi[:, c0:c0 + rt],
                       moving=c_rows[:, ck, :], accumulate=(ck > 0))
        nisa.nc_matmul(dst=pv_ps[0:rt, :], stationary=p_lo[:, c0:c0 + rt],
                       moving=c_rows[:, ck, :], accumulate=True)
    out_sb = _sb((ROW_TILE, latent), nl.float32)
    nisa.tensor_copy(dst=out_sb[0:rt, :], src=pv_ps[0:rt, :])
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, rt], [1, latent]], offset=out_off),
                  src=out_sb[0:rt, :])


def _lse_rows(lse_hbm, lse_off, row_sum, neg_top, lim_f, rt):
    """Store ``lse = ln(row_sum) - neg_top`` of ``rt`` rows at ``lse_off``; EMPTY_LSE at limit 0.

    ``neg_top`` is the exp's bias, ``-softmax_scale * max``, so this is ``softmax_scale * m +
    ln(l)``. ``lim_f`` is the row's limit, a non-negative whole number: ``live_m1 = min(limit,
    1) - 1`` is 0.0 or -1.0, and ``lse * (live_m1 + 1) + live_m1 * 1e30`` is exactly the lse
    or exactly EMPTY_LSE (every factor is 0, 1 or -1).
    """
    live_m1 = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=live_m1[0:rt, :], data=lim_f[0:rt, :], op0=nl.minimum, operand0=1.0,
                       op1=nl.add, operand1=-1.0, engine=nisa.engine.vector)
    ln_sum = _col(ROW_TILE, nl.float32)
    nisa.activation(dst=ln_sum[0:rt, :], op=nl.log, data=row_sum[0:rt, :])
    raw = _col(ROW_TILE, nl.float32)
    nisa.tensor_tensor(dst=raw[0:rt, :], data1=ln_sum[0:rt, :], data2=neg_top[0:rt, :],
                       op=nl.subtract)
    dead = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=dead[0:rt, :], data=live_m1[0:rt, :], op0=nl.multiply,
                       operand0=-EMPTY_LSE, engine=nisa.engine.vector)
    live = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=live[0:rt, :], data=live_m1[0:rt, :], op0=nl.add, operand0=1.0,
                       engine=nisa.engine.vector)
    lse = _col(ROW_TILE, nl.float32)
    nisa.scalar_tensor_tensor(dst=lse[0:rt, :], data=raw[0:rt, :], op0=nl.multiply,
                              operand0=live[0:rt, :], op1=nl.add, operand1=dead[0:rt, :])
    nisa.dma_copy(dst=lse_hbm.ap(pattern=[[1, rt], [1, 1]], offset=lse_off), src=lse[0:rt, :])


# Positions in the list :func:`_window_operands` returns (the kernel front end resolves
# plain names and integer subscripts, not tuple targets).
_W_ROWS = 0     # the window, keys on partitions: MM2's moving operand
_W_T = 1        # the window transposed, latent on partitions: MM1's moving operand
_W_COLS = 2     # every window row's column index, the same on every partition


def _window_operands(window_hbm, key_rows, latent):
    """The window in SBUF twice (keys on partitions, and transposed) and its column index."""
    ops = _window_rows(window_hbm, key_rows, latent, False)
    width = ops[_W_T].shape[2]
    # ---- the column index of every window row, the same on every partition -----------
    cols_f = _sb((ROW_TILE, width), nl.float32)
    nisa.iota(dst=cols_f, pattern=[[1, width]], offset=0)
    ops.append(cols_f)
    return ops


def _window_rows(window_hbm, key_rows, latent, banked):
    """The window in SBUF twice: ``[c_rows, c_t]``, keys on partitions and transposed.

    ``c_rows`` is ``[128, chunks, L]`` (row ``ck * 128 + p`` on partition ``p``, the last
    chunk's pad rows zeroed) and ``c_t`` ``[128, L / 128, chunks * 128]`` (latent tile
    ``li`` on the partitions). ``c_t`` is built by PE transposes of 128 x 128 blocks. With
    ``banked``, the transposes of as many chunks as fill one PSUM bank reach ``c_t`` in one
    copy on the scalar engine (the masked kernel, whose Vector engine binds); without it,
    each block is copied by itself on the engine the compiler picks (the dense kernels).
    """
    n_lat = latent // LATENT_TILE
    n_chunks = (key_rows + KEY_CHUNK - 1) // KEY_CHUNK
    width = n_chunks * KEY_CHUNK
    kv_dtype = window_hbm.dtype

    # ---- the window, keys on partitions, the last chunk's pad rows zeroed ------------
    c_rows = _sb((KEY_CHUNK, n_chunks, latent), kv_dtype)
    for ck in range(n_chunks):
        kc = min(KEY_CHUNK, key_rows - ck * KEY_CHUNK)
        if kc < KEY_CHUNK:
            nisa.memset(dst=c_rows[:, ck, :], value=0.0)
        nisa.dma_copy(dst=c_rows[0:kc, ck, :],
                      src=window_hbm.ap(pattern=[[latent, kc], [1, latent]],
                                        offset=ck * KEY_CHUNK * latent))

    # ---- the window again, latent on partitions: MM1's moving operand ----------------
    c_t = _sb((LATENT_TILE, n_lat, width), kv_dtype)
    if banked:
        # A 2-byte chunk transposed is L / 128 blocks of 128 x 128: 2 L bytes per partition.
        per_bank = _PSUM_BANK_BYTES // (2 * latent)
        for c0 in range(0, n_chunks, per_bank):
            nc = min(per_bank, n_chunks - c0)
            t_ps = nl.ndarray((LATENT_TILE, n_lat, per_bank * KEY_CHUNK), dtype=kv_dtype,
                              buffer=nl.psum)
            for j in range(nc):
                for li in range(n_lat):
                    nisa.nc_transpose(
                        dst=t_ps[:, li, j * KEY_CHUNK:(j + 1) * KEY_CHUNK],
                        data=c_rows[:, c0 + j, li * LATENT_TILE:(li + 1) * LATENT_TILE])
            nisa.tensor_copy(dst=c_t[:, :, c0 * KEY_CHUNK:(c0 + nc) * KEY_CHUNK],
                             src=t_ps[:, :, 0:nc * KEY_CHUNK], engine=nisa.engine.scalar)
    else:
        for ck in range(n_chunks):
            t_ps = nl.ndarray((LATENT_TILE, n_lat * KEY_CHUNK), dtype=kv_dtype,
                              buffer=nl.psum)
            for li in range(n_lat):
                nisa.nc_transpose(dst=t_ps[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK],
                                  data=c_rows[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE])
            for li in range(n_lat):
                nisa.tensor_copy(dst=c_t[:, li, ck * KEY_CHUNK:(ck + 1) * KEY_CHUNK],
                                 src=t_ps[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK])
    ops = []
    ops.append(c_rows)
    ops.append(c_t)
    return ops


def _dense_body(q_hbm, window_hbm, limits_hbm, softmax_scale, key_rows, active_rows,
                out_hbm):
    """Attend ``window_hbm`` rows ``0 .. min(limit, key_rows) - 1`` for query rows
    ``0 .. active_rows - 1``, and write zero rows from ``active_rows`` on.

    ``q_hbm`` is ``[rows, L]`` (queries x heads, row-major), ``limits_hbm`` ``[rows]``
    int32, ``out_hbm`` ``[rows, L]`` fp32. ``key_rows`` is a trace-time int no larger
    than the staged window; ``active_rows`` a trace-time int in ``1 .. rows``. The limits
    of the rows past ``active_rows`` are not read.
    """
    rows, latent = q_hbm.shape
    ops = _window_operands(window_hbm, key_rows, latent)
    c_rows = ops[_W_ROWS]
    c_t = ops[_W_T]
    cols_f = ops[_W_COLS]

    # Work is dealt round-robin over the programs: the whole tiles of active rows, then
    # the partial tile, then the zero-row chunks. ``program_id`` is a trace-time int (the
    # kernel is traced once per program), so each program keeps only its own jobs.
    n_full = active_rows // ROW_TILE
    tail = active_rows - n_full * ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for qt in nl.affine_range(prg, n_full, n_prgs):
        r0 = qt * ROW_TILE
        _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
                  key_rows, ROW_TILE, latent, r0 * latent, r0, r0 * latent, None, 0)
    job = n_full
    if tail > 0:
        if job % n_prgs == prg:
            r0 = n_full * ROW_TILE
            _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
                      key_rows, tail, latent, r0 * latent, r0, r0 * latent, None, 0)
        job += 1
    if active_rows < rows:
        zero = _sb((ROW_TILE, latent), nl.float32)
        nisa.memset(dst=zero, value=0.0)
        for z0 in range(active_rows, rows, ROW_TILE):
            if job % n_prgs == prg:
                zn = min(ROW_TILE, rows - z0)
                nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, zn], [1, latent]],
                                             offset=z0 * latent),
                              src=zero[0:zn, :])
            job += 1


@nki.jit
def mla_dense_window_kernel(q_hbm, bank_hbm, table_hbm, limits_hbm, written_hbm,
                            write_offset_hbm, softmax_scale, page_size, key_rows,
                            active_rows, source_digest=0, staging_digest=0):
    """Dense causal attention of query rows over the window a block table names.

    Args:
        q_hbm: ``[rows, L]`` bf16 or fp16, row-major query rows (queries x heads). ``L`` is
            a multiple of 128, at most :data:`LATENT_MAX`.
        bank_hbm: ``[slots, L]``, the whole latent bank, the dtype of ``q_hbm``.
        table_hbm: ``[pages, 1]`` int32, the window's pages in order (``-1`` pads).
        limits_hbm: ``[rows]`` int32, row ``r`` attends window rows ``0 .. limits[r] - 1``.
        written_hbm: ``[tokens, L]`` or None, this step's rows, overlaid on the window.
        write_offset_hbm: ``[1, 1]`` int32 or None, the window row ``written`` starts at.
        softmax_scale: python float applied to the scores before the softmax.
        page_size: python int, rows per page.
        key_rows: python int, the staged window rows (``pages * page_size``), at most
            :data:`MAX_KEY_ROWS`.
        active_rows: python int in ``1 .. rows``: rows from it on are written as zeros,
            with no compute, and their limits are not read (a padded chunk's rows).
        source_digest, staging_digest: python ints that only key the compiled-kernel cache
            on this file and on ``mla_sparse.py`` (the staging helper).

    Returns:
        ``[rows, L]`` fp32 in shared HBM. A row whose limit is 0 is zero, and so is every
        row from ``active_rows`` on.

    Under a two-program launch the work is dealt round-robin between the programs: the
    whole 128-row tiles of active rows, the partial tile, then the zero-row chunks.
    """
    rows, latent = q_hbm.shape
    out_hbm = nl.ndarray((rows, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    window_hbm = _ms._staged_window(bank_hbm, table_hbm, written_hbm, write_offset_hbm,
                                    page_size)
    _dense_body(q_hbm, window_hbm, limits_hbm, softmax_scale, key_rows, active_rows,
                out_hbm)
    return out_hbm


def _dense_partial_body(q_hbm, window_hbm, limits_hbm, softmax_scale, key_rows, active,
                        out_hbm, lse_hbm):
    """Partial mode: attend window rows ``0 .. min(limit, key_rows) - 1`` per query, every head.

    ``q_hbm`` is ``[S, Hq, L]``, ``limits_hbm`` ``[S]`` int32 (one length per query, shared by
    its heads; 0 allowed), ``out_hbm`` ``[Hq, S, L]`` and ``lse_hbm`` ``[Hq, S]`` float32.
    ``active`` is a trace-time int in ``1 .. S``: queries from it on are written as 0 with
    lse EMPTY_LSE, and their limits are not read.

    The rows are taken head by head, in tiles of 128 queries of one head, so a tile's query
    rows are one strided DMA (``Hq * L`` apart) and its output rows and lses are contiguous
    in the head-major outputs. The jobs are numbered head by head (every head's whole
    tiles, then every head's partial tile, then the zero-row chunks) and dealt round-robin
    over the programs. ``program_id`` is a trace-time int, so each program keeps its own.
    """
    seq, heads, latent = q_hbm.shape
    ops = _window_operands(window_hbm, key_rows, latent)
    c_rows = ops[_W_ROWS]
    c_t = ops[_W_T]
    cols_f = ops[_W_COLS]
    n_full = active // ROW_TILE
    tail = active - n_full * ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    q_stride = heads * latent
    for h in range(heads):
        # Job h * n_full + qt is this program's when it is prg modulo n_prgs.
        first = (prg - h * n_full) % n_prgs
        if first < n_full:
            for qt in nl.affine_range(first, n_full, n_prgs):
                s0 = qt * ROW_TILE
                _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
                          key_rows, ROW_TILE, q_stride, (s0 * heads + h) * latent, s0,
                          (h * seq + s0) * latent, lse_hbm, h * seq + s0)
    job = heads * n_full
    if tail > 0:
        s0 = n_full * ROW_TILE
        for h in range(heads):
            if job % n_prgs == prg:
                _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
                          key_rows, tail, q_stride, (s0 * heads + h) * latent, s0,
                          (h * seq + s0) * latent, lse_hbm, h * seq + s0)
            job += 1
    if active < seq:
        zero = _sb((ROW_TILE, latent), nl.float32)
        nisa.memset(dst=zero, value=0.0)
        empty = _col(ROW_TILE, nl.float32)
        nisa.memset(dst=empty, value=EMPTY_LSE)
        for h in range(heads):
            for z0 in range(active, seq, ROW_TILE):
                if job % n_prgs == prg:
                    zn = min(ROW_TILE, seq - z0)
                    nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, zn], [1, latent]],
                                                 offset=(h * seq + z0) * latent),
                                  src=zero[0:zn, :])
                    nisa.dma_copy(dst=lse_hbm.ap(pattern=[[1, zn], [1, 1]],
                                                 offset=h * seq + z0),
                                  src=empty[0:zn, :])
                job += 1


@nki.jit
def mla_dense_window_partial_kernel(q_hbm, bank_hbm, table_hbm, limits_hbm, written_hbm,
                                    write_offset_hbm, softmax_scale, page_size, key_rows,
                                    active_rows, source_digest=0, staging_digest=0):
    """Partial mode: ``(partial [Hq, S, L], lse [Hq, S])`` float32 over this rank's window.

    The operands are :func:`mla_dense_window_kernel`'s except ``q_hbm``, ``[S, Hq, L]``
    (every head of the CP group), ``limits_hbm``, ``[S]`` int32 (each query's length in
    this rank's own window, 0 allowed), and ``active_rows``, a count of queries. A query
    whose length is 0, and every query from ``active_rows`` on, returns 0 and EMPTY_LSE.
    """
    seq, heads, latent = q_hbm.shape
    out_hbm = nl.ndarray((heads, seq, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    lse_hbm = nl.ndarray((heads, seq), dtype=nl.float32, buffer=nl.shared_hbm)
    window_hbm = _ms._staged_window(bank_hbm, table_hbm, written_hbm, write_offset_hbm,
                                    page_size)
    _dense_partial_body(q_hbm, window_hbm, limits_hbm, softmax_scale, key_rows, active_rows,
                        out_hbm, lse_hbm)
    return out_hbm, lse_hbm


def dense_window_serves(q_dtype: torch.dtype, kv_dtype: torch.dtype, heads: int,
                        latent: int, key_rows: int) -> bool:
    """True when :func:`mla_dense_window_attention` serves this geometry.

    It takes plain values, so a caller can decide the regime at trace time before the
    query exists: the query and cache in the same 2-byte float, ``1 <= heads <= 128``,
    ``latent`` a whole number of 128-wide tiles no wider than :data:`LATENT_MAX`, and
    ``1 <= key_rows <= MAX_KEY_ROWS`` window rows.
    """
    try:
        _require_values(1, int(heads), int(latent), int(latent), q_dtype, kv_dtype,
                        int(key_rows))
    except MlaDenseWindowError:
        return False
    return True


def _require_geometry(q_lift: Tensor, c_kv: Tensor, key_rows: int) -> None:
    if q_lift.ndim != 3 or c_kv.ndim != 2:
        raise MlaDenseWindowError(
            f"q_lift must be [seq, heads, latent] and c_kv [slots, latent]; got "
            f"{tuple(q_lift.shape)} and {tuple(c_kv.shape)}"
        )
    seq, heads, latent = (int(d) for d in q_lift.shape)
    _require_values(seq, heads, latent, int(c_kv.shape[1]), q_lift.dtype, c_kv.dtype,
                    int(key_rows))


def _require_values(seq: int, heads: int, latent: int, cache_latent: int,
                    q_dtype: torch.dtype, kv_dtype: torch.dtype, key_rows: int) -> None:
    _require_operands(seq, heads, latent, cache_latent, q_dtype, kv_dtype)
    if key_rows < 1 or key_rows > MAX_KEY_ROWS:
        raise MlaDenseWindowError(
            f"the kernel holds the window in SBUF twice, so it reads at most "
            f"{MAX_KEY_ROWS} key rows; got key rows={key_rows}"
        )


def _require_operands(seq: int, heads: int, latent: int, cache_latent: int,
                      q_dtype: torch.dtype, kv_dtype: torch.dtype) -> None:
    """The query, cache and head checks, without the window bound: each kernel bounds its own."""
    if cache_latent != latent:
        raise MlaDenseWindowError(
            f"q_lift and c_kv must share the latent rank; got {latent} against "
            f"{cache_latent}"
        )
    if q_dtype not in (torch.bfloat16, torch.float16) or kv_dtype != q_dtype:
        raise MlaDenseWindowError(
            f"the dense window kernel feeds the PE the stored 2-byte query and cache, so "
            f"both must be the same 2-byte float; got q_lift {q_dtype} and c_kv "
            f"{kv_dtype}"
        )
    if seq < 1 or heads < 1 or heads > ROW_TILE:
        raise MlaDenseWindowError(
            f"need at least one query and 1 <= heads <= {ROW_TILE}; got seq={seq}, "
            f"heads={heads}"
        )
    if latent < LATENT_TILE or latent % LATENT_TILE != 0 or latent > LATENT_MAX:
        raise MlaDenseWindowError(
            f"the latent must be a whole number of {LATENT_TILE}-wide tiles and at most "
            f"{LATENT_MAX} (one fp32 PSUM bank per MM2 row); got latent={latent}"
        )


def _programs(rows: int) -> int:
    """Two programs on an LNC2 core when the rows span two or more 128-row tiles.

    :func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair` refuses a setting other
    than unset, 1 or 2 (``LaunchGridError``).
    """
    if lnc_pair() and rows > ROW_TILE:
        return 2
    return 1


def _checked_call(q_lift: Tensor, c_kv: Tensor, lens: Tensor, softmax_scale: float,
                  block_table_row: Tensor, written: Tensor | None,
                  write_offset: Tensor | None, page_size: int, active_rows: int | None,
                  min_len: int, name: str) -> tuple[int, int, int, int, int]:
    """Both seams' checks; returns ``(staged rows, S, heads, L, active queries)``.

    ``lens`` (called ``name`` in the messages) is each query's length in the window; an
    eager call's entries must be in ``[min_len, staged]``: 1 for the attention seam, 0 for
    the partial seam, whose non-owner ranks see no row of a request's first block.
    """
    if not softmax_scale > 0:
        raise MlaDenseWindowError(f"softmax_scale must be positive; got {softmax_scale}")
    if block_table_row is None or block_table_row.ndim != 2:
        raise MlaDenseWindowError(
            "the dense window kernel reads a paged window only: pass block_table_row "
            "[pages, 1]"
        )
    page = int(page_size)
    table = block_table_row
    try:
        staged = _ms._require_paged(c_kv, table, written, write_offset, page, None)
    except _ms.MlaSparseAttentionError as err:
        raise MlaDenseWindowError(str(err)) from err
    _require_geometry(q_lift, c_kv, staged)
    seq, heads, latent = (int(d) for d in q_lift.shape)
    if lens.ndim != 1 or int(lens.shape[0]) != seq:
        raise MlaDenseWindowError(
            f"{name} must be [seq] = [{seq}], one causal length per query; got "
            f"{tuple(lens.shape)}"
        )
    active = seq if active_rows is None else int(active_rows)
    if not 1 <= active <= seq:
        raise MlaDenseWindowError(
            f"active_rows must be in [1, {seq}], a prefix of the queries; got {active_rows!r}"
        )
    # Checked eagerly only, as the sparse seam checks its indices: a traced call has no
    # values to read, and the kernel clamps each row's length at the staged rows.
    if values_are_readable(lens):
        lo, hi = int(lens[:active].min()), int(lens[:active].max())
        if lo < min_len or hi > staged:
            raise MlaDenseWindowError(
                f"every {name} entry must be in [{min_len}, {staged}], the window rows this "
                f"call stages; got the range [{lo}, {hi}]"
            )
    return staged, seq, heads, latent, active


def mla_dense_window_attention(q_lift: Tensor, c_kv: Tensor, seq_lens: Tensor,
                               softmax_scale: float, block_table_row: Tensor,
                               written: Tensor | None = None,
                               write_offset: Tensor | None = None,
                               page_size: int = 0,
                               active_rows: int | None = None) -> Tensor:
    """Dense causal MLA attention over the window a block table names. ``[S, H, L]`` fp32.

    Args:
        q_lift: ``[S, H, L]`` bf16/fp16, the absorbed query.
        c_kv: ``[slots, L]`` the whole latent bank, same dtype.
        seq_lens: ``[S]`` int, each query's causal length: query ``s`` attends window rows
            ``0 .. seq_lens[s] - 1``. The same column the indexer's bypass reads.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32, the window's pages in order, ``-1`` pads.
        written: ``[tokens, L]`` this step's own rows, overlaid at ``write_offset``.
        write_offset: ``[1, 1]`` int32, read on device.
        page_size: rows per page, a trace-time int.
        active_rows: a trace-time int in ``1 .. S``, or None for ``S``: queries from it
            on (a padded chunk's rows) come back as zero rows, written by the kernel, and
            their ``seq_lens`` entries are not read.

    The whole window is staged, as the sparse seam stages it: this step's rows are
    overlaid at a runtime offset, and only the window's own extent is a trace-time
    bound on where they land.

    Raises:
        MlaDenseWindowError: for a malformed call or a geometry the kernel does not serve.
    """
    staged, seq, heads, latent, active = _checked_call(
        q_lift, c_kv, seq_lens, softmax_scale, block_table_row, written, write_offset,
        page_size, active_rows, 1, "seq_lens")
    page = int(page_size)
    table = block_table_row
    rows = seq * heads
    programs = _programs(rows)
    _count_dispatch(programs)
    q_rows = q_lift.contiguous().reshape(rows, latent)
    limits = seq_lens.to(torch.int32).reshape(seq, 1).expand(seq, heads).reshape(rows)
    overlaid = written is not None and int(written.shape[0]) > 0
    call = wrap_nki(mla_dense_window_kernel)
    if programs == 2:
        call = call[2]
    out = call(
        q_rows,
        c_kv.contiguous(),
        table.contiguous().to(torch.int32),
        limits.contiguous(),
        written.contiguous() if overlaid else None,
        write_offset.contiguous().to(torch.int32) if overlaid else None,
        float(softmax_scale),
        page,
        int(staged),
        active * heads,
        SOURCE_DIGEST,
        STAGING_DIGEST,
    )
    return out.reshape(seq, heads, latent)


def mla_dense_window_attention_partial(q_lift: Tensor, c_kv: Tensor, owned_lens: Tensor,
                                       softmax_scale: float, block_table_row: Tensor,
                                       written: Tensor | None = None,
                                       write_offset: Tensor | None = None,
                                       page_size: int = 0,
                                       active_rows: int | None = None
                                       ) -> tuple[Tensor, Tensor]:
    """One DCP rank's dense window attention: the partial contract of ``dcp_merge``.

    Args:
        q_lift: ``[S, Hq, L]`` bf16/fp16, every head of the CP group (``Hq = CP * H``).
        c_kv: ``[slots, L]`` the whole latent bank, same dtype.
        owned_lens: ``[S]`` int32, each query's causal length in this rank's own window
            (its owned 128-token blocks in order): ``own_c(n) = sum over blocks b with
            b % CP == c of min(128, max(0, n - 128 b))`` for global causal length n. 0 is
            allowed: a non-owner rank in a request's first block sees no row.
        softmax_scale, block_table_row, page_size: as :func:`mla_dense_window_attention`;
            the table names this rank's own pages.
        written: ``[T, L]`` only this rank's owned rows of the step, compacted in window
            order; at CP 1 the whole step. Under DCP each chunk key row is attended on
            exactly one rank: its owner, ``(position // 128) % CP``. The owned rows of a
            contiguous step are one contiguous run of the window, and the kernel does not
            mask the overlay: a row of another rank's in ``written`` is attended here. T is
            ``n / CP`` at every step start when ``128 * CP`` divides the step's n rows;
            otherwise it varies, and pad rows of a static T need window rows too.
        write_offset: ``[1, 1]`` int32, ``own_c`` of the first owned position: the window
            row of ``written[0]``. ``write_offset + T`` must stay inside the window
            (refused eagerly, unchecked in a traced graph).
        active_rows: a trace-time int in ``1 .. S``, or None for ``S``: queries from it on
            come back as 0 with lse EMPTY_LSE, and their ``owned_lens`` are not read.

    Returns:
        ``(partial [Hq, S, L], lse [Hq, S])`` float32: the attention normalised over this
        rank's rows, head-major, and ``softmax_scale * m + ln(l)`` over them. A query of
        length 0 returns 0 and :data:`~vllm_neuron.functional.attention.dcp_merge.EMPTY_LSE`.

    Raises:
        MlaDenseWindowError: for a malformed call or a geometry the kernel does not serve.
    """
    if owned_lens.dtype != torch.int32:
        raise MlaDenseWindowError(
            f"owned_lens must be int32, the dtype the kernel reads (no cast is traced here); "
            f"got {owned_lens.dtype}"
        )
    staged, seq, heads, latent, active = _checked_call(
        q_lift, c_kv, owned_lens, softmax_scale, block_table_row, written, write_offset,
        page_size, active_rows, 0, "owned_lens")
    programs = _programs(seq * heads)
    _count_dispatch(programs)
    overlaid = written is not None and int(written.shape[0]) > 0
    call = wrap_nki(mla_dense_window_partial_kernel)
    if programs == 2:
        call = call[2]
    return call(
        q_lift.contiguous(),
        c_kv.contiguous(),
        block_table_row.contiguous().to(torch.int32),
        owned_lens.contiguous(),
        written.contiguous() if overlaid else None,
        write_offset.contiguous().to(torch.int32) if overlaid else None,
        float(softmax_scale),
        int(page_size),
        int(staged),
        active,
        SOURCE_DIGEST,
        STAGING_DIGEST,
    )


# --------------------------------------------------------------------------- #
# Masked mode: the sparse selection as a predicate over the resident window.
#
# Work is dealt by 128-query block, and a block's tiles are head-major (128 queries of
# one head), so a block's heads share its predicate:
# * which pools a query selects. The last column of the g-th group of P columns names a
#   complete selected pool (or holds -1). A pool id is split as a count row, a word of
#   that row and a bit of that word (:func:`_pool_split`), and the PE sums, per query, the
#   one-hot row of each group's count row times its word weighted 2 ** bit: ``count[row,
#   word] = sum_g [row_g == row] [word_g == word] 2 ** bit_g``, whose set bits are the
#   selected pools, exact in fp32. The groups ride the contraction partitions (one PE
#   transpose of the ids per 128 groups), so a group costs ``row_width + 2 * words``
#   elements on the Vector engine instead of a compare against every pool. A selected pool
#   keeps its P rows; the open pool (the one holding the query's last row, ``seq - 1``)
#   keeps its first ``seq % P``; every other pool keeps none: one uint8 predicate tile
#   over the window.
# * per head: MM1 of every 512-row block of the window into fp32 PSUM, copied into one fp32
#   score row by the predicate (an unselected row reads _UNSELECTED_SCORE), the block
#   maximum taken by the same instruction; then the exp of the row against its maximum, in
#   place, with its block sums; then p normalised, put on the key partitions by PE
#   transposes, split hi/lo, and MM2 into one PSUM bank over the whole window.
# The score row is kept (4 bytes per window row per partition) so MM1 runs once.
# --------------------------------------------------------------------------- #

#: The score a window row outside the selection reads: a power of two so far below any
#: score that ``exp(scale * (_UNSELECTED_SCORE - max))`` is exactly 0.0.
_UNSELECTED_SCORE = -(2.0 ** 30)
#: The row-max floor: a query that selects no row exponentiates every row to 0.0 and
#: returns zeros (its sum is floored at _TOTAL_FLOOR).
_SELECTED_MAX_FLOOR = _UNSELECTED_SCORE / 2
#: Queries whose one-hot rows and pool counts one pass of the selection count builds.
_COUNT_QUERIES = 128
#: Pools per count word: pool ``b`` of a word weighs ``2 ** b`` in the count's fp8 e5m2
#: operand, which holds 2**0 .. 2**15 exactly, and a word's count, a sum of distinct
#: weights, is at most 2**16 - 1: exact in fp32 (PSUM and SBUF) and in int32.
_WORD_BITS = 16
#: SBUF bytes per partition on this core (224 KiB).
SBUF_PARTITION_BYTES = 229376


def _pool_split(pools: int) -> tuple[int, int]:
    """``(row_width, words)``: with ``B`` = _WORD_BITS, pool ``p`` is bit ``p % B`` of word
    ``(p // B) % words`` of count row ``p // (B * words)``.

    A group costs the Vector engine ``row_width`` one-hot elements and ``2 * words``
    encoded ones (:func:`_selected_pools`), with ``B * row_width * words >= pools``.
    ``words`` is the largest power of two with ``2 * B * words ** 2 <= pools``, which keeps
    the sum within about 1.25x of its least, ``sqrt(8 * pools / B)``, and both widths at
    most ``sqrt(2 * pools / B) + 1``: below 256, so every field and ramp value is an integer
    exact in bf16, for any window below 65536 pools, far past what SBUF holds.
    """
    words = 1
    for k in range(1, 16):
        if 2 * _WORD_BITS * (1 << (2 * k)) <= pools:
            words = 1 << k
    return -(-pools // (_WORD_BITS * words)), words


def _log2(n: int) -> int:
    """``log2(n)`` of a power of two ``n`` below 2**31 (a loop the kernel front end traces)."""
    k = 0
    for i in range(1, 31):
        if (1 << i) <= n:
            k = i
    return k


# Fields of a group's pool id in the count's id tiles (:func:`_selected_pools`).
_ID_ROW = 0         # the count row, one-hot over row_width
_ID_WORD = 1        # the word of the row, one-hot over words
_ID_WEIGHT = 2      # 2 ** bit, the pool's weight in its word
_ID_FIELDS = 3


def _selected_pools(rows_i, qn, pool_size, row_ramp, word_ramp, sel_i):
    """Fill ``sel_i[q, p]`` with 1 where query row ``q`` selects complete pool ``p``, else 0.

    ``rows_i`` holds the block's selection rows ``[128, cols]`` int32 (rows ``0 .. qn -
    1``); ``row_ramp`` and ``word_ramp`` are the bf16 ramps ``[128, row_width, Q]`` and
    ``[128, words, Q]`` (value ``v`` at ``[:, v, :]``, every partition and query column
    alike; ``Q`` is _COUNT_QUERIES); ``sel_i`` is a ``[128, B * row_width * words]`` int32
    tile, ``B`` being _WORD_BITS.

    The last column of group ``g`` holds ``P * id + P - 1`` for a complete pool ``id`` and
    -1 otherwise (:func:`_selection_predicate`). Shifted right by ``log2 P`` it is the id,
    split as count row, word and bit (:func:`_pool_split`); from a -1 the count row is a
    value no ramp holds (-1, or a large value if the shift is logical), so the group's
    one-hot row is zero. Per query the PE sums ``count[row, word] = sum_g [row_g == row]
    [word_g == word] 2 ** bit_g``. A query names each pool at most once
    (:func:`mla_masked_window_kernel`), so a word's count is a sum of distinct powers of
    two, and its set bits are the pools the query selects. Every field is a small integer,
    exact in bf16 (:func:`_pool_split`), every weight a power of two at most 2**15, exact in
    bf16 and fp8 e5m2, and every count at most 2**16 - 1, exact in fp32 and int32. Every PE
    transpose is bit-accurate on this core, so the counts reach the query partitions exact.

    No count accumulates in PSUM: each group tile's matmul writes its partial count whole,
    and the partial counts, sums over disjoint groups, are added on the query partitions.
    The Tensor Engine tracks accumulation per PSUM bank, and a matmul that starts a group
    resets the bank, so the chains of the queries sharing one, which the scheduler
    interleaves, would lose terms; the NKI simulator, running the program order, does not
    show it. The matmuls are plain: in double-row mode the PE adds a partition's two
    products before the column sum and keeps only the larger when their exponents differ
    by 11 or more, which a count of distinct powers of two cannot afford.
    """
    cols = rows_i.shape[1]
    groups_n = cols // pool_size
    row_width = row_ramp.shape[1]
    words = word_ramp.shape[1]
    n_gt = -(-groups_n // ROW_TILE)
    pool_shift = _log2(pool_size)
    bit_shift = _log2(_WORD_BITS)

    # ---- each group's pool id as count row, word and 2 ** bit, in fp32 for the PE ---------
    last = rows_i.reshape((ROW_TILE, groups_n, pool_size))[0:qn, :, pool_size - 1]
    part_i = _sb((ROW_TILE, groups_n), nl.int32)
    ids_f = _sb((ROW_TILE, _ID_FIELDS, groups_n), nl.float32)
    nisa.tensor_scalar(dst=part_i[0:qn, :], data=last, op0=nl.right_shift,
                       operand0=pool_shift + bit_shift + _log2(words),
                       engine=nisa.engine.vector)
    nisa.tensor_copy(dst=ids_f[0:qn, _ID_ROW, :], src=part_i[0:qn, :],
                     engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=part_i[0:qn, :], data=last, op0=nl.right_shift,
                       operand0=pool_shift + bit_shift, op1=nl.bitwise_and,
                       operand1=words - 1, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=ids_f[0:qn, _ID_WORD, :], src=part_i[0:qn, :],
                     engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=part_i[0:qn, :], data=last, op0=nl.right_shift,
                       operand0=pool_shift, op1=nl.bitwise_and, operand1=_WORD_BITS - 1,
                       engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=part_i[0:qn, :], data=part_i[0:qn, :], op0=nl.left_shift,
                       operand0=1, reverse0=True, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=ids_f[0:qn, _ID_WEIGHT, :], src=part_i[0:qn, :],
                     engine=nisa.engine.vector)

    # ---- the groups onto the partitions: [group, field, tile, query], 128 groups a tile ---
    ids_t = _sb((ROW_TILE, _ID_FIELDS, n_gt, ROW_TILE), nl.bfloat16)
    for t in range(n_gt):
        g0 = t * ROW_TILE
        gt = min(ROW_TILE, groups_n - g0)
        ids_ps = nl.ndarray((ROW_TILE, _ID_FIELDS, ROW_TILE), dtype=nl.float32,
                            buffer=nl.psum)
        for f in range(_ID_FIELDS):
            nisa.nc_transpose(dst=ids_ps[0:gt, f, 0:qn], data=ids_f[0:qn, f, g0:g0 + gt])
        nisa.tensor_copy(dst=ids_t[0:gt, :, t, 0:qn], src=ids_ps[0:gt, :, 0:qn],
                         engine=nisa.engine.scalar)

    # ---- count[row, (q, word)]: per query, its groups' one-hot rows times weighted words ---
    # The one-hot and word tiles are [group, tile, value, query], the query innermost: the
    # ids broadcast over the value axis and the ramps are materialised, so every operand
    # streams with unit stride. Every operand is exact in fp8 (the one-hot rows in e4m3,
    # the weighted words in e5m2). Per query, each group tile's matmul writes its partial
    # count whole (accumulate=False); a PSUM bank holds every tile's partial counts of
    # ``bank_queries`` queries, copied out together.
    bank_queries = _PSUM_BANK_BYTES // (4 * n_gt * words)
    counts = _sb((ROW_TILE, n_gt, ROW_TILE, words), nl.float32)
    for qs0 in range(0, qn, _COUNT_QUERIES):
        qs_n = min(_COUNT_QUERIES, qn - qs0)
        oh_row = _sb((ROW_TILE, n_gt, row_width, _COUNT_QUERIES), nl.float8_e4m3)
        weighted = _sb((ROW_TILE, n_gt, words, _COUNT_QUERIES), nl.float8_e5m2)
        hit = _sb((ROW_TILE, words, _COUNT_QUERIES), nl.bfloat16)
        for t in range(n_gt):
            gt = min(ROW_TILE, groups_n - t * ROW_TILE)
            nisa.tensor_tensor(
                dst=oh_row[0:gt, t, :, 0:qs_n],
                data1=ids_t[0:gt, _ID_ROW, t, qs0:qs0 + qs_n].expand_dim(1).broadcast(
                    1, row_width),
                data2=row_ramp[0:gt, :, 0:qs_n], op=nl.equal, engine=nisa.engine.vector)
            nisa.tensor_tensor(
                dst=hit[0:gt, :, 0:qs_n],
                data1=ids_t[0:gt, _ID_WORD, t, qs0:qs0 + qs_n].expand_dim(1).broadcast(
                    1, words),
                data2=word_ramp[0:gt, :, 0:qs_n], op=nl.equal, engine=nisa.engine.vector)
            nisa.tensor_tensor(
                dst=weighted[0:gt, t, :, 0:qs_n], data1=hit[0:gt, :, 0:qs_n],
                data2=ids_t[0:gt, _ID_WEIGHT, t, qs0:qs0 + qs_n].expand_dim(1).broadcast(
                    1, words),
                op=nl.multiply, engine=nisa.engine.vector)
        for qb0 in range(0, qs_n, bank_queries):
            qb_n = min(bank_queries, qs_n - qb0)
            count_ps = nl.ndarray((row_width, n_gt, bank_queries, words), dtype=nl.float32,
                                  buffer=nl.psum)
            for j in range(qb_n):
                qq = qb0 + j
                for t in range(n_gt):
                    gt = min(ROW_TILE, groups_n - t * ROW_TILE)
                    nisa.nc_matmul(dst=count_ps[:, t, j, :], stationary=oh_row[0:gt, t, :, qq],
                                   moving=weighted[0:gt, t, :, qq], accumulate=False)
            q0 = qs0 + qb0
            nisa.tensor_copy(dst=counts[0:row_width, :, q0:q0 + qb_n, :],
                             src=count_ps[:, :, 0:qb_n, :], engine=nisa.engine.scalar)

    # ---- back onto the query partitions, the partial counts added; bit b of word w of row r:
    # pool (r, w, b). A PSUM bank takes every tile's transposes of ``per_bank`` words.
    mask_i = _sb((ROW_TILE, row_width, words), nl.int32)
    per_bank = _PSUM_BANK_BYTES // (4 * n_gt * row_width)
    for w0 in range(0, words, per_bank):
        nw = min(per_bank, words - w0)
        back_ps = nl.ndarray((ROW_TILE, n_gt, per_bank, row_width), dtype=nl.float32,
                             buffer=nl.psum)
        for j in range(nw):
            for t in range(n_gt):
                nisa.nc_transpose(dst=back_ps[0:qn, t, j, :],
                                  data=counts[0:row_width, t, 0:qn, w0 + j])
        nisa.tensor_reduce(dst=mask_i[0:qn, :, w0:w0 + nw].permute((0, 2, 1)), op=nl.add,
                           data=back_ps[0:qn, :, 0:nw, :].permute((0, 2, 3, 1)), axis=3)
    sel4 = sel_i.reshape((ROW_TILE, row_width, words, _WORD_BITS))
    for b in range(_WORD_BITS):
        nisa.tensor_scalar(dst=sel4[0:qn, :, :, b], data=mask_i[0:qn, :, :], op0=nl.right_shift,
                           operand0=b, op1=nl.bitwise_and, operand1=1, engine=nisa.engine.vector)


def _selection_predicate(index_hbm, s0, qn, pool_size, row_ramp, word_ramp, pool_ids, ranks,
                         pred):
    """Fill ``pred[q, p, r]`` (window row ``p * P + r``) for queries ``s0 .. s0 + qn - 1``.

    ``pool_ids`` is the fp32 ramp ``1 .. pools`` on every partition (pool ``p`` at column
    ``p``, valued ``p + 1``), ``ranks`` the fp32 ramp ``1 .. P`` and ``pred`` a ``[128,
    pools, P]`` uint8 tile: 1 on a row the query selects, 0 elsewhere.

    Two layouts reach here, and both are read the same way, one group of ``P`` columns at a
    time. ``dsa_index_expand`` writes the g-th selected pool's rows ``P * id + o`` at
    columns ``g * P + o``, then the open pool's rows in ``P - 1`` tail columns, then -1;
    the short-sequence bypass writes rows ``0 .. seq - 1`` at columns ``0 .. seq - 1``, so
    its open pool sits in a group of its own, short. In both, a group whose last column
    holds a row names a complete pool (``row // P``), any other group's last column is -1,
    every row is below the query's causal length ``seq``, and the open pool's rows ``P *
    (seq // P) .. seq - 1`` are all present, so ``seq`` is the largest row plus one whenever
    that pool is not empty. Every group is counted, so the count of selected pools is not
    needed. Every row is far below 2**24, so its fp32 value is exact, and ``P`` is a power
    of two, so the scalings by ``1 / P`` are exact. The open pool is found first, so the
    selection rows are dead before the count's one-hot tiles are allocated.

    A complete pool lies below ``seq``, so it is never the open pool, and row ``r`` of pool
    ``p`` is attended when ``p`` is selected or is the open pool with ``r < seq % P``: per
    row, one compare of the 1-based pool ids against ``open + 1`` (against 0, which no id
    holds, where ``r >= seq % P``), taken with the larger of the selection bit.
    """
    pools = pool_ids.shape[1]
    row_width, words = row_ramp.shape[1], word_ramp.shape[1]
    cols = index_hbm.shape[1]
    inv_pool = 1.0 / pool_size
    rows_i = _sb((ROW_TILE, cols), nl.int32)
    nisa.dma_copy(dst=rows_i[0:qn, :],
                  src=index_hbm.ap(pattern=[[cols, qn], [1, cols]], offset=s0 * cols))

    # ---- the open pool keeps its rows below seq: seq % P of them --------------------------
    seq_f = _col(ROW_TILE, nl.float32)
    nisa.tensor_reduce(dst=seq_f[0:qn, :], op=nl.maximum, data=rows_i[0:qn, :], axis=1)
    nisa.tensor_scalar(dst=seq_f[0:qn, :], data=seq_f[0:qn, :], op0=nl.add, operand0=1.0,
                       engine=nisa.engine.vector)
    seq_i = _col(ROW_TILE, nl.int32)
    nisa.tensor_copy(dst=seq_i[0:qn, :], src=seq_f[0:qn, :], engine=nisa.engine.vector)
    rem_i = _col(ROW_TILE, nl.int32)
    nisa.tensor_scalar(dst=rem_i[0:qn, :], data=seq_i[0:qn, :], op0=nl.bitwise_and,
                       operand0=pool_size - 1, engine=nisa.engine.vector)
    count = _col(ROW_TILE, nl.float32)
    nisa.tensor_copy(dst=count[0:qn, :], src=rem_i[0:qn, :], engine=nisa.engine.vector)
    # open + 1 = (seq - seq % P) / P + 1, the open pool's 1-based id
    open_1 = _col(ROW_TILE, nl.float32)
    nisa.tensor_tensor(dst=open_1[0:qn, :], data1=seq_f[0:qn, :], data2=count[0:qn, :],
                       op=nl.subtract, engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=open_1[0:qn, :], data=open_1[0:qn, :], op0=nl.multiply,
                       operand0=inv_pool, op1=nl.add, operand1=1.0, engine=nisa.engine.vector)

    sel_i = _sb((ROW_TILE, _WORD_BITS * row_width * words), nl.int32)
    _selected_pools(rows_i, qn, pool_size, row_ramp, word_ramp, sel_i)

    # ---- row r of a pool: its bit, or the open pool's while r < seq % P -------------------
    # target[q, r] = open + 1 where r + 1 <= seq % P, else 0 (no pool's 1-based id)
    target = _sb((ROW_TILE, pool_size), nl.float32)
    nisa.tensor_scalar(dst=target[0:qn, :], data=ranks[0:qn, :], op0=nl.less_equal,
                       operand0=count[0:qn, :], op1=nl.multiply, operand1=open_1[0:qn, :],
                       engine=nisa.engine.vector)
    for r in range(pool_size):
        nisa.scalar_tensor_tensor(dst=pred[0:qn, :, r], data=pool_ids[0:qn, :], op0=nl.equal,
                                  operand0=target[0:qn, r:r + 1], op1=nl.maximum,
                                  operand1=sel_i[0:qn, 0:pools])


def _window_scores(q_t, k_t, rt, t0, tw):
    """``[128, 512]`` fp32 PSUM: rows ``0 .. rt - 1`` hold ``q . window`` over rows
    ``t0 .. t0 + tw - 1``."""
    n_lat = q_t.shape[1]
    s_ps = nl.ndarray((ROW_TILE, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
    for li in range(n_lat):
        nisa.nc_matmul(dst=s_ps[0:rt, 0:tw], stationary=q_t[:, li, 0:rt],
                       moving=k_t[:, li, t0:t0 + tw], accumulate=(li > 0))
    return s_ps


def _masked_tile(q_hbm, out_hbm, k_t, v_rows, pred_rows, softmax_scale, rt, row_stride,
                 row_off):
    """One tile of ``rt`` query rows of one head over the whole window: softmax, MM2, store.

    Row ``p`` of the tile reads its query at element ``row_off + p * row_stride`` of
    ``q_hbm`` and writes its output at the same element of ``out_hbm``; ``pred_rows[p]``
    is its query's predicate over the window.
    """
    latent = v_rows.shape[2]
    width = pred_rows.shape[1]
    kv_dtype = v_rows.dtype
    n_blocks = (width + MOVING_MAX - 1) // MOVING_MAX
    q_t = _query_columns(q_hbm, rt, row_stride, row_off, latent)

    # ---- MM1 once: the selected scores and the block maxima -------------------------
    scores = _sb((ROW_TILE, width), nl.float32)
    block_max = _sb((ROW_TILE, n_blocks), nl.float32)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        tw = min(MOVING_MAX, width - t0)
        s_ps = _window_scores(q_t, k_t, rt, t0, tw)
        nisa.select_reduce(dst=scores[0:rt, t0:t0 + tw], predicate=pred_rows[0:rt, t0:t0 + tw],
                           on_true=s_ps[0:rt, 0:tw], on_false=_UNSELECTED_SCORE,
                           reduce_res=block_max[0:rt, b:b + 1],
                           reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_op=nl.maximum)
    row_max = _col(ROW_TILE, nl.float32)
    nisa.tensor_reduce(dst=row_max[0:rt, :], op=nl.maximum, data=block_max[0:rt, :], axis=1)
    nisa.tensor_scalar(dst=row_max[0:rt, :], data=row_max[0:rt, :], op0=nl.maximum,
                       operand0=_SELECTED_MAX_FLOOR, engine=nisa.engine.vector)
    neg_top = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=neg_top[0:rt, :], data=row_max[0:rt, :], op0=nl.multiply,
                       operand0=-softmax_scale, engine=nisa.engine.vector)

    # ---- the exps in place and their block sums --------------------------------------
    block_sum = _sb((ROW_TILE, n_blocks), nl.float32)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        tw = min(MOVING_MAX, width - t0)
        nisa.activation(dst=scores[0:rt, t0:t0 + tw], op=nl.exp, data=scores[0:rt, t0:t0 + tw],
                        bias=neg_top[0:rt, :], scale=softmax_scale, reduce_op=nl.add,
                        reduce_res=block_sum[0:rt, b:b + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
    row_sum = _col(ROW_TILE, nl.float32)
    nisa.tensor_reduce(dst=row_sum[0:rt, :], op=nl.add, data=block_sum[0:rt, :], axis=1)
    nisa.tensor_scalar(dst=row_sum[0:rt, :], data=row_sum[0:rt, :], op0=nl.maximum,
                       operand0=_TOTAL_FLOOR, engine=nisa.engine.vector)
    recip = _col(ROW_TILE, nl.float32)
    nisa.reciprocal(dst=recip[0:rt, :], data=row_sum[0:rt, :])

    # ---- p normalised, onto the key partitions, hi/lo, MM2 ---------------------------
    pv_ps = nl.ndarray((ROW_TILE, latent), dtype=nl.float32, buffer=nl.psum)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        tw = min(MOVING_MAX, width - t0)
        p = scores[:, t0:t0 + tw]
        nisa.tensor_scalar(dst=p[0:rt, :], data=p[0:rt, :], op0=nl.multiply,
                           operand0=recip[0:rt, :], engine=nisa.engine.vector)
        n_ck = tw // KEY_CHUNK
        pt_ps = nl.ndarray((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), dtype=nl.float32,
                           buffer=nl.psum)
        for ck in range(n_ck):
            nisa.nc_transpose(dst=pt_ps[:, ck * ROW_TILE:ck * ROW_TILE + rt],
                              data=p[0:rt, ck * KEY_CHUNK:(ck + 1) * KEY_CHUNK])
        # hi = bf16(p), lo = bf16(p - hi): the difference is exact in fp32. A whole tile's
        # transposes fill their columns back to back, so one instruction splits them all.
        p_hi = _sb((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), kv_dtype)
        p_lo = _sb((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), kv_dtype)
        n_spans = n_ck
        span = rt
        if rt == ROW_TILE:
            n_spans = 1
            span = n_ck * ROW_TILE
        for sp in range(n_spans):
            c0 = sp * ROW_TILE
            c1 = c0 + span
            nisa.tensor_copy(dst=p_hi[:, c0:c1], src=pt_ps[:, c0:c1], engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=p_lo[:, c0:c1], data1=pt_ps[:, c0:c1], data2=p_hi[:, c0:c1],
                               op=nl.subtract, engine=nisa.engine.vector)
        for ck in range(n_ck):
            c0 = ck * ROW_TILE
            chunk = t0 // KEY_CHUNK + ck
            nisa.nc_matmul(dst=pv_ps[0:rt, :], stationary=p_hi[:, c0:c0 + rt],
                           moving=v_rows[:, chunk, :], accumulate=(chunk > 0))
            nisa.nc_matmul(dst=pv_ps[0:rt, :], stationary=p_lo[:, c0:c0 + rt],
                           moving=v_rows[:, chunk, :], accumulate=True)
    out_sb = _sb((ROW_TILE, latent), nl.float32)
    nisa.tensor_copy(dst=out_sb[0:rt, :], src=pv_ps[0:rt, :])
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[row_stride, rt], [1, latent]], offset=row_off),
                  src=out_sb[0:rt, :])


def _masked_block(q_hbm, index_hbm, out_hbm, k_t, v_rows, consts, softmax_scale, pool_size,
                  s0, qn):
    """Queries ``s0 .. s0 + qn - 1``, every head: their predicate once, then one tile per head.

    ``consts`` is ``[row_ramp, word_ramp, pool_ids, ranks]`` (:func:`_masked_body`).
    """
    heads, latent = q_hbm.shape[1], q_hbm.shape[2]
    pool_ids = consts[2]
    pools = pool_ids.shape[1]
    pred = _sb((ROW_TILE, pools, pool_size), nl.uint8)
    _selection_predicate(index_hbm, s0, qn, pool_size, consts[0], consts[1], pool_ids,
                         consts[3], pred)
    pred_rows = pred.reshape((ROW_TILE, pools * pool_size))
    for h in range(heads):
        _masked_tile(q_hbm, out_hbm, k_t, v_rows, pred_rows, softmax_scale, qn,
                     heads * latent, (s0 * heads + h) * latent)


def _masked_body(q_hbm, window_hbm, index_hbm, softmax_scale, key_rows, pool_size,
                 out_hbm):
    """Attend each query's selection, given as ``index_hbm``, over the resident window.

    The query blocks are dealt round-robin over the programs: the whole 128-query blocks,
    then the partial one. ``program_id`` is a trace-time int (the kernel is traced once
    per program), so each program keeps only its own blocks; every loop bound is a
    trace-time int.
    """
    seq, latent = q_hbm.shape[0], q_hbm.shape[2]
    ops = _window_rows(window_hbm, key_rows, latent, True)
    v_rows = ops[_W_ROWS]
    k_t = ops[_W_T]
    pools = key_rows // pool_size
    row_width, words = _pool_split(pools)
    row_ramp = _sb((ROW_TILE, row_width, _COUNT_QUERIES), nl.bfloat16)
    nisa.iota(dst=row_ramp, pattern=[[1, row_width], [0, _COUNT_QUERIES]], offset=0)
    word_ramp = _sb((ROW_TILE, words, _COUNT_QUERIES), nl.bfloat16)
    nisa.iota(dst=word_ramp, pattern=[[1, words], [0, _COUNT_QUERIES]], offset=0)
    pool_ids = _sb((ROW_TILE, pools), nl.float32)
    nisa.iota(dst=pool_ids, pattern=[[1, pools]], offset=1)
    ranks = _sb((ROW_TILE, pool_size), nl.float32)
    nisa.iota(dst=ranks, pattern=[[1, pool_size]], offset=1)
    consts = []
    consts.append(row_ramp)
    consts.append(word_ramp)
    consts.append(pool_ids)
    consts.append(ranks)
    n_full = seq // ROW_TILE
    tail = seq - n_full * ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for qb in nl.affine_range(prg, n_full, n_prgs):
        _masked_block(q_hbm, index_hbm, out_hbm, k_t, v_rows, consts, softmax_scale,
                      pool_size, qb * ROW_TILE, ROW_TILE)
    if tail > 0 and n_full % n_prgs == prg:
        _masked_block(q_hbm, index_hbm, out_hbm, k_t, v_rows, consts, softmax_scale,
                      pool_size, n_full * ROW_TILE, tail)


@nki.jit
def mla_masked_window_kernel(q_hbm, bank_hbm, table_hbm, index_hbm, written_hbm,
                             write_offset_hbm, softmax_scale, page_size, key_rows, pool_size,
                             source_digest=0, staging_digest=0):
    """Attention of each query over its selected rows, as a predicate over the whole window.

    Args:
        q_hbm: ``[S, H, L]`` bf16 or fp16, the absorbed queries. ``L`` is a multiple of
            128, at most :data:`LATENT_MAX`.
        bank_hbm: ``[slots, L]``, the whole latent bank, the dtype of ``q_hbm``.
        table_hbm: ``[pages, 1]`` int32, the window's pages in order (``-1`` pads).
        index_hbm: ``[S, cols]`` int32, each query's selected rows (-1 pads), in
            ``dsa_index_expand``'s layout or the short-sequence bypass's
            (:func:`_selection_predicate`): whole pools of ``P`` rows in groups of ``P``
            columns, each pool at most once, and the open pool's rows below the query's
            causal length. ``cols`` is a multiple of ``P``. Every selected row lies in the
            window.
        written_hbm: ``[tokens, L]`` or None, this step's rows, overlaid on the window.
        write_offset_hbm: ``[1, 1]`` int32 or None, the window row ``written`` starts at.
        softmax_scale: python float applied to the scores before the softmax.
        page_size: python int, rows per page.
        key_rows: python int, the staged window rows (``pages * page_size``), a multiple of
            128 and of ``pool_size``.
        pool_size: python int ``P``, a power of two: rows per pool.
        source_digest, staging_digest: python ints that only key the compiled-kernel cache
            on this file and on ``mla_sparse.py`` (the staging helper).

    Returns:
        ``[S, H, L]`` fp32 in shared HBM: ``softmax(scale * q . k) @ k`` over the rows the
        query selects. A query that selects no row returns zeros.

    Under a two-program launch the 128-query blocks are dealt round-robin between the
    programs.
    """
    seq, heads, latent = q_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    window_hbm = _ms._staged_window(bank_hbm, table_hbm, written_hbm, write_offset_hbm,
                                    page_size)
    _masked_body(q_hbm, window_hbm, index_hbm, softmax_scale, key_rows, pool_size, out_hbm)
    return out_hbm


# ---- the masked mode's dispatch rule: SBUF capacity and a cost model ------------------ #
#
# Every rate below is a hardware rate of this core with its provenance; the crossover is
# not a constant but the root of masked_window_cost_us == gather_cost_us at a call's shape.

#: A Vector-engine pass over fp32 data: a fixed cost plus one per element of the free
#: axis, the same on any number of partitions. MEASURED for max8 / nc_match_replace8
#: (``reports/rotational_topk.md`` section 4): 170 ns + 1.04 ns per element.
DVE_PASS_NS = 170.0
DVE_FP32_NS = 1.04
#: The Vector engine's 2x mode: a bf16 ``tensor_tensor`` with every operand in SBUF takes
#: two elements per cycle (DOCUMENTED, the architecture guide's 512 bf16 against 256 fp32
#: input elements per cycle; ``reports/dsa_score_gemm.md`` section 5).
DVE_BF16_NS = DVE_FP32_NS / 2
#: The scalar engine: one element per 1.2 GHz cycle, MEASURED for ``activation`` from PSUM
#: (``reports/dsa_score_gemm.md`` section 5: 1017-1030 ns at 1024 elements, 1878 ns at
#: 2048), whose fixed cost, DERIVED from the same two sizes, is about DVE_PASS_NS.
ACT_ELEMENT_NS = 0.834
#: The PE streams one 2-byte moving column per 2.4 GHz cycle.
PE_COLUMN_NS = 1.0 / 2.4
#: One query's serial gather of the sparse kernel, MEASURED for 2176 rows of 1 KiB
#: (``reports/mla_8k.md``, profile of 2026-10-10 02:35Z): descriptor generation 8.72 us and
#: DMA 8.05 us. The sparse kernel's other costs (block waits, MM1, softmax) are left out,
#: so this prices the gather path at its floor.
GATHER_DESCRIPTOR_NS = 8720.0 / 2176
GATHER_NS_PER_BYTE = 8050.0 / (2176 * 1024)
#: HBM bytes per ns of one logical core (716e9 B/s, ``trn2_constants.json``
#: ``hbm_bw_bytes_per_s``).
HBM_BYTES_PER_NS = 716.0


def masked_window_sbuf_bytes(key_rows: int, latent: int, pool_size: int, index_cols: int,
                             kv_bytes: int) -> int:
    """SBUF bytes per partition at the masked kernel's peak, its tiles counted by lifetime.

    For the whole call: the window twice (``c_rows`` and ``c_t``), the staging tiles and the
    constants (ramps, pool ids, ranks). Per query block: its predicate and
    the open pool's targets, held while its heads run, plus the largest of three phases
    whose tiles are dead before the next phase allocates, the selection bits ``sel`` live
    through the first two: loading the selection (its rows, the group ids and their
    fields), the count (the transposed fields, the one-hot and weighted tiles, the partial
    counts and the words read back), and one head tile (query, scores, statistics, one bank's hi
    and lo halves, output row).
    """
    pools = key_rows // pool_size
    row_width, words = _pool_split(pools)
    groups = index_cols // pool_size
    n_gt = -(-groups // ROW_TILE)
    n_blocks = -(-key_rows // MOVING_MAX)
    line = _LINE * 4
    ids_t = _ID_FIELDS * n_gt * ROW_TILE * 2
    sel = _WORD_BITS * row_width * words * 4
    resident = (2 * key_rows * latent * kv_bytes // ROW_TILE + 2 * latent * kv_bytes
                + (row_width + words) * _COUNT_QUERIES * 2 + pools * 4 + pool_size * 4)
    predicate = pools * pool_size + pool_size * 4
    load = sel + index_cols * 4 + groups * (1 + _ID_FIELDS) * 4 + ids_t
    count = (sel + ids_t + n_gt * (row_width + words) * _COUNT_QUERIES
             + words * _COUNT_QUERIES * 2 + n_gt * ROW_TILE * words * 4
             + row_width * words * 4)
    tile = (latent * (2 * kv_bytes + 4) + key_rows * 4 + 2 * n_blocks * 4
            + 2 * _TRANSPOSES_PER_BANK * ROW_TILE * kv_bytes + 4 * line + latent * 4)
    return resident + predicate + max(load, count, tile)


def masked_window_cost_us(seq: int, heads: int, latent: int, key_rows: int, pool_size: int,
                          index_cols: int, kv_bytes: int, programs: int) -> float:
    """DERIVED device time of one masked call, from the rates above.

    Each engine's work per program is summed, and the call costs the busiest engine's sum
    after the window is staged and loaded (three HBM passes). Once per program: the PE
    transposes the window and the scalar engine copies it out of PSUM. Per query block,
    priced as the fullest one (``min(seq, 128)`` queries), the predicate: on the Vector
    engine one pass over the selection columns (the causal length), seven over the groups
    (each group's pool id as count row, word and weight), the one-hot and weighted tiles
    of every group tile in 2x mode with the queries on the free axis, the partial counts
    added out of PSUM, one pass over the words per bit and ``P`` passes over the pool axis (the
    predicate's rows of each pool); on the scalar engine the copies of the fields and the
    counts out of PSUM; on the PE the transposes of the fields and the counts and the count
    matmuls. Per head: the Vector engine takes the selected scores and their block maxima
    from PSUM, normalises p, splits it hi/lo out of PSUM and copies the output row; the
    scalar engine runs the exp with its block sums; the PE runs MM1, the fp32 transposes of
    p and MM2 over both halves. The ``[128, 1]`` column ops, the waits between engines and
    the PE's per-instruction issue cost are left out.
    """
    pools = key_rows // pool_size
    row_width, words = _pool_split(pools)
    groups = index_cols // pool_size
    n_gt = -(-groups // ROW_TILE)
    n_lat = latent // LATENT_TILE
    n_blocks = -(-key_rows // MOVING_MAX)
    chunks = key_rows // KEY_CHUNK
    rows = min(seq, ROW_TILE)
    sub_blocks = -(-rows // _COUNT_QUERIES)
    q_blocks = -(-seq // ROW_TILE)
    blocks = -(-q_blocks // programs)
    window_per_bank = _PSUM_BANK_BYTES // (2 * latent)
    window_act_ns = (-(-chunks // window_per_bank)
                     * (DVE_PASS_NS + window_per_bank * latent * ACT_ELEMENT_NS))
    window_pe_ns = PE_COLUMN_NS * chunks * n_lat * KEY_CHUNK
    back_passes = -(-words // (_PSUM_BANK_BYTES // (4 * n_gt * row_width)))
    count_copies = -(-rows // (_PSUM_BANK_BYTES // (4 * n_gt * words)))
    word_pass_ns = DVE_PASS_NS + row_width * words * DVE_FP32_NS
    predicate_dve_ns = (DVE_PASS_NS + index_cols * DVE_FP32_NS
                        + 7 * (DVE_PASS_NS + groups * DVE_FP32_NS)
                        + n_gt * (3 * sub_blocks * DVE_PASS_NS
                                  + (row_width + 2 * words) * rows * DVE_BF16_NS)
                        + back_passes * DVE_PASS_NS + n_gt * row_width * words * DVE_FP32_NS
                        + _WORD_BITS * word_pass_ns
                        + pool_size * (DVE_PASS_NS + pools * DVE_FP32_NS))
    predicate_act_ns = (n_gt * (DVE_PASS_NS + _ID_FIELDS * rows * ACT_ELEMENT_NS)
                        + count_copies * DVE_PASS_NS + n_gt * rows * words * ACT_ELEMENT_NS)
    # A PE transpose streams one column per partition of its input; a query's count takes
    # one matmul per group tile, each tile's partial counts transposed back on their own.
    predicate_pe_ns = PE_COLUMN_NS * (_ID_FIELDS * n_gt * rows + n_gt * words * row_width
                                      + rows * n_gt * words)
    head_dve_ns = (4 * n_blocks * (DVE_PASS_NS + MOVING_MAX * DVE_FP32_NS)
                   + DVE_PASS_NS + latent * DVE_FP32_NS)
    head_act_ns = n_blocks * (DVE_PASS_NS + MOVING_MAX * ACT_ELEMENT_NS)
    head_pe_ns = PE_COLUMN_NS * (n_blocks * n_lat * MOVING_MAX + chunks * (rows + 2 * latent))
    dve_ns = blocks * (predicate_dve_ns + heads * head_dve_ns)
    act_ns = window_act_ns + blocks * (predicate_act_ns + heads * head_act_ns)
    pe_ns = window_pe_ns + blocks * (predicate_pe_ns + heads * head_pe_ns)
    window_ns = 3 * key_rows * latent * kv_bytes / HBM_BYTES_PER_NS
    return (window_ns + max(dve_ns, act_ns, pe_ns)) / 1000.0


def gather_cost_us(seq: int, latent: int, index_cols: int, kv_bytes: int) -> float:
    """DERIVED floor of one gathered call: every query's serial gather, split over two
    programs (the sparse kernel's best launch), whatever the head count."""
    per_query_ns = index_cols * (GATHER_DESCRIPTOR_NS + latent * kv_bytes * GATHER_NS_PER_BYTE)
    return -(-seq // 2) * per_query_ns / 1000.0


def masked_window_serves(seq: int, heads: int, latent: int, key_rows: int, pool_size: int,
                         index_cols: int, q_dtype: torch.dtype, kv_dtype: torch.dtype) -> bool:
    """True when :func:`mla_masked_window_attention` serves this call and beats the gather.

    It takes plain values, so a caller decides at trace time: the geometry is one the
    kernel serves (:func:`_require_masked`), its SBUF footprint
    (:func:`masked_window_sbuf_bytes`) fits a partition, and its DERIVED cost
    (:func:`masked_window_cost_us`) is below the gather path's floor
    (:func:`gather_cost_us`). The gather keeps every other call. Two kinds always stay
    there, at latent 512 in bf16 with 4-row pools and 2176 selection columns. A decode call
    of one or two rows: the gather's floor is one query's gather, and the masked cost is at
    least 2.4x of it at every window of 17 pages or more (the sparse regime's) and every
    head count, since the window's staging and transposes and the predicate's passes over
    the selection columns and the pool axis are paid whatever a block's fill. A window too
    wide to hold resident: the footprint passes 224 KiB above 9216 rows (72 pages of 128),
    the largest multiple of 128 that :func:`masked_window_sbuf_bytes` admits.
    """
    try:
        _require_masked(int(seq), int(heads), int(latent), int(latent), q_dtype, kv_dtype,
                        int(key_rows), int(pool_size), int(index_cols))
    except MlaDenseWindowError:
        return False
    kv_bytes = torch.empty((), dtype=kv_dtype).element_size()
    masked = masked_window_cost_us(seq, heads, latent, key_rows, pool_size, index_cols,
                                   kv_bytes, _masked_programs(seq))
    return masked < gather_cost_us(seq, latent, index_cols, kv_bytes)


def _masked_programs(seq: int) -> int:
    """Two programs on an LNC2 core when the queries span two or more 128-query blocks."""
    return _programs(seq)


def _require_masked(seq: int, heads: int, latent: int, cache_latent: int,
                    q_dtype: torch.dtype, kv_dtype: torch.dtype, key_rows: int,
                    pool_size: int, index_cols: int) -> None:
    _require_operands(seq, heads, latent, cache_latent, q_dtype, kv_dtype)
    if pool_size < 1 or pool_size & (pool_size - 1) != 0:
        raise MlaDenseWindowError(
            f"pool_size must be a power of two, so a pool id is P * id / P exactly in "
            f"fp32; got {pool_size}"
        )
    if index_cols < 1 or index_cols % pool_size != 0:
        raise MlaDenseWindowError(
            f"the selection is read in groups of P = {pool_size} columns, so its width "
            f"must be a positive multiple of {pool_size}; got {index_cols} columns"
        )
    if key_rows < KEY_CHUNK or key_rows % KEY_CHUNK != 0 or key_rows % pool_size != 0:
        raise MlaDenseWindowError(
            f"the window must be whole {KEY_CHUNK}-row chunks of whole pools of "
            f"{pool_size}; got {key_rows} rows"
        )
    kv_bytes = torch.empty((), dtype=kv_dtype).element_size()
    need = masked_window_sbuf_bytes(key_rows, latent, pool_size, index_cols, kv_bytes)
    if need > SBUF_PARTITION_BYTES:
        raise MlaDenseWindowError(
            f"the masked kernel holds the window resident: {key_rows} rows of latent "
            f"{latent} need {need} SBUF bytes per partition, above {SBUF_PARTITION_BYTES}"
        )
    row_width, words = _pool_split(key_rows // pool_size)
    group_tiles = -(-(index_cols // pool_size) // ROW_TILE)
    if 4 * group_tiles * max(row_width, words) > _PSUM_BANK_BYTES:
        raise MlaDenseWindowError(
            f"the selection count keeps one partial count per {ROW_TILE}-group tile in a "
            f"PSUM bank: {group_tiles} tiles of {max(row_width, words)} fp32 values exceed "
            f"its {_PSUM_BANK_BYTES} bytes"
        )


def mla_masked_window_attention(q_lift: Tensor, c_kv: Tensor, topk_indices: Tensor,
                                softmax_scale: float, block_table_row: Tensor,
                                written: Tensor | None = None,
                                write_offset: Tensor | None = None,
                                page_size: int = 0, pool_size: int = 0) -> Tensor:
    """Sparse MLA attention over each query's selection, without gathering it.

    The operands and the result are ``mla_sparse.mla_sparse_attention``'s paged NoPE call,
    plus the selection's pool size; that seam calls this one when
    :func:`masked_window_serves` says so.

    Args:
        q_lift: ``[S, H, L]`` bf16/fp16, the absorbed query.
        c_kv: ``[slots, L]`` the whole latent bank, same dtype.
        topk_indices: ``[S, cols]`` int, ``dsa_index_expand``'s output for pools of
            ``pool_size`` rows, or the short-sequence bypass's causal prefix: every
            selected pool complete, inside the window and named once, and the open
            pool's rows.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32, the window's pages in order, ``-1`` pads.
        written: ``[tokens, L]`` this step's own rows, overlaid at ``write_offset``.
        write_offset: ``[1, 1]`` int32, read on device.
        page_size: rows per page, a trace-time int.
        pool_size: rows per pool (the indexer's ``index_kpool``), a power of two.

    Returns:
        ``[S, H, L]`` fp32, the sparse kernel's answer up to the order of its fp32 sums.

    Every window row is read, selected or not: the predicate drops an unselected row's
    score whatever its value, but MM2 multiplies the row by its zero probability, and zero
    times a non-finite value is NaN. So every row of the window must be finite, where the
    gather path needs only the selected ones. The bank is allocated zeroed
    (``NeuronModelRunner.initialize_kv_cache``) and holds only latents the model wrote, so
    a non-finite window row means the model already wrote a non-finite latent.

    The DCP partial mode (``mla_sparse_attention_partial``) has no masked form: it stays
    on the gather path, which ``dcp_merge`` and its gate are measured against.

    Raises:
        MlaDenseWindowError: for a malformed call or a geometry the kernel does not serve.
    """
    if not softmax_scale > 0:
        raise MlaDenseWindowError(f"softmax_scale must be positive; got {softmax_scale}")
    if block_table_row is None or block_table_row.ndim != 2:
        raise MlaDenseWindowError(
            "the masked window kernel reads a paged window only: pass block_table_row "
            "[pages, 1]"
        )
    if q_lift.ndim != 3 or c_kv.ndim != 2 or topk_indices.ndim != 2:
        raise MlaDenseWindowError(
            f"q_lift must be [seq, heads, latent], c_kv [slots, latent] and topk_indices "
            f"[seq, cols]; got {tuple(q_lift.shape)}, {tuple(c_kv.shape)} and "
            f"{tuple(topk_indices.shape)}"
        )
    page = int(page_size)
    try:
        staged = _ms._require_paged(c_kv, block_table_row, written, write_offset, page, None)
    except _ms.MlaSparseAttentionError as err:
        raise MlaDenseWindowError(str(err)) from err
    seq, heads, latent = (int(d) for d in q_lift.shape)
    cols = int(topk_indices.shape[1])
    if int(topk_indices.shape[0]) != seq:
        raise MlaDenseWindowError(
            f"topk_indices must have one row per query ({seq}); got {topk_indices.shape[0]}"
        )
    _require_masked(seq, heads, latent, int(c_kv.shape[1]), q_lift.dtype, c_kv.dtype,
                    int(staged), int(pool_size), cols)
    # Checked eagerly only, as the sparse seam checks its indices: a traced call relies on
    # the indexer's contract (every selected pool complete, inside the window, and named
    # once: the top-k's nc_match_replace8 pairs tied scores with distinct positions).
    if values_are_readable(topk_indices):
        lo, hi = int(topk_indices.min()), int(topk_indices.max())
        if lo < -1 or hi >= staged:
            raise MlaDenseWindowError(
                f"every selected row must be in [0, {staged}) or the -1 sentinel; got the "
                f"range [{lo}, {hi}]"
            )
        pool = int(pool_size)
        named = topk_indices.reshape(seq, cols // pool, pool)[:, :, pool - 1]
        ordered = named.sort(dim=1).values
        if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
            raise MlaDenseWindowError(
                "a query names a selected pool twice; the masked kernel counts each pool "
                "a query selects once"
            )
    programs = _masked_programs(seq)
    _count_masked_dispatch()
    overlaid = written is not None and int(written.shape[0]) > 0
    call = wrap_nki(mla_masked_window_kernel)
    if programs == 2:
        call = call[2]
    return call(
        q_lift.contiguous(),
        c_kv.contiguous(),
        block_table_row.contiguous().to(torch.int32),
        topk_indices.contiguous().to(torch.int32),
        written.contiguous() if overlaid else None,
        write_offset.contiguous().to(torch.int32) if overlaid else None,
        float(softmax_scale),
        page,
        int(staged),
        int(pool_size),
        SOURCE_DIGEST,
        STAGING_DIGEST,
    )
