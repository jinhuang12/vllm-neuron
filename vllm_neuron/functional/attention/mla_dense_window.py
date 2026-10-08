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

This module never decides the regime and holds no model dial: the caller passes the
window and each row's causal length. The call site is
``vllm_neuron/model/glm5_next/dsa_dense_window.py``; its kill switch is
``VLLM_NEURON_MLA_DENSE_WINDOW`` (``vllm_neuron/envs.py``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as _ms
from vllm_neuron.utils.neuron_utils import values_are_readable

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
SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)
STAGING_DIGEST = int(_ms.SOURCE_DIGEST)


class MlaDenseWindowError(ValueError):
    """Raised for a call this kernel does not serve. There is no fallback here."""


@dataclass
class _Counters:
    nki_dispatch: int = 0
    two_program_dispatch: int = 0


_COUNTERS = _Counters()


def reset_mla_dense_window_dispatch_counters() -> None:
    """Zero the dispatch counters (outside a trace; the counts are host-side)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.two_program_dispatch = 0


def mla_dense_window_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)``; the fallback count is always 0."""
    return (_COUNTERS.nki_dispatch, 0)


def mla_dense_window_route_counts() -> tuple[int, int]:
    """``(nki_dispatch, two_program_dispatch)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.two_program_dispatch)


@torch._dynamo.assume_constant_result
def _count_dispatch(programs: int) -> None:
    _COUNTERS.nki_dispatch += 1
    if programs == 2:
        _COUNTERS.two_program_dispatch += 1


# Helpers take positional arguments only: the NKI front end drops keyword defaults.
def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _col(parts, dtype):
    """A ``[parts, 1]`` view of a tile one 32-byte line wide."""
    return nl.ndarray((parts, _LINE), dtype=dtype, buffer=nl.sbuf)[:, 0:1]


def _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
              key_rows, r0, rt):
    """One tile of ``rt`` query rows starting at row ``r0``: MM1, softmax, MM2, store."""
    latent = q_hbm.shape[1]
    n_lat = latent // LATENT_TILE
    n_chunks = c_rows.shape[1]
    width = n_chunks * KEY_CHUNK
    kv_dtype = c_rows.dtype

    # ---- the query rows, transposed onto the latent axis --------------------------
    # A PE transpose writes PSUM in its input's dtype and a PSUM write must be whole
    # 4-byte words, so an odd row count of a 2-byte dtype is transposed in fp32 (exact on
    # the PE) and rounded back, losslessly, to its own dtype -- mla_decode's rule.
    q_nat = _sb((ROW_TILE, latent), q_hbm.dtype)
    nisa.dma_copy(dst=q_nat[0:rt, :],
                  src=q_hbm.ap(pattern=[[latent, rt], [1, latent]], offset=r0 * latent))
    q_f = _sb((ROW_TILE, latent), nl.float32)
    nisa.tensor_copy(dst=q_f[0:rt, :], src=q_nat[0:rt, :])
    q_ps = nl.ndarray((LATENT_TILE, n_lat * ROW_TILE), dtype=nl.float32, buffer=nl.psum)
    for li in range(n_lat):
        nisa.nc_transpose(dst=q_ps[:, li * ROW_TILE:li * ROW_TILE + rt],
                          data=q_f[0:rt, li * LATENT_TILE:(li + 1) * LATENT_TILE])
    q_t = _sb((LATENT_TILE, n_lat, ROW_TILE), q_hbm.dtype)
    for li in range(n_lat):
        nisa.tensor_copy(dst=q_t[:, li, 0:rt], src=q_ps[:, li * ROW_TILE:li * ROW_TILE + rt])

    # ---- which window rows each query row may see: j < min(seq_len, key_rows) ------
    lim_i = _col(ROW_TILE, nl.int32)
    nisa.dma_copy(dst=lim_i[0:rt, :], src=limits_hbm.ap(pattern=[[1, rt], [1, 1]], offset=r0))
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
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, rt], [1, latent]], offset=r0 * latent),
                  src=out_sb[0:rt, :])


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
    for ck in range(n_chunks):
        t_ps = nl.ndarray((LATENT_TILE, n_lat * KEY_CHUNK), dtype=kv_dtype, buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_transpose(dst=t_ps[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK],
                              data=c_rows[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE])
        for li in range(n_lat):
            nisa.tensor_copy(dst=c_t[:, li, ck * KEY_CHUNK:(ck + 1) * KEY_CHUNK],
                             src=t_ps[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK])

    # ---- the column index of every window row, the same on every partition -----------
    cols_f = _sb((ROW_TILE, width), nl.float32)
    nisa.iota(dst=cols_f, pattern=[[1, width]], offset=0)

    # Work is dealt round-robin over the programs: the whole tiles of active rows, then
    # the partial tile, then the zero-row chunks. ``program_id`` is a trace-time int (the
    # kernel is traced once per program), so each program keeps only its own jobs.
    n_full = active_rows // ROW_TILE
    tail = active_rows - n_full * ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for qt in nl.affine_range(prg, n_full, n_prgs):
        _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
                  key_rows, qt * ROW_TILE, ROW_TILE)
    job = n_full
    if tail > 0:
        if job % n_prgs == prg:
            _row_tile(q_hbm, limits_hbm, out_hbm, c_t, c_rows, cols_f, softmax_scale,
                      key_rows, n_full * ROW_TILE, tail)
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
    if key_rows < 1 or key_rows > MAX_KEY_ROWS:
        raise MlaDenseWindowError(
            f"the kernel holds the window in SBUF twice, so it reads at most "
            f"{MAX_KEY_ROWS} key rows; got key rows={key_rows}"
        )


def _programs(rows: int) -> int:
    """Two programs on an LNC2 core when the rows span two or more 128-row tiles."""
    # NEURON_LOGICAL_NC_CONFIG is the Neuron runtime's core setting, read here the way
    # every LNC2 kernel in functional/ reads it; it is not a vllm-neuron knob.
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and rows > ROW_TILE:
        return 2
    return 1


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
    if seq_lens.ndim != 1 or int(seq_lens.shape[0]) != seq:
        raise MlaDenseWindowError(
            f"seq_lens must be [seq] = [{seq}], one causal length per query; got "
            f"{tuple(seq_lens.shape)}"
        )
    active = seq if active_rows is None else int(active_rows)
    if not 1 <= active <= seq:
        raise MlaDenseWindowError(
            f"active_rows must be in [1, {seq}], a prefix of the queries; got {active_rows!r}"
        )
    # Checked eagerly only, as the sparse seam checks its indices: a traced call has no
    # values to read, and the kernel clamps each row's length at the staged rows.
    if values_are_readable(seq_lens):
        lo, hi = int(seq_lens[:active].min()), int(seq_lens[:active].max())
        if lo < 1 or hi > staged:
            raise MlaDenseWindowError(
                f"every seq_lens entry must be in [1, {staged}], the window rows this call "
                f"stages; got the range [{lo}, {hi}]"
            )

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

