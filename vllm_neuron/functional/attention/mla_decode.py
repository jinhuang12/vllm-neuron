# SPDX-License-Identifier: Apache-2.0
"""Batched MLA decode attention over a paged latent bank: dense or selected rows.

``T`` query rows per request, ``T = 1`` being the plain decode step and ``T = 1 + k`` a
speculative verify step. Row ``t`` of request ``b`` is row ``b * T + t`` of ``q_lift``
and ``written`` (request-major) and sits at window row ``position[b] + t``; it attends
what a single-token step at that position would attend, so the T-row call equals ``T``
sequential single-row steps bit for bit (``test_mla_decode_trow.py``).

Request ``b`` owns the window its block-table row names (``pages * page_size`` rows
of the bank, in table order), and this step's own latent rows ``written[b * T ..
b * T + T - 1]`` sit at window rows ``position[b] .. position[b] + T - 1``. Two modes:

* **dense** -- row ``t`` attends window rows ``0 .. position[b] + t``. This is the DSA layer's
  short-context regime: while ``seq_len <= index_topk + index_kpool - 1`` the
  indexer's selection keeps every token, so attending the causal prefix directly
  gives the same row set without scoring or selecting anything.
* **selected** -- attend the window rows ``topk_indices[b]`` names; ``-1`` is a
  column that carries no token. Duplicates count once per column, as the sparse
  kernel counts them.

Either way the step's own rows stand in for the bank rows at their positions, so
the result does not depend on whether this step's cache writes have landed: row
``t``'s own row rides the softmax as one more column (``s_self``), and the step's
earlier rows ``0 .. t - 1`` are overlaid onto the window tile at their own columns
before anything is scored, which is what keeps the T-row call bit-equal to the
sequential steps (whose bank held those rows at exactly those columns). With one row
per request the overlay traces nothing.

What changes against ``mla_sparse``: no window is staged. Dense rows are read
page by page straight from the bank -- once per request, shared by its ``T`` rows
-- and selected rows are gathered from the bank by translating each window row
through the request's own block-table row, so the cost follows the rows attended,
and each request reads its own table. Each row's matmuls keep the one-row shapes
(the CPU simulator's matmul rounds by operand shape), so rows run one after the
other on the request's partitions. The query
and the cache are 2-byte floats, so MM1 runs single-pass (exact products, fp32
accumulation). MM2 runs as two single-pass matmuls on a bf16 hi/lo split of the
fp32 probabilities, which carries about 16 significand bits.

Two kernels serve it. One head (the served per-rank shape), a latent of whole
128-column tiles per core and pages of whole 128-row chunks take the key-split
kernel (:func:`_split_serves`): under LNC2 both physical cores take
half of every request's keys, B=1 included, and swap softmax partials (max, exp-sum,
value sum) over ``sendrecv`` to merge them; see :func:`_split_body`. Any other shape
takes the general kernel. Under LNC2 it runs on both physical cores (``[2]`` grid) when
there is work for two (:func:`_programs`): two or more requests split over the cores,
each request whole to one core; one request of ``T >= 2`` rows (the verify step at B=1)
splits its rows instead -- program ``p`` takes rows ``p * ceil(T / 2)`` up to
``min(T, (p + 1) * ceil(T / 2))`` (:func:`_row_share`; 2 + 2 at ``T = 4``), and each
program loads, overlays and transposes the request's window for its own rows. A row's
instruction stream does not depend on the program that runs it, so the two-program
result equals the one-program result bit for bit. A single request of one row runs on
one core.

**DCP mode** (:func:`mla_decode_partial_attention`). Under decode context parallelism
each of ``CP`` ranks holds every ``CP``-th 128-token block of a sequence's latent KV
(its *local window*: the owned blocks in sequence order) and the queries of all
``Hq = CP * H`` heads (the q all-gather). Each rank attends its local window only and
returns the softmax-normalised ``partial [Hq, B, L]`` (head-major) and its log-sum-exp
``lse [Hq, B] = softmax_scale * max + ln(sum)``, both float32; a rank with no column to
attend returns ``partial = 0`` and ``lse =`` :data:`EMPTY_LSE`. The llama3 LSE merge of
the ``CP`` partials (``attention_decode.py``'s DCP path) gives the CP=1 result. The
step's own row is a column on its owner rank only: dense mode takes an ``owner [B]``
input (llama3's ``dcp_active_owner``); selected mode needs none, because its ids are
local rows (-1 where another rank holds the row) and a non-owner's ``position`` is its
owned length, which no local id equals. DCP mode runs on the key-split kernel, which
serves up to :data:`GROUP` heads (:func:`_split_partial_serves`), so one request still
runs on both cores; other geometries are refused. One query row per request: a
speculative decoding step (``T > 1`` rows per request) is refused by name, since its rows
can fall in two ranks' blocks. With DCP off nothing changes: :func:`mla_decode_attention`
launches today's kernels.
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

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest
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
#: The DCP partial's ``lse`` where the rank attends nothing (its ``partial`` is 0). A
#: finite constant, MASKED_SCORE's magnitude, not -inf: a row no rank attends (a padded
#: request) merges as ``lse - logsumexp(lse) = 0`` to a zero result instead of NaN.
EMPTY_LSE = MASKED_SCORE
#: A softmax denominator that holds an attended column is at least 1 -- the max column
#: contributes exp(0) -- and one that holds none is exactly 0 (every masked column's
#: exp underflows to 0.0 under the _MAX_FLOOR floor). Anything at or above this is live.
_LIVE_TOTAL = 0.5



#: This file's content digest, handed to both kernels as a trace-time int. The
#: compiled-kernel cache keys on a kernel's own source and its arguments, and both
#: entry points below are thin wrappers over :func:`_decode_body`: without this, an
#: edit to the body or its helpers would be served the previous build.
SOURCE_DIGEST = source_digest(__file__)


class MlaDecodeAttentionError(ValueError):
    """Raised for a call this kernel does not serve; there is no torch fallback."""


@dataclass
class _MlaDecodeDispatchCounters:
    nki_dispatch: int = 0
    dense_dispatch: int = 0
    selected_dispatch: int = 0
    two_program_dispatch: int = 0
    split_dispatch: int = 0
    split_two_program_dispatch: int = 0
    partial_dispatch: int = 0
    partial_split_dispatch: int = 0


_MLA_DECODE_COUNTERS = _MlaDecodeDispatchCounters()


def reset_mla_decode_dispatch_counters() -> None:
    """Zero every count this module keeps."""
    _MLA_DECODE_COUNTERS.nki_dispatch = 0
    _MLA_DECODE_COUNTERS.dense_dispatch = 0
    _MLA_DECODE_COUNTERS.selected_dispatch = 0
    _MLA_DECODE_COUNTERS.two_program_dispatch = 0
    _MLA_DECODE_COUNTERS.split_dispatch = 0
    _MLA_DECODE_COUNTERS.split_two_program_dispatch = 0
    _MLA_DECODE_COUNTERS.partial_dispatch = 0
    _MLA_DECODE_COUNTERS.partial_split_dispatch = 0


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


def mla_decode_split_counts() -> tuple[int, int]:
    """``(split_dispatch, split_two_program_dispatch)``: calls the key-split kernel served."""
    return (
        _MLA_DECODE_COUNTERS.split_dispatch,
        _MLA_DECODE_COUNTERS.split_two_program_dispatch,
    )


def mla_decode_partial_counts() -> tuple[int, int]:
    """``(partial_dispatch, partial_split_dispatch)``: DCP-mode calls, and those the
    key-split kernel served."""
    return (
        _MLA_DECODE_COUNTERS.partial_dispatch,
        _MLA_DECODE_COUNTERS.partial_split_dispatch,
    )


@torch._dynamo.assume_constant_result
def _count_dispatch(dense: bool, programs: int, split: bool) -> None:
    _MLA_DECODE_COUNTERS.nki_dispatch += 1
    if dense:
        _MLA_DECODE_COUNTERS.dense_dispatch += 1
    else:
        _MLA_DECODE_COUNTERS.selected_dispatch += 1
    if programs == 2:
        _MLA_DECODE_COUNTERS.two_program_dispatch += 1
    if split:
        _MLA_DECODE_COUNTERS.split_dispatch += 1
        if programs == 2:
            _MLA_DECODE_COUNTERS.split_two_program_dispatch += 1



@torch._dynamo.assume_constant_result
def _count_partial(split: bool) -> None:
    _MLA_DECODE_COUNTERS.partial_dispatch += 1
    if split:
        _MLA_DECODE_COUNTERS.partial_split_dispatch += 1

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


def _load_selected_rows(c_rows, bank_hbm, table_hbm, topk_hbm, b, row, page_size):
    """Gather row ``row``'s selected window rows (request ``b``'s table) from the bank.

    Returns ``idx``, the ``[KEY_CHUNK, n_chunks]`` int32 tile of selected window rows,
    laid out as the gathered keys are: column ``j * KEY_CHUNK + r`` on partition ``r``
    of chunk ``j``.
    """
    latent = bank_hbm.shape[1]
    pages = table_hbm.shape[1]
    width = topk_hbm.shape[1]
    n_chunks = width // KEY_CHUNK
    shift = _log2(page_size)
    # idx[r, j] = topk[row, j * 128 + r]: one index per partition per chunk.
    idx = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.dma_copy(dst=idx, src=topk_hbm.ap(pattern=[[1, KEY_CHUNK], [KEY_CHUNK, n_chunks]],
                                           offset=row * width))
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
    bank_row = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.tensor_scalar(dst=bank_row, data=page_id, op0=nl.maximum, operand0=0,
                       op1=nl.multiply, operand1=page_size)
    nisa.tensor_tensor(dst=bank_row, data1=bank_row, data2=in_page, op=nl.add)
    for j in range(n_chunks):
        nisa.dma_copy(dst=c_rows[:, j, :],
                      src=bank_hbm.ap(pattern=[[latent, KEY_CHUNK], [1, latent]],
                                      vector_offset=bank_row[:, j:j + 1], indirect_dim=0))
    return idx


def _overlay_own_rows(c_rows, written_hbm, rel_f, own_rows, first_row):
    """Put the step's rows ``0 .. own_rows - 1`` into ``c_rows`` where the window has them.

    ``rel_f[r, j]`` is the window row that key ``(r, j)`` of ``c_rows`` stands for, minus
    the request's start position (so ``0 .. own_rows - 1`` names one of this step's rows,
    anything else does not). Every key is gathered from ``written`` at its clamped
    relative row and copied in only where the predicate holds, so the addressing never
    depends on which partition a row lands on. ``first_row`` is the request's first row
    of ``written``.
    """
    latent = written_hbm.shape[1]
    n_chunks = c_rows.shape[1]
    low = _sb((KEY_CHUNK, n_chunks), nl.float32)
    nisa.tensor_scalar(dst=low, data=rel_f, op0=nl.greater_equal, operand0=0.0)
    high = _sb((KEY_CHUNK, n_chunks), nl.float32)
    nisa.tensor_scalar(dst=high, data=rel_f, op0=nl.less, operand0=float(own_rows))
    is_own = _sb((KEY_CHUNK, n_chunks), nl.float32)
    nisa.tensor_tensor(dst=is_own, data1=low, data2=high, op=nl.multiply)
    clamped = _sb((KEY_CHUNK, n_chunks), nl.float32)
    nisa.tensor_scalar(dst=clamped, data=rel_f, op0=nl.maximum, operand0=0.0,
                       op1=nl.minimum, operand1=float(own_rows - 1))
    own_idx = _sb((KEY_CHUNK, n_chunks), nl.int32)
    nisa.tensor_copy(dst=own_idx, src=clamped)
    ones = _sb((KEY_CHUNK, latent), nl.float32)
    nisa.memset(dst=ones, value=1.0)
    for j in range(n_chunks):
        gathered = _sb((KEY_CHUNK, latent), c_rows.dtype)
        nisa.dma_copy(dst=gathered,
                      src=written_hbm.ap(pattern=[[latent, KEY_CHUNK], [1, latent]],
                                         offset=first_row * latent,
                                         vector_offset=own_idx[:, j:j + 1], indirect_dim=0))
        mask = _sb((KEY_CHUNK, latent), nl.uint8)
        nisa.tensor_scalar(dst=mask, data=ones, op0=nl.multiply, operand0=is_own[:, j:j + 1])
        nisa.tensor_copy_predicated(dst=c_rows[:, j, :], src=gathered, predicate=mask)


def _transpose_block(c_t, c_rows, t0, tw, n_lat, at):
    """``c_t[:, li, at:at + tw] = c_rows[t0 .. t0 + tw, :, li-th latent tile]`` transposed:
    one ``MOVING_MAX`` block of keys onto the latent axis (one PE transpose per key chunk
    and latent tile, one PSUM copy out per latent tile)."""
    kv_dtype = c_rows.dtype
    for li in range(n_lat):
        t_ps = nl.ndarray((LATENT_TILE, MOVING_MAX), dtype=kv_dtype, buffer=nl.psum)
        for ck in range(tw // KEY_CHUNK):
            nisa.nc_transpose(
                dst=t_ps[:, ck * KEY_CHUNK:(ck + 1) * KEY_CHUNK],
                data=c_rows[:, t0 // KEY_CHUNK + ck,
                            li * LATENT_TILE:(li + 1) * LATENT_TILE])
        nisa.tensor_copy(dst=c_t[:, li, at:at + tw], src=t_ps[:, 0:tw])


def _transpose_keys(c_t, c_rows, width, n_lat):
    """``c_t[:, li, c] = c_rows[c, :, li-th latent tile]``: the whole window onto the latent
    axis, once, for the rows that share it (``rows > 1``)."""
    for t0 in range(0, width, MOVING_MAX):
        tw = min(MOVING_MAX, width - t0)
        _transpose_block(c_t, c_rows, t0, tw, n_lat, t0)


def _row_share(batch, rows, programs, program):
    """``(first, end)``: the rows ``first .. end - 1`` of each of its requests that program
    ``program`` of ``programs`` attends.

    With at least as many requests as programs, the requests split over the programs and
    each program attends every row of its own (``(0, rows)``). One request on two programs
    splits its rows: program ``p`` takes ``ceil(rows / 2)`` of them from row
    ``p * ceil(rows / 2)``, the last share cut at ``rows``.
    """
    if batch >= programs:
        return 0, rows
    share = (rows + programs - 1) // programs
    first = program * share
    end = first + share
    if end > rows:
        end = rows
    return first, end


def _decode_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                 softmax_scale, page_size, dense, rows, out_hbm):
    """``rows`` query rows per request; request ``b``'s rows are ``b * rows + t``.

    Two or more requests split over the programs, each request whole to one program. One
    request on two programs (``batch < n_prgs``) runs on both, each program attending its
    :func:`_row_share` of the rows; the window, the overlay of the step's own rows and the
    transposed keys are each program's own copy, built as the one-program form builds them.
    """
    heads, latent = q_hbm.shape[1], q_hbm.shape[2]
    batch = table_hbm.shape[0]
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
        if rows > 1:
            # The window row each key of ``c_rows`` stands for, keys on the partitions.
            key_row_f = _sb((KEY_CHUNK, n_chunks), nl.float32)
            nisa.iota(dst=key_row_f, pattern=[[KEY_CHUNK, n_chunks]], offset=0,
                      channel_multiplier=1)

    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    first_row, end_row = _row_share(batch, rows, n_prgs, prg)
    req0, req_step = prg, n_prgs
    if batch < n_prgs:
        # One request on both programs: each program takes the request, and its share of
        # the rows below.
        req0, req_step = 0, 1
    for b in nl.affine_range(req0, batch, req_step):
        # ---- this request's start position, on the heads and on the key partitions --
        start_i = _col(heads, nl.int32)
        nisa.dma_copy(dst=start_i, src=pos_hbm.ap(pattern=[[0, heads], [1, 1]], offset=b))
        start_f = _col(heads, nl.float32)
        nisa.tensor_copy(dst=start_f, src=start_i)
        if rows > 1:
            start_key_i = _col(KEY_CHUNK, nl.int32)
            nisa.dma_copy(dst=start_key_i,
                          src=pos_hbm.ap(pattern=[[0, KEY_CHUNK], [1, 1]], offset=b))
            start_key_f = _col(KEY_CHUNK, nl.float32)
            nisa.tensor_copy(dst=start_key_f, src=start_key_i)

        if dense and rows > 1:
            # ---- the window once, shared by the request's rows -----------------------
            c_rows = _sb((KEY_CHUNK, n_chunks, latent), kv_dtype)
            _load_dense_rows(c_rows, bank_hbm, table_hbm, b, page_size)
            rel_f = _sb((KEY_CHUNK, n_chunks), nl.float32)
            nisa.tensor_scalar(dst=rel_f, data=key_row_f, op0=nl.subtract,
                               operand0=start_key_f)
            _overlay_own_rows(c_rows, written_hbm, rel_f, rows, b * rows)
            c_t = _sb((LATENT_TILE, n_lat, width), kv_dtype)
            _transpose_keys(c_t, c_rows, width, n_lat)

        # One row per request (today's served line) keeps the one-row kernel's instruction
        # stream: the row's position is the start itself, the window is loaded after the
        # query and transposed block by block inside the score loop. Every T-row addition
        # below is under ``rows > 1`` / ``t > 0``. ``t`` is the row's index in the request
        # whichever program attends it, so its stream is the one-program stream.
        for t in range(first_row, end_row):
            row = b * rows + t
            if t == 0:
                pos_f = start_f
            else:
                pos_f = _col(heads, nl.float32)
                nisa.tensor_scalar(dst=pos_f, data=start_f, op0=nl.add, operand0=float(t))

            # ---- this row's query and own latent row ----------------------------------
            q_nat = _sb((heads, latent), q_hbm.dtype)
            nisa.dma_copy(dst=q_nat, src=q_hbm.ap(pattern=[[latent, heads], [1, latent]],
                                                  offset=row * heads * latent))
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
                                                      offset=row * latent))
            own_f = _sb((heads, latent), nl.float32)
            nisa.tensor_copy(dst=own_f, src=own)
            prod = _sb((heads, latent), nl.float32)
            nisa.tensor_tensor(dst=prod, data1=q_nat, data2=own, op=nl.multiply)
            s_self = _col(heads, nl.float32)
            nisa.tensor_reduce(dst=s_self, op=nl.add, data=prod, axis=1)

            if dense:
                if rows == 1:
                    # ---- the one row's window, rows on partitions --------------------
                    c_rows = _sb((KEY_CHUNK, n_chunks, latent), kv_dtype)
                    _load_dense_rows(c_rows, bank_hbm, table_hbm, b, page_size)
            else:
                # ---- this row's selected rows, rows on partitions ----------------------
                c_rows = _sb((KEY_CHUNK, n_chunks, latent), kv_dtype)
                idx = _load_selected_rows(c_rows, bank_hbm, table_hbm, topk_hbm, b, row,
                                          page_size)
                if t > 0:
                    idx_key_f = _sb((KEY_CHUNK, n_chunks), nl.float32)
                    nisa.tensor_copy(dst=idx_key_f, src=idx)
                    rel_f = _sb((KEY_CHUNK, n_chunks), nl.float32)
                    nisa.tensor_scalar(dst=rel_f, data=idx_key_f, op0=nl.subtract,
                                       operand0=start_key_f)
                    _overlay_own_rows(c_rows, written_hbm, rel_f, t, b * rows)
                if rows > 1:
                    c_t = _sb((LATENT_TILE, n_lat, width), kv_dtype)
                    _transpose_keys(c_t, c_rows, width, n_lat)

            # ---- which columns carry a token, and how often the own row counts ----------
            valid = _sb((heads, width), nl.float32)
            count = _col(heads, nl.float32)
            if dense:
                nisa.tensor_scalar(dst=valid, data=rows_f, op0=nl.less, operand0=pos_f)
                nisa.memset(dst=count, value=1.0)
            else:
                idx_i = _sb((heads, width), nl.int32)
                nisa.dma_copy(dst=idx_i, src=topk_hbm.ap(pattern=[[0, heads], [1, width]],
                                                         offset=row * width))
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

            # ---- MM1: scores[H, width] = q . c, keys on the latent axis ------------------
            # Only columns that carry a token are copied out of PSUM; the rest keep
            # MASKED_SCORE, so a stale or never-written row in the window changes nothing.
            scores = _sb((heads, width), nl.float32)
            nisa.memset(dst=scores, value=MASKED_SCORE)
            for t0 in range(0, width, MOVING_MAX):
                tw = min(MOVING_MAX, width - t0)
                if rows == 1:
                    # One row: transpose this block here, as the one-row kernel does.
                    c_t = _sb((LATENT_TILE, n_lat, MOVING_MAX), kv_dtype)
                    _transpose_block(c_t, c_rows, t0, tw, n_lat, 0)
                    at = 0
                else:
                    at = t0
                s_ps = nl.ndarray((heads, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
                for li in range(n_lat):
                    nisa.nc_matmul(dst=s_ps[:, 0:tw],
                                   stationary=q_t[:, li * heads:(li + 1) * heads],
                                   moving=c_t[:, li, at:at + tw], accumulate=(li > 0))
                nisa.tensor_copy_predicated(dst=scores[:, t0:t0 + tw], src=s_ps[:, 0:tw],
                                            predicate=pred[:, t0:t0 + tw])

            # ---- one softmax over every column plus this row's own row ----------------
            # s_own = s_self where this row's own row is attended, else MASKED_SCORE.
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
            nisa.tensor_scalar(dst=neg_top, data=top, op0=nl.multiply,
                               operand0=-softmax_scale)
            p = _sb((heads, width), nl.float32)
            col_sum = _col(heads, nl.float32)
            nisa.activation(dst=p, op=nl.exp, data=scores, bias=neg_top, scale=softmax_scale,
                            reduce_op=nl.add, reduce_res=col_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            e_own = _col(heads, nl.float32)
            nisa.activation(dst=e_own, op=nl.exp, data=s_own, bias=neg_top,
                            scale=softmax_scale)
            p_own = _col(heads, nl.float32)
            nisa.tensor_tensor(dst=p_own, data1=e_own, data2=count, op=nl.multiply)
            total = _col(heads, nl.float32)
            nisa.tensor_tensor(dst=total, data1=col_sum, data2=p_own, op=nl.add)
            nisa.tensor_scalar(dst=total, data=total, op0=nl.maximum, operand0=_TOTAL_FLOOR)

            # ---- MM2: out[H, L] = p . c over a bf16 hi/lo split of p -------------------
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

            # ---- normalise, add this row's own row, store ------------------------------
            num = _sb((heads, latent), nl.float32)
            nisa.scalar_tensor_tensor(dst=num, data=own_f, op0=nl.multiply, operand0=p_own,
                                      op1=nl.add, operand1=pv_ps)
            recip = _col(heads, nl.float32)
            nisa.reciprocal(dst=recip, data=total)
            res = _sb((heads, latent), nl.float32)
            nisa.tensor_scalar(dst=res, data=num, op0=nl.multiply, operand0=recip)
            nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                                         offset=row * heads * latent),
                          src=res)


@nki.jit
def mla_decode_dense_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                            softmax_scale, page_size, source_digest):
    """Dense decode attention: row ``t`` of request ``b`` attends window rows
    ``0 .. position[b] + t``.

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    total, heads, latent = q_hbm.shape
    out = nl.ndarray((total, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _decode_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, None,
                 softmax_scale, page_size, True, total // table_hbm.shape[0], out)
    return out


@nki.jit
def mla_decode_selected_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                               topk_hbm, softmax_scale, page_size, source_digest):
    """Selected decode attention: each row attends the window rows its ``topk`` row
    names, ``-1`` masked.

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    total, heads, latent = q_hbm.shape
    out = nl.ndarray((total, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _decode_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                 softmax_scale, page_size, False, total // table_hbm.shape[0], out)
    return out


# ---------------------------------------------------------------------------------- #
# The key-split kernel: every request on both programs.
#
# Program ``c`` of ``P`` attends chunks ``[c * nck, (c + 1) * nck)`` of every request
# (``nck = n_chunks / P``), keys on partitions, and keeps three partials per (request,
# head) row: its max ``M_c``, its exp-sum ``T_c`` and its unnormalised value sum ``A_c``.
# The two programs swap partials (``sendrecv``) and each finishes half of the latent
# columns:
#
#   M = max(M_0, M_1, s_own), w_c = exp(scale * (M_c - M)) / T, out = sum_c w_c A_c + w_o written
#
# Per 128-key chunk the Tensor engine turns the four latent tiles of the rows onto the
# latent axis (MM1 needs the latent on partitions), scores them with the transposed rows
# as the stationary operand and a one-hot slice of the queries as the moving operand (the
# scores land keys-on-partitions, so the softmax works on [128, chunks] tiles instead of
# one 2048-wide row), and later sums the values with the hi/lo probability pairs as a
# stationary operand and the rows as the moving operand, every chunk of the group into
# one [2 * rows, latent] PSUM tile (hi and lo rows summed by one more matmul). The
# softmax of a group's (request, head) rows runs as one set of instructions.
#
# ``H`` heads (the DCP partial's ``Hq``; today's kernel serves one): a request's rows are
# loaded, transposed and multiplied once for all its heads. Its ``H`` queries ride the
# one-hot moving operand ``nck`` columns apart, so one MM1 per chunk scores every head,
# and its ``H`` hi/lo pairs share one MM2 stationary operand per chunk. Score column
# ``(g * H + h) * nck + ck`` holds request ``g``'s head ``h`` on chunk ``ck``: from the
# softmax on, (request, head) row ``v = g * H + h`` is the unit a one-head request is.
# ---------------------------------------------------------------------------------- #

#: (Request, head) rows whose softmax runs as one instruction set, at most. A request's
#: rows stay resident from the score pass to the value pass, so a group is also capped at
#: :data:`_ROW_BUDGET` bytes of rows per partition.
GROUP = 8
_ROW_BUDGET = 64 * 1024


def _split_group(nck: int, latent: int, heads: int) -> int:
    """Requests per group: :data:`GROUP` (request, head) rows, within the row budget."""
    group = _ROW_BUDGET // (nck * latent * 2)
    group = 1 if group < 1 else (GROUP if group > GROUP else group)
    if heads > 1:
        by_heads = GROUP // heads
        group = 1 if by_heads < 1 else (group if group < by_heads else by_heads)
    return group


def _head_pattern(outer, head, inner, heads):
    """An access pattern with a head entry between ``outer`` and ``inner``, dropped at one
    head so the one-head pattern is today's entry for entry."""
    if heads > 1:
        return outer + [head] + inner
    return outer + inner


def _rows_onto_heads(dst, src_hbm, first, requests, heads, row):
    """``dst[g * heads + h] = src_hbm[first + g]`` (rows of ``row`` elements): each
    request's row onto its heads' partitions. One head: one DMA, today's. More heads: one
    DMA per request -- a single DMA whose partitions span (request, head) is refused by
    the compiler ("failed to lower dma_copy") though the simulator runs it."""
    if heads == 1:
        nisa.dma_copy(dst=dst, src=src_hbm.ap(pattern=[[row, requests], [1, row]],
                                              offset=first * row))
    else:
        for g in range(requests):
            nisa.dma_copy(dst=dst[g * heads:(g + 1) * heads, :],
                          src=src_hbm.ap(pattern=[[0, heads], [1, row]],
                                         offset=(first + g) * row))


def _lse_tile(neg_top, total, live, parts):
    """``ln(total) - neg_top`` where ``live`` (0/1) holds, else EMPTY_LSE.

    ``neg_top`` is the ``-softmax_scale * max`` the exponentials were biased with, the
    rounded value itself, so its rounding cancels out of ``exp(lse) * partial`` in the
    merge. The empty rows take EMPTY_LSE by predicated copy, so whatever the logarithm
    gives for their floored denominator never reaches the result.
    """
    ln_total = _col(parts, nl.float32)
    nisa.activation(dst=ln_total, op=nl.log, data=total)
    raw = _col(parts, nl.float32)
    nisa.tensor_tensor(dst=raw, data1=ln_total, data2=neg_top, op=nl.subtract)
    live_u8 = _col(parts, nl.uint8)
    nisa.tensor_copy(dst=live_u8, src=live)
    lse = _col(parts, nl.float32)
    nisa.memset(dst=lse, value=EMPTY_LSE)
    nisa.tensor_copy_predicated(dst=lse, src=raw, predicate=live_u8)
    return lse


def _split_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                softmax_scale, page_size, dense, out_hbm, owner_hbm, lse_hbm):
    """The key-split kernel (see the block comment above).

    ``lse_hbm`` None is today's kernel: ``out_hbm [B, H, L]``. Otherwise the DCP partial:
    ``out_hbm [H, B, L]`` (head-major) and ``lse_hbm [H, B]``, stored by program 0; in dense
    mode the own row counts ``owner_hbm[b]`` (0 or 1) times instead of once. The partial
    computes every value today's kernel does, in the same order, so at one head and every
    owner 1 its ``out`` is today's bit for bit.
    """
    batch, heads, latent = q_hbm.shape
    partial = lse_hbm is not None
    pages = table_hbm.shape[1]
    n_lat = latent // LATENT_TILE
    width = pages * page_size if dense else topk_hbm.shape[1]
    n_chunks = width // KEY_CHUNK
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    nck = n_chunks // n_prgs
    ck0 = prg * nck
    pieces = page_size // KEY_CHUNK
    kv_dtype = bank_hbm.dtype
    # Latent columns this program finishes, and the ones it hands the other program.
    half = latent // n_prgs
    my_c0 = prg * half
    their_c0 = (1 - prg) * half
    shift = _log2(page_size)
    banked = bank_hbm.reshape((bank_hbm.shape[0] // KEY_CHUNK, KEY_CHUNK, latent))
    group = _split_group(nck, latent, heads)

    # ---- constants -------------------------------------------------------------
    ones_col = _sb((KEY_CHUNK, _LINE), nl.float32)
    nisa.memset(dst=ones_col, value=1.0)
    gmax = group if group < batch else batch
    vmax = gmax * heads
    ones_row = _sb((vmax, KEY_CHUNK), nl.float32)
    nisa.memset(dst=ones_row, value=1.0)
    eye_i = _sb((vmax, vmax), nl.float32)
    nisa.iota(dst=eye_i, pattern=[[1, vmax]], offset=0, channel_multiplier=-1)
    eye = _sb((vmax, vmax), nl.float32)
    nisa.tensor_scalar(dst=eye, data=eye_i, op0=nl.equal, operand0=0.0)
    # pair[k, m] = 1 where k // 2 == m: sums MM2's hi and lo rows of each (request, head).
    pair_i = _sb((2 * vmax, vmax), nl.float32)
    nisa.iota(dst=pair_i, pattern=[[-2, vmax]], offset=0, channel_multiplier=1)
    pair_lo = _sb((2 * vmax, vmax), nl.float32)
    nisa.tensor_scalar(dst=pair_lo, data=pair_i, op0=nl.greater_equal, operand0=0.0)
    pair = _sb((2 * vmax, vmax), nl.float32)
    nisa.tensor_scalar(dst=pair, data=pair_i, op0=nl.less_equal, operand0=1.0)
    nisa.tensor_tensor(dst=pair, data1=pair, data2=pair_lo, op=nl.multiply)
    # MM1's moving operands: request g's head-h query in column wide + h * nck of slot g,
    # zeros elsewhere, so the slice [wide - jv, wide - jv + gvn) puts it in score column
    # jv + h * nck alone (jv = g * H * nck + ck). Two copies, by group parity, so one
    # group's MM1 need not wait for the previous group's.
    wide = vmax * nck
    q_hot = []
    for _ in range(2 if batch > group else 1):
        hot = _sb((LATENT_TILE, gmax, n_lat, 2 * wide), q_hbm.dtype)
        nisa.memset(dst=hot, value=0.0)
        q_hot.append(hot)
    if dense:
        # Window row of key slot (r, ck): (ck0 + ck) * 128 + r.
        rows_f = _sb((KEY_CHUNK, nck), nl.float32)
        nisa.iota(dst=rows_f, pattern=[[KEY_CHUNK, nck]], offset=ck0 * KEY_CHUNK,
                  channel_multiplier=1)
        if pieces > 1:
            piece_i = _sb((1, gmax * nck), nl.int32)
            nisa.iota(dst=piece_i, pattern=[[0, gmax], [0, nck // pieces], [1, pieces]],
                      offset=0)

    for b0 in range(0, batch, group):
        gs = group if b0 + group <= batch else batch - b0
        gn = gs * nck
        # (request, head) rows of the group, and their score columns.
        gv = gs * heads
        gvn = gv * nck

        # ---- per-request scalars, queries, own rows --------------------------------
        pos_i = _sb((KEY_CHUNK, gs), nl.int32)
        nisa.dma_copy(dst=pos_i, src=pos_hbm.ap(pattern=[[0, KEY_CHUNK], [1, gs]], offset=b0))
        pos_f = _sb((KEY_CHUNK, gs), nl.float32)
        nisa.tensor_copy(dst=pos_f, src=pos_i)

        q_nat = _sb((gv, latent), q_hbm.dtype)
        nisa.dma_copy(dst=q_nat, src=q_hbm.ap(pattern=[[latent, gv], [1, latent]],
                                              offset=b0 * heads * latent))
        q_f = _sb((gv, latent), nl.float32)
        nisa.tensor_copy(dst=q_f, src=q_nat)
        q_ps = nl.ndarray((LATENT_TILE, n_lat * gv), dtype=nl.float32, buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_transpose(dst=q_ps[:, li * gv:(li + 1) * gv],
                              data=q_f[:, li * LATENT_TILE:(li + 1) * LATENT_TILE])
        q_t = _sb((LATENT_TILE, n_lat * gv), q_hbm.dtype)
        nisa.tensor_copy(dst=q_t, src=q_ps)
        hot = q_hot[(b0 // group) % len(q_hot)]
        if heads == 1:
            hot_q = hot[:, 0:gs, :, wide:wide + 1]
        else:
            hot_q = hot.ap(pattern=[[gmax * n_lat * 2 * wide, LATENT_TILE],
                                    [n_lat * 2 * wide, gs], [2 * wide, n_lat], [nck, heads]],
                           offset=wide)
        nisa.tensor_copy(dst=hot_q,
                         src=q_t.ap(pattern=_head_pattern(
                             [[n_lat * gv, LATENT_TILE], [heads, gs], [gv, n_lat]],
                             [1, heads], [[1, 1]] if heads == 1 else [], heads), offset=0))

        own = _sb((gv, latent), written_hbm.dtype)
        _rows_onto_heads(own, written_hbm, b0, gs, heads, latent)
        own_f = _sb((gv, half), nl.float32)
        nisa.tensor_copy(dst=own_f, src=own[:, my_c0:my_c0 + half])
        prod = _sb((gv, latent), nl.float32)
        nisa.tensor_tensor(dst=prod, data1=q_nat, data2=own, op=nl.multiply)
        s_self = _col(gv, nl.float32)
        nisa.tensor_reduce(dst=s_self, op=nl.add, data=prod, axis=1)

        # ---- where each key slot's row lives -------------------------------------
        if dense:
            # Key slot (r, ck) holds window row (ck0 + ck) * 128 + r: one page piece per
            # chunk. The table entries once, the -1 pad clamped onto page 0 (masked rows).
            npg = nck // pieces
            raw = _sb((1, gs * npg), nl.int32)
            nisa.dma_copy(dst=raw, src=table_hbm.ap(pattern=[[gs * npg, 1], [pages, gs], [1, npg]],
                                                    offset=b0 * pages + ck0 // pieces))
            held = _sb((1, gn), nl.int32)
            nisa.tensor_scalar(dst=held, data=raw.ap(pattern=[[gs * npg, 1], [1, gs * npg],
                                                              [0, pieces]]),
                               op0=nl.maximum, operand0=0, op1=nl.multiply, operand1=pieces)
            if pieces > 1:
                nisa.tensor_tensor(dst=held, data1=held, data2=piece_i[:, 0:gn], op=nl.add)
        else:
            # Key slot (r, ck) of request g holds selected column ck0*128 + r*nck + ck:
            # each partition reads nck contiguous indices.
            idx = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.dma_copy(dst=idx, src=topk_hbm.ap(
                pattern=[[nck, KEY_CHUNK], [width, gs], [1, nck]],
                offset=b0 * width + ck0 * KEY_CHUNK))
            safe = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.tensor_scalar(dst=safe, data=idx, op0=nl.maximum, operand0=0)
            page_no = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.tensor_scalar(dst=page_no, data=safe, op0=nl.right_shift, operand0=shift)
            in_page = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.tensor_scalar(dst=in_page, data=safe, op0=nl.bitwise_and,
                               operand0=page_size - 1)
            # Each request's table row on every partition, then a per-partition gather
            # of entry (g * pages + page_no) -- no DMA per lookup.
            tb = _sb((KEY_CHUNK, gs * pages), nl.int32)
            nisa.dma_copy(dst=tb, src=table_hbm.ap(
                pattern=[[0, KEY_CHUNK], [pages, gs], [1, pages]], offset=b0 * pages))
            base = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.iota(dst=base, pattern=[[pages, gs], [0, nck]], offset=0)
            entry = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.tensor_tensor(dst=entry, data1=page_no, data2=base, op=nl.add)
            entry_u = _sb((KEY_CHUNK, gn), nl.uint32)
            nisa.tensor_copy(dst=entry_u, src=entry)
            page_id = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.nc_n_gather(dst=page_id, data=tb, indices=entry_u)
            row = _sb((KEY_CHUNK, gn), nl.int32)
            nisa.tensor_scalar(dst=row, data=page_id, op0=nl.maximum, operand0=0,
                               op1=nl.multiply, operand1=page_size)
            nisa.tensor_tensor(dst=row, data1=row, data2=in_page, op=nl.add)

        # ---- rows, then MM1: scores keys-on-partitions ---------------------------
        # One accumulation group per PSUM tile throughout: on trn2 several groups in one
        # tile go wrong once other kernels ran before in the graph (device-checked; the
        # simulator and a lone call do not show it). So every chunk's MM1 writes all
        # of s_ps, the one-hot query slice keeping the other columns unchanged.
        s_ps = nl.ndarray((KEY_CHUNK, gvn), dtype=nl.float32, buffer=nl.psum)
        c_rows = []
        for g in range(gs):
            rows_g = _sb((KEY_CHUNK, nck, latent), kv_dtype)
            for ck in range(nck):
                j = g * nck + ck
                if dense:
                    # One page piece per DMA; issuing them from one queue is the bound,
                    # so they rotate over the Sync and Scalar hardware DGE queues and
                    # the GpSimd software DGE.
                    page_src = banked.ap(pattern=[[latent, KEY_CHUNK], [1, latent]], offset=0,
                                         scalar_offset=held[0:1, j:j + 1], indirect_dim=0)
                    if j % 3 == 2:
                        nisa.dma_copy(dst=rows_g[:, ck, :], src=page_src,
                                      dge_mode=nisa.dge_mode.swdge)
                    else:
                        nisa.dma_copy(dst=rows_g[:, ck, :], src=page_src,
                                      dge_mode=nisa.dge_mode.hwdge,
                                      engine=nisa.engine.sync if j % 3 == 0
                                      else nisa.engine.scalar)
                else:
                    nisa.dma_copy(
                        dst=rows_g[:, ck, :],
                        src=bank_hbm.ap(pattern=[[latent, KEY_CHUNK], [1, latent]],
                                        vector_offset=row[:, j:j + 1], indirect_dim=0))
            c_rows.append(rows_g)
            for ck in range(nck):
                j = g * nck + ck
                jv = g * heads * nck + ck
                t_ps = nl.ndarray((LATENT_TILE, n_lat * KEY_CHUNK), dtype=kv_dtype,
                                  buffer=nl.psum)
                for li in range(n_lat):
                    nisa.nc_transpose(
                        dst=t_ps[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK],
                        data=rows_g[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE])
                c_t = _sb((LATENT_TILE, n_lat * KEY_CHUNK), kv_dtype)
                # The Vector engine takes every copy: the Scalar queue issues row loads.
                nisa.tensor_copy(dst=c_t, src=t_ps, engine=nisa.engine.vector)
                for li in range(n_lat):
                    nisa.nc_matmul(dst=s_ps,
                                   stationary=c_t[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK],
                                   moving=hot[:, g, li, wide - jv:wide - jv + gvn],
                                   accumulate=(j > 0 or li > 0))

        # ---- which slots carry a token (per request; every head shares them) -------
        valid = _sb((KEY_CHUNK, gn), nl.float32)
        if dense:
            nisa.tensor_tensor(dst=valid,
                               data1=rows_f.ap(pattern=[[nck, KEY_CHUNK], [0, gs], [1, nck]]),
                               data2=pos_f.ap(pattern=[[gs, KEY_CHUNK], [1, gs], [0, nck]]),
                               op=nl.less)
        else:
            idx_f = _sb((KEY_CHUNK, gn), nl.float32)
            nisa.tensor_copy(dst=idx_f, src=idx)
            live = _sb((KEY_CHUNK, gn), nl.float32)
            nisa.tensor_scalar(dst=live, data=idx_f, op0=nl.greater,
                               operand0=float(SENTINEL_INDEX))
            is_own = _sb((KEY_CHUNK, gn), nl.float32)
            nisa.tensor_tensor(dst=is_own, data1=idx_f,
                               data2=pos_f.ap(pattern=[[gs, KEY_CHUNK], [1, gs], [0, nck]]),
                               op=nl.equal)
            # valid = live and not own = live - is_own (is_own implies live).
            nisa.tensor_tensor(dst=valid, data1=live, data2=is_own, op=nl.subtract)
            own_rows = _sb((KEY_CHUNK, gs), nl.float32)
            nisa.tensor_reduce(dst=own_rows, op=nl.add,
                               data=is_own.reshape((KEY_CHUNK, gs, nck)), axis=2)
        pred = _sb((KEY_CHUNK, gvn), nl.uint8)
        if heads == 1:
            nisa.tensor_copy(dst=pred, src=valid)
        else:
            nisa.tensor_copy(dst=pred.reshape((KEY_CHUNK, gs, heads, nck)),
                             src=valid.ap(pattern=[[gn, KEY_CHUNK], [nck, gs], [0, heads],
                                                   [1, nck]], offset=0))
        scores = _sb((KEY_CHUNK, gvn), nl.float32)
        nisa.memset(dst=scores, value=MASKED_SCORE)
        nisa.tensor_copy_predicated(dst=scores, src=s_ps, predicate=pred)

        # ---- this program's softmax partials, all gv rows at once ----------------
        rmax = _sb((KEY_CHUNK, gv), nl.float32)
        nisa.tensor_reduce(dst=rmax, op=nl.maximum, data=scores.reshape((KEY_CHUNK, gv, nck)),
                           axis=2)
        rm_ps = nl.ndarray((gv, KEY_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=rm_ps, data=rmax)
        m_c = _col(gv, nl.float32)
        nisa.tensor_reduce(dst=m_c, op=nl.maximum, data=rm_ps, axis=1)
        nisa.tensor_scalar(dst=m_c, data=m_c, op0=nl.maximum, operand0=_MAX_FLOOR)
        diag = _sb((gv, gv), nl.float32)
        nisa.tensor_scalar(dst=diag, data=eye[0:gv, 0:gv], op0=nl.multiply, operand0=m_c)
        mb_ps = nl.ndarray((KEY_CHUNK, gv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=mb_ps, stationary=ones_row[0:gv, :], moving=diag,
                       is_stationary_onezero=True)
        diff = _sb((KEY_CHUNK, gvn), nl.float32)
        nisa.tensor_tensor(dst=diff, data1=scores,
                           data2=mb_ps.ap(pattern=[[gv, KEY_CHUNK], [1, gv], [0, nck]]),
                           op=nl.subtract)
        p = _sb((KEY_CHUNK, gvn), nl.float32)
        nisa.activation(dst=p, op=nl.exp, data=diff, scale=softmax_scale)
        rsum = _sb((KEY_CHUNK, gv), nl.float32)
        nisa.tensor_reduce(dst=rsum, op=nl.add, data=p.reshape((KEY_CHUNK, gv, nck)), axis=2)
        sum_ps = nl.ndarray((gv, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sum_ps, stationary=rsum, moving=ones_col[:, 0:1],
                       is_moving_onezero=True)
        # hi = bf16(p), lo = bf16(p - hi), written straight into MM2's stationary
        # operand: slot j = g * nck + ck holds row v = g * H + h's pair in columns 2v,
        # 2v + 1 and zeros elsewhere, so every chunk accumulates into one [2 gv, latent]
        # tile.
        w2 = _sb((KEY_CHUNK, gn, 2 * gv), nl.bfloat16)
        nisa.memset(dst=w2, value=0.0)
        pair_ap = _head_pattern([[gn * 2 * gv, KEY_CHUNK], [nck * 2 * gv + 2 * heads, gs]],
                                [2, heads], [[2 * gv, nck]], heads)
        hi = w2.ap(pattern=pair_ap, offset=0)
        lo = w2.ap(pattern=pair_ap, offset=1)
        p_v = p.reshape(tuple(_head_pattern([KEY_CHUNK, gs], heads, [nck], heads)))
        nisa.tensor_copy(dst=hi, src=p_v)
        hi_f = _sb((KEY_CHUNK, gv, nck), nl.float32)
        hi_f_v = hi_f if heads == 1 else hi_f.reshape((KEY_CHUNK, gs, heads, nck))
        nisa.tensor_copy(dst=hi_f_v, src=hi)
        nisa.tensor_tensor(dst=lo, data1=p_v, data2=hi_f_v, op=nl.subtract)

        # ---- MM2: value sums, (request, head) rows on partitions -----------------
        hl_ps = nl.ndarray((2 * gv, latent), dtype=nl.float32, buffer=nl.psum)
        for g in range(gs):
            for ck in range(nck):
                j = g * nck + ck
                nisa.nc_matmul(dst=hl_ps, stationary=w2[:, j, :], moving=c_rows[g][:, ck, :],
                               accumulate=(j > 0))
        hl = _sb((2 * gv, latent), nl.float32)
        nisa.tensor_copy(dst=hl, src=hl_ps)
        a_ps = nl.ndarray((gv, latent), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=a_ps, stationary=pair[0:2 * gv, 0:gv], moving=hl,
                       is_stationary_onezero=True)

        # ---- partials: [A for the other program's columns | M | T | own count] ----
        cnt_m = _col(gv, nl.float32)
        t_m = _col(gv, nl.float32)
        nisa.tensor_copy(dst=t_m, src=sum_ps)
        if dense and partial:
            # The own row counts on its owner rank only: owner / n_prgs per program.
            owner_i = _col(gv, nl.int32)
            _rows_onto_heads(owner_i, owner_hbm, b0, gs, heads, 1)
            owner_f = _col(gv, nl.float32)
            nisa.tensor_copy(dst=owner_f, src=owner_i)
            nisa.tensor_scalar(dst=cnt_m, data=owner_f, op0=nl.multiply,
                               operand0=1.0 / n_prgs)
        elif dense:
            nisa.memset(dst=cnt_m, value=1.0 / n_prgs)
        else:
            if heads == 1:
                own_rows_v = own_rows
            else:
                own_rows_v = _sb((KEY_CHUNK, gv), nl.float32)
                nisa.tensor_copy(dst=own_rows_v.reshape((KEY_CHUNK, gs, heads)),
                                 src=own_rows.ap(pattern=[[gs, KEY_CHUNK], [1, gs], [0, heads]],
                                                 offset=0))
            own_ps = nl.ndarray((gv, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=own_ps, stationary=own_rows_v, moving=ones_col[:, 0:1],
                           is_moving_onezero=True)
            nisa.tensor_copy(dst=cnt_m, src=own_ps)
        a_mine = _sb((gv, half), nl.float32)
        nisa.tensor_copy(dst=a_mine, src=a_ps[:, my_c0:my_c0 + half])
        if n_prgs == 2:
            mine = _sb((gv, half + _LINE), nl.float32)
            nisa.tensor_copy(dst=mine[:, 0:half], src=a_ps[:, their_c0:their_c0 + half])
            nisa.tensor_copy(dst=mine[:, half:half + 1], src=m_c)
            nisa.tensor_copy(dst=mine[:, half + 1:half + 2], src=t_m)
            nisa.tensor_copy(dst=mine[:, half + 2:half + 3], src=cnt_m)
            theirs = _sb((gv, half + _LINE), nl.float32)
            nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - prg, recv_from_rank=1 - prg,
                          pipe_id=0)
            m_t = theirs[:, half:half + 1]
            count = _col(gv, nl.float32)
            nisa.tensor_tensor(dst=count, data1=cnt_m, data2=theirs[:, half + 2:half + 3],
                               op=nl.add)
        else:
            count = cnt_m

        # ---- merge: both halves and this step's own row --------------------------
        own_on = _col(gv, nl.float32)
        nisa.tensor_scalar(dst=own_on, data=count, op0=nl.minimum, operand0=1.0)
        own_off = _col(gv, nl.float32)
        nisa.tensor_scalar(dst=own_off, data=own_on, op0=nl.subtract, operand0=1.0,
                           op1=nl.multiply, operand1=-MASKED_SCORE)
        s_own = _col(gv, nl.float32)
        nisa.tensor_tensor(dst=s_own, data1=s_self, data2=own_on, op=nl.multiply)
        nisa.tensor_tensor(dst=s_own, data1=s_own, data2=own_off, op=nl.add)
        top = _col(gv, nl.float32)
        nisa.tensor_tensor(dst=top, data1=m_c, data2=s_own, op=nl.maximum)
        if n_prgs == 2:
            nisa.tensor_tensor(dst=top, data1=top, data2=m_t, op=nl.maximum)
        nisa.tensor_scalar(dst=top, data=top, op0=nl.maximum, operand0=_MAX_FLOOR)
        neg_top = _col(gv, nl.float32)
        nisa.tensor_scalar(dst=neg_top, data=top, op0=nl.multiply, operand0=-softmax_scale)
        f_m = _col(gv, nl.float32)
        nisa.activation(dst=f_m, op=nl.exp, data=m_c, bias=neg_top, scale=softmax_scale)
        e_own = _col(gv, nl.float32)
        nisa.activation(dst=e_own, op=nl.exp, data=s_own, bias=neg_top, scale=softmax_scale)
        p_own = _col(gv, nl.float32)
        nisa.tensor_tensor(dst=p_own, data1=e_own, data2=count, op=nl.multiply)
        total = _col(gv, nl.float32)
        nisa.scalar_tensor_tensor(dst=total, data=t_m, op0=nl.multiply, operand0=f_m,
                                  op1=nl.add, operand1=p_own)
        if n_prgs == 2:
            f_t = _col(gv, nl.float32)
            nisa.activation(dst=f_t, op=nl.exp, data=m_t, bias=neg_top, scale=softmax_scale)
            nisa.scalar_tensor_tensor(dst=total, data=theirs[:, half + 1:half + 2],
                                      op0=nl.multiply, operand0=f_t, op1=nl.add,
                                      operand1=total)
        if partial:
            live_rows = _col(gv, nl.float32)
            nisa.tensor_scalar(dst=live_rows, data=total, op0=nl.greater_equal,
                               operand0=_LIVE_TOTAL)
        nisa.tensor_scalar(dst=total, data=total, op0=nl.maximum, operand0=_TOTAL_FLOOR)
        recip = _col(gv, nl.float32)
        nisa.reciprocal(dst=recip, data=total)
        w_m = _col(gv, nl.float32)
        nisa.tensor_tensor(dst=w_m, data1=f_m, data2=recip, op=nl.multiply)
        w_o = _col(gv, nl.float32)
        nisa.tensor_tensor(dst=w_o, data1=p_own, data2=recip, op=nl.multiply)
        res = _sb((gv, half), nl.float32)
        nisa.tensor_scalar(dst=res, data=own_f, op0=nl.multiply, operand0=w_o)
        nisa.scalar_tensor_tensor(dst=res, data=a_mine, op0=nl.multiply, operand0=w_m,
                                  op1=nl.add, operand1=res)
        if n_prgs == 2:
            w_t = _col(gv, nl.float32)
            nisa.tensor_tensor(dst=w_t, data1=f_t, data2=recip, op=nl.multiply)
            nisa.scalar_tensor_tensor(dst=res, data=theirs[:, 0:half], op0=nl.multiply,
                                      operand0=w_t, op1=nl.add, operand1=res)
        if partial:
            # Head-major: request g's heads land batch * latent apart.
            for g in range(gs):
                nisa.dma_copy(dst=out_hbm.ap(pattern=[[batch * latent, heads], [1, half]],
                                             offset=(b0 + g) * latent + my_c0),
                              src=res[g * heads:(g + 1) * heads, :])
            # The lse is program 0's: program 1 normalised its half by the same sum added
            # in its own order, which differs from program 0's by fp32 rounding only.
            if prg == 0:
                lse = _lse_tile(neg_top, total, live_rows, gv)
                for g in range(gs):
                    nisa.dma_copy(dst=lse_hbm.ap(pattern=[[batch, heads], [1, 1]], offset=b0 + g),
                                  src=lse[g * heads:(g + 1) * heads, :])
        else:
            nisa.dma_copy(dst=out_hbm.ap(pattern=[[heads * latent, gv], [1, half]],
                                         offset=b0 * heads * latent + my_c0),
                          src=res)


@nki.jit
def mla_decode_dense_split_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                                  softmax_scale, page_size, source_digest):
    """Dense decode attention, keys split over the programs (one head).

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    out = nl.ndarray((batch, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _split_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, None,
                softmax_scale, page_size, True, out, None, None)
    return out


@nki.jit
def mla_decode_selected_split_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                                     topk_hbm, softmax_scale, page_size, source_digest):
    """Selected decode attention, keys split over the programs (one head).

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    out = nl.ndarray((batch, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _split_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                softmax_scale, page_size, False, out, None, None)
    return out


@nki.jit
def mla_decode_dense_split_partial_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm,
                                          owner_hbm, softmax_scale, page_size, source_digest):
    """DCP dense partial, keys split over the programs, any head count
    :func:`_split_partial_serves` admits: ``(out [H, B, L], lse [H, B])``.

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    out = nl.ndarray((heads, batch, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    lse = nl.ndarray((heads, batch), dtype=nl.float32, buffer=nl.shared_hbm)
    _split_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, None,
                softmax_scale, page_size, True, out, owner_hbm, lse)
    return out, lse


@nki.jit
def mla_decode_selected_split_partial_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm,
                                             written_hbm, topk_hbm, softmax_scale, page_size,
                                             source_digest):
    """DCP selected partial, keys split over the programs, any head count
    :func:`_split_partial_serves` admits: ``(out [H, B, L], lse [H, B])``.

    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    out = nl.ndarray((heads, batch, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    lse = nl.ndarray((heads, batch), dtype=nl.float32, buffer=nl.shared_hbm)
    _split_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                softmax_scale, page_size, False, out, None, lse)
    return out, lse


# ---------------------------------------------------------------------------------- #
# DCP mode: one rank's (partial, lse) on the key-split kernel. Host-side only; the
# kernels are the two split partial entries above.
# ---------------------------------------------------------------------------------- #


def _split_partial_serves(heads: int, latent: int, page: int, width: int,
                          programs: int) -> bool:
    """True when the key-split partial kernel serves the geometry: today's one-head
    geometry, at most :data:`GROUP` heads, and a group's scores in one MM1 PSUM tile.

    Today's kernel keeps its one-head route (:func:`_split_serves`): its multi-head calls
    stay on the general kernel, so their results do not change.
    """
    if not (1 <= heads <= GROUP and _split_serves(1, latent, page, width, programs)):
        return False
    nck = width // KEY_CHUNK // programs
    return _split_group(nck, latent, heads) * heads * nck <= MOVING_MAX


def _require_dcp_operands(q_lift, block_table, owner) -> None:
    if q_lift.ndim == 3 and block_table.ndim == 2 and \
            int(q_lift.shape[0]) != int(block_table.shape[0]):
        raise MlaDecodeAttentionError(
            f"speculative decoding (T > 1 query rows per request) is not supported under "
            f"decode context parallelism: q_lift has {int(q_lift.shape[0])} rows for "
            f"{int(block_table.shape[0])} requests, and DCP mode takes one row per request. "
            f"A step's rows can cross a 128-token block, so two ranks can each own part of "
            f"the step; follow-up: per-row DCP ownership (run_start, run_first, run_rows "
            f"per request) on a T-row key-split kernel"
        )
    batch = int(block_table.shape[0]) if block_table.ndim == 2 else -1
    if owner.ndim != 1 or int(owner.shape[0]) != batch:
        raise MlaDecodeAttentionError(
            f"owner must be [batch] = [{batch}]; got {tuple(owner.shape)}"
        )
    if owner.dtype.is_floating_point or owner.dtype.is_complex:
        raise MlaDecodeAttentionError(f"owner must be bool or integer; got {owner.dtype}")
    if values_are_readable(owner) and bool(((owner != 0) & (owner != 1)).any()):
        raise MlaDecodeAttentionError(
            f"owner must be 0 or 1 per request; got values {sorted(set(owner.tolist()))}"
        )


def _require_dcp_positions(position, owner, window, topk_indices) -> None:
    pos = position.to(torch.int64)
    held = owner.to(torch.int64) != 0
    late = held & (pos >= window)
    if bool(late.any()):
        b = int(late.nonzero()[0, 0])
        raise MlaDecodeAttentionError(
            f"an owner's position must lie inside its local window: request {b} owns this "
            f"step's row at {int(pos[b])} against a window of {window}"
        )
    long = ~held & (pos > window)
    if bool(long.any()):
        b = int(long.nonzero()[0, 0])
        raise MlaDecodeAttentionError(
            f"a non-owner's position is its owned length, at most the window: request {b} "
            f"has {int(pos[b])} against a window of {window}"
        )
    if topk_indices is not None and values_are_readable(topk_indices):
        named = ~held & (topk_indices.to(torch.int64) == pos[:, None]).any(dim=1)
        if bool(named.any()):
            b = int(named.nonzero()[0, 0])
            raise MlaDecodeAttentionError(
                f"a non-owner's selection must not name its position (its owned length "
                f"{int(pos[b])}): the kernel would count request {b}'s own row there"
            )


def mla_decode_partial_attention(q_lift: Tensor, bank: Tensor, block_table: Tensor,
                                 position: Tensor, written: Tensor, owner: Tensor,
                                 softmax_scale: float, page_size: int,
                                 topk_indices: Tensor | None = None
                                 ) -> tuple[Tensor, Tensor]:
    """One DCP rank's decode attention over its local window: ``(partial, lse)``.

    ``partial [Hq, B, L]`` float32 is the softmax-normalised result over the columns this
    rank attends, head-major; ``lse [Hq, B]`` float32 is ``softmax_scale * max + ln(sum)``
    over them, :data:`EMPTY_LSE` (and ``partial`` 0) where there is none. The llama3 LSE
    merge of every rank's pair is the CP=1 :func:`mla_decode_attention` result.

    Args:
        q_lift: ``[B, Hq, L]`` the gathered queries of all ``Hq = CP * H`` heads, one row
            per request (a T-row speculative step is refused).
        bank: ``[slots, L]`` this rank's latent bank.
        block_table: ``[B, local_pages]`` the pages of each request's local window (the
            128-token blocks this rank holds, in sequence order; ``-1`` pads).
        position: ``[B]`` the step's local row where this rank owns it, else the owned
            length (local rows before the step's block).
        written: ``[B, L]`` this step's own latent rows (read where ``owner`` is 1).
        owner: ``[B]`` bool/int, 1 where this rank holds the step's row (llama3's
            ``dcp_active_owner``). Dense mode counts the own row there only; selected
            mode needs no input (a non-owner's local ids never equal its position), so
            ``owner`` there only checks that.
        softmax_scale: positive.
        page_size: rows per page, a trace-time int.
        topk_indices: ``[B, K]`` local window rows (``-1`` where no token or another
            rank holds the row), or None for dense mode.
    """
    _require_dcp_operands(q_lift, block_table, owner)
    # Today's checks, except that a non-owner's position (its owned length) may equal
    # the window: _require sees it one row inside, and the DCP checks below take it.
    window = int(block_table.shape[1]) * int(page_size)
    readable = values_are_readable(position) and values_are_readable(owner)
    checked = position.clamp(max=window - 1) if readable else position
    _require(q_lift, bank, block_table, checked, written, page_size, topk_indices,
             float(softmax_scale))
    if readable:
        _require_dcp_positions(position, owner, window, topk_indices)
    batch, heads, latent = (int(d) for d in q_lift.shape)
    dense = topk_indices is None
    width = window if dense else int(topk_indices.shape[1])
    programs = 2 if lnc_pair() else 1
    if not _split_partial_serves(heads, latent, int(page_size), width, programs):
        raise MlaDecodeAttentionError(
            f"DCP mode runs on the key-split kernel, which does not serve this geometry: "
            f"{heads} heads (at most {GROUP}), latent {latent} (whole "
            f"{LATENT_TILE * programs}-column tiles over {programs} program(s)), page "
            f"{page_size} (whole {KEY_CHUNK}-row chunks), a {width}-column window (whole "
            f"pages per program, at most {_ROW_BUDGET} row bytes per partition)"
        )
    _count_dispatch(dense, programs, True)
    _count_partial(True)
    entry = (mla_decode_dense_split_partial_kernel if dense
             else mla_decode_selected_split_partial_kernel)
    call = wrap_nki(entry)
    if programs == 2:
        call = call[2]
    args = (q_lift.contiguous(), bank.contiguous(),
            block_table.contiguous().to(torch.int32),
            position.contiguous().to(torch.int32), written.contiguous())
    if dense:
        return call(*args, owner.contiguous().to(torch.int32), float(softmax_scale),
                    int(page_size), SOURCE_DIGEST)
    return call(*args, topk_indices.contiguous().to(torch.int32), float(softmax_scale),
                int(page_size), SOURCE_DIGEST)


def _programs(batch: int, rows: int = 1) -> int:
    """Programs of the general kernel: 2 (an LNC2 pair) for two or more requests, which
    split over the pair, or for one request of two or more rows, whose rows split over it
    (:func:`_row_share`); else 1.

    :func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair` refuses a setting other
    than unset, 1 or 2 (``LaunchGridError``).
    """
    if lnc_pair() and (batch >= 2 or rows >= 2):
        return 2
    return 1


def _split_serves(heads: int, latent: int, page: int, width: int, programs: int) -> bool:
    """True when the key-split kernel serves the geometry ``_require`` admitted."""
    chunks = width // KEY_CHUNK
    # One request's rows of one program fit the row budget (2-byte rows).
    return (heads == 1 and page % KEY_CHUNK == 0 and latent % (LATENT_TILE * programs) == 0
            and chunks % programs == 0 and (chunks // programs) % (page // KEY_CHUNK) == 0
            and (chunks // programs) * latent * 2 <= _ROW_BUDGET)


def _require(q_lift, bank, block_table, position, written, page_size, topk_indices,
             softmax_scale) -> None:
    if q_lift.ndim != 3 or bank.ndim != 2:
        raise MlaDecodeAttentionError(
            f"q_lift must be [batch * rows, heads, latent] and bank [slots, latent]; got "
            f"{tuple(q_lift.shape)} and {tuple(bank.shape)}"
        )
    total, heads, latent = (int(d) for d in q_lift.shape)
    if block_table.ndim != 2 or int(block_table.shape[0]) < 1:
        raise MlaDecodeAttentionError(
            f"block_table must be [batch, pages]; got {tuple(block_table.shape)}"
        )
    batch = int(block_table.shape[0])
    if total < batch or total % batch:
        raise MlaDecodeAttentionError(
            f"q_lift must hold a whole number of rows per request: {total} rows do not "
            f"split over block_table's {batch} requests"
        )
    rows = total // batch
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
    if written.dtype != bank.dtype or tuple(written.shape) != (total, latent):
        raise MlaDecodeAttentionError(
            f"written must be [batch * rows, latent] = {(total, latent)} in the bank's dtype; "
            f"got {tuple(written.shape)} {written.dtype}"
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
        if topk_indices.ndim != 2 or int(topk_indices.shape[0]) != total:
            raise MlaDecodeAttentionError(
                f"topk_indices must be [batch * rows, width] = [{total}, width], one row "
                f"per query row; got {tuple(topk_indices.shape)}"
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
        if lo < 0 or hi + rows - 1 >= window:
            raise MlaDecodeAttentionError(
                f"each request's {rows} rows must index its window: positions "
                f"[{lo}, {hi}] + {rows - 1} against {window}"
            )
    if not softmax_scale > 0:
        raise MlaDecodeAttentionError(f"softmax_scale must be positive; got {softmax_scale}")


def _route(batch: int, rows: int, heads: int, latent: int, page: int, width: int,
           dense: bool):
    """The kernel, program count and key-split flag for a call ``_require`` admitted.

    One query row per request (``rows == 1``) at a shape :func:`_split_serves` admits runs
    the key-split kernel, every request on both programs of an LNC2 core. The key-split
    kernel takes exactly one query row per request: its request groups, score columns,
    softmax partials, cross-program exchange and own-row merge are laid out one row per
    request, and ``T`` rows per request would index the table past the batch. A ``T``-row
    step (``rows > 1``, the speculative verify step's ``1 + k`` rows) therefore runs the
    general kernel, which attends each request's ``T`` rows with per-row causal limits;
    under LNC2 it runs on both programs, two or more requests split over them and one
    request's rows split over them (:func:`_programs`, :func:`_row_share`). Tests pin both
    routes at the served shard shape (``test_mla_decode_frontend_compile.py``,
    ``test_mla_decode_trow.py``).
    """
    split_programs = 2 if lnc_pair() else 1
    split = rows == 1 and _split_serves(heads, latent, page, width, split_programs)
    programs = split_programs if split else _programs(batch, rows)
    if split:
        entry = mla_decode_dense_split_kernel if dense else mla_decode_selected_split_kernel
    else:
        entry = mla_decode_dense_kernel if dense else mla_decode_selected_kernel
    return entry, programs, split


def mla_decode_attention(q_lift: Tensor, bank: Tensor, block_table: Tensor,
                         position: Tensor, written: Tensor, softmax_scale: float,
                         page_size: int, topk_indices: Tensor | None = None) -> Tensor:
    """Decode attention for ``B`` requests, ``T`` query rows each. ``[B * T, H, L]`` fp32.

    ``T = q_lift.shape[0] // block_table.shape[0]`` (a trace-time int); row ``t`` of
    request ``b`` is row ``b * T + t`` of ``q_lift``, ``written``, ``topk_indices`` and
    the result, and sits at window row ``position[b] + t``. The one-row call is ``T = 1``.

    Args:
        q_lift: ``[B * T, H, L]`` bf16/fp16, the absorbed queries, request-major.
        bank: ``[slots, L]`` the whole latent bank, same dtype.
        block_table: ``[B, pages]`` int, each request's pages (``-1`` pads).
        position: ``[B]`` int, the window row of each request's first row this step.
        written: ``[B * T, L]`` this step's own latent rows, request-major.
        softmax_scale: positive.
        page_size: rows per page, a trace-time int.
        topk_indices: ``[B * T, K]`` int window rows per query row, ``-1`` for "no
            token", or None for the dense causal prefix ``0 .. position[b] + t``.
    """
    _require(q_lift, bank, block_table, position, written, page_size, topk_indices,
             float(softmax_scale))
    batch = int(block_table.shape[0])
    total, heads, latent = (int(d) for d in q_lift.shape)
    dense = topk_indices is None
    width = int(block_table.shape[1]) * int(page_size) if dense else int(topk_indices.shape[1])
    entry, programs, split = _route(batch, total // batch, heads, latent, int(page_size),
                                    width, dense)
    _count_dispatch(dense, programs, split)
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
    """CPU reference for tests: assemble each window, overlay the step's ``T`` rows at
    ``position .. position + T - 1``, attend each row in fp32 (row ``t`` up to its own
    window row in the dense form; its ``topk_indices`` row in the selected form)."""
    total, heads, latent = q_lift.shape
    batch = int(block_table.shape[0])
    rows = total // batch
    out = torch.empty(total, heads, latent, dtype=torch.float32)
    page = int(page_size)
    for b in range(batch):
        table = block_table[b].to(torch.int64).clamp(min=0)
        window = bank.reshape(-1, page, latent)[table].reshape(-1, latent).to(torch.float32)
        window = window.clone()
        start = int(position[b])
        first = b * rows
        window[start:start + rows] = written[first:first + rows].to(torch.float32)
        for t in range(rows):
            row = first + t
            pos = start + t
            if topk_indices is None:
                idx = torch.arange(pos + 1)
                keep = torch.ones(pos + 1, dtype=torch.bool)
            else:
                idx = topk_indices[row].to(torch.int64)
                keep = idx >= 0
                idx = idx.clamp(min=0)
            gathered = window[idx]
            scores = q_lift[row].to(torch.float32) @ gathered.t()
            scores = scores.masked_fill(~keep, float("-inf")) * softmax_scale
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
            out[row] = weights @ gathered
    return out
