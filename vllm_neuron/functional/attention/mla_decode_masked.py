# SPDX-License-Identifier: Apache-2.0
"""Selecting-step forms of the batched MLA decode attention that :mod:`mla_decode` does
not serve.

:mod:`mla_decode`'s selected mode gathers each query row's ``K`` selected window rows on
its own. When a request's window holds fewer rows than its ``T`` query rows gather
between them (``T * K``), reading the window once per request and masking, per row, the
columns outside that row's selection attends the same rows from fewer bytes. A one-row
step's ``K`` rows split over an LNC2 pair only at an even count of 128-row chunks, and
the served ``K`` holds an odd count. The kernels of both forms live in this file, beside
:mod:`mla_decode` and not in it, and reuse its helpers by import.

**The selection as a pool mask.** The DSA selection picks whole pools of ``pool`` tokens:
the ``k`` best of each row's bounded candidate pool scores (``[B * T, C]`` fp32, ``C *
pool`` the window), ties to the lowest candidates. The kernel selects them itself, with
:mod:`decode_select`'s search on the request's rows (the pools
``dsa_decode_select_mask_torch_oracle`` marks 1.0), while the window's pages are in
flight. Row ``t`` of request ``b`` sits at position ``p = position[b] + t`` and attends
window column ``j`` when ``j <= p`` and either

* the pool ``j // pool`` is selected for the row, or
* ``j`` is a token of the row's incomplete pool, ``j >= (p + 1) & -pool`` (the tail
  ``dsa_index_expand`` appends after the selected pools' tokens).

That is the token set ``dsa_decode_select`` lists for the row, each token once (a
selected pool is complete, so it never holds a tail token). The step's own rows stand in
at their columns as in :mod:`mla_decode`: column ``position[b] + t'`` of row ``t``
(``t' <= t``) reads ``written[b * T + t']``, never the bank, and counts when the same
rule admits it.

**The key-split form** (:func:`mla_decode_masked_attention`). Program ``c`` of ``P`` (the
LNC2 pair, or one program) reads chunks ``[c * n, (c + 1) * n)`` of the request's window
(``n = window / 128 / P``) with one page DMA per chunk, keys on partitions, and scores all
``T`` rows against them in one instruction stream: per chunk and latent tile, one matmul
with the transposed rows as the stationary operand and the ``T`` queries as the moving
one, each in its own column of a zero-padded operand, so every chunk writes the whole
``[128, T * n]`` score tile (one accumulation group per PSUM tile). The pool mask reaches
the key partitions through one matmul by the 0/1 pool-to-token matrix. Each program
keeps, per row, the softmax max, exp-sum and value sum of its keys; the programs swap
them over ``sendrecv`` and each finishes half of the latent columns, adding the step's
own rows as up to ``T`` more columns per row. The window columns the own rows stand on
are left to that merge, so a program's keys count only columns before
``position[b]``.

The scores take single-pass bf16 products with fp32 sums and the value sum a bf16 hi/lo
split of the fp32 probabilities, as in :mod:`mla_decode`; each program's half of the keys
runs under its own max and the merge rescales each once, so the result differs from the
selected kernel by fp32 summation order (``test_mla_decode_masked.py`` holds it to the
oracle at relative L2 ``2e-5``).

**The one-row key-split form at an odd chunk count**
(:func:`mla_decode_padded_split_attention`). A selecting step of one row per request
takes ``K`` token indices, ``K / 128`` chunks; the served ``K`` (``k`` pools and one tail
chunk) is an odd count, which :mod:`mla_decode`'s key-split kernel does not split over
the pair, so it would run the general kernel on one program. This kernel copies the
indices into a private scratch of whole chunks per program, ``-1`` (no token) past
them, and runs :mod:`mla_decode`'s key-split body on it: the same token set, one more
chunk of masked keys.

The file is kept out of every other graph's compile cache key
(:mod:`vllm_neuron.compile_cache_key` folds into a graph's key the kernel files the
graph's kernels reach), and three rules keep it so
(``test_mla_decode_masked_keys.py`` pins them):

* no other digested file imports it, so its source reaches only the graphs that call
  its kernels;
* it rebinds no attribute or item of a module it imports: every graph that reaches the
  patched module would fold this file;
* it imports nothing dynamically and nothing of ``nkilib``: the first makes every key
  fold the whole-package digest, the second makes every graph that calls that ``nkilib``
  kernel fold this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_decode
from vllm_neuron.functional.attention.mla_decode import (
    _LINE,
    _MAX_FLOOR,
    _ROW_BUDGET,
    _TOTAL_FLOOR,
    KEY_CHUNK,
    LATENT_TILE,
    MASKED_SCORE,
    MOVING_MAX,
    SENTINEL_INDEX,
    MlaDecodeAttentionError,
    _col,
    _log2,
    _require,
    _sb,
    _split_body,
    _split_serves,
)
from vllm_neuron.functional.dsa import decode_select
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.decode_select import (
    _fold_copy,
    _request_matrices,
    _select_mask,
    fold_layout,
    select_fold_cap,
    tile_requests,
)
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest

#: This file's, :mod:`mla_decode`'s and :mod:`decode_select`'s content, handed to the
#: kernel as a trace-time int: the compiled-kernel cache keys on a kernel's own source and
#: its arguments, and the kernel's body reuses the two modules' helpers.
SOURCE_DIGEST = source_digest(__file__, mla_decode.__file__, decode_select.__file__)


@dataclass
class _MaskedDispatchCounters:
    nki_dispatch: int = 0
    two_program_dispatch: int = 0
    padded_split_dispatch: int = 0


_COUNTERS = _MaskedDispatchCounters()


def reset_mla_decode_masked_dispatch_counters() -> None:
    """Zero every count this module keeps."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.two_program_dispatch = 0
    _COUNTERS.padded_split_dispatch = 0


def mla_decode_masked_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)``; ``torch_fallback`` is always 0: a call this
    kernel does not serve is refused."""
    return (_COUNTERS.nki_dispatch, 0)


def mla_decode_masked_two_program_count() -> int:
    """Calls that ran on both programs of an LNC2 pair since the last reset."""
    return _COUNTERS.two_program_dispatch


def mla_decode_padded_split_count() -> int:
    """:func:`mla_decode_padded_split_attention` calls since the last reset (each runs on
    both programs of an LNC2 pair)."""
    return _COUNTERS.padded_split_dispatch


@torch._dynamo.assume_constant_result
def _count_dispatch(programs: int) -> None:
    _COUNTERS.nki_dispatch += 1
    if programs == 2:
        _COUNTERS.two_program_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_padded_split() -> None:
    _COUNTERS.padded_split_dispatch += 1


def _iota_f(shape, pattern, offset, channel_multiplier):
    out = _sb(shape, nl.float32)
    nisa.iota(dst=out, pattern=pattern, offset=offset, channel_multiplier=channel_multiplier)
    return out


def _between(data, low, high):
    """``1.0`` where ``low <= data <= high`` (fp32 tile of ``data``'s shape), else 0."""
    lo = _sb(data.shape, nl.float32)
    nisa.tensor_scalar(dst=lo, data=data, op0=nl.greater_equal, operand0=float(low))
    out = _sb(data.shape, nl.float32)
    nisa.scalar_tensor_tensor(dst=out, data=data, op0=nl.less_equal, operand0=float(high),
                              op1=nl.multiply, operand1=lo)
    return out


def _latent_major(nat, rows, n_lat):
    """``[rows, n_lat * 128]`` onto the latent axis: column ``li * rows + t`` of the result
    is row ``t``'s latent tile ``li``. Through the PE in fp32 (a PSUM write is whole
    4-byte words) and back to ``nat``'s dtype, exact both ways."""
    nat_f = _sb((rows, n_lat * LATENT_TILE), nl.float32)
    nisa.tensor_copy(dst=nat_f, src=nat)
    t_ps = nl.ndarray((LATENT_TILE, n_lat * rows), dtype=nl.float32, buffer=nl.psum)
    for li in range(n_lat):
        nisa.nc_transpose(dst=t_ps[:, li * rows:(li + 1) * rows],
                          data=nat_f[:, li * LATENT_TILE:(li + 1) * LATENT_TILE])
    out = _sb((LATENT_TILE, n_lat * rows), nat.dtype)
    nisa.tensor_copy(dst=out, src=t_ps)
    return out


def _masked_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, scores_hbm,
                 softmax_scale, page_size, pool_size, select_k, fold_cap, out_hbm):
    total, heads, latent = q_hbm.shape
    batch, pages = table_hbm.shape
    rows = total // batch
    cands = scores_hbm.shape[1]
    # The rows' pool masks, selected below and read back as the window's and the own
    # rows' masks; each program selects every row of the request on its own copy.
    sel_hbm = nl.ndarray((total, cands), dtype=nl.float32, buffer=nl.private_hbm)
    fold, chunk, request_major = fold_layout(rows, cands, fold_cap)
    parts = fold * rows
    n_lat = latent // LATENT_TILE
    n_chunks = pages * page_size // KEY_CHUNK
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    nck = n_chunks // n_prgs
    ck0 = prg * nck
    pieces = page_size // KEY_CHUNK
    npg = nck // pieces
    kv_dtype = bank_hbm.dtype
    half = latent // n_prgs
    my_c0 = prg * half
    their_c0 = (1 - prg) * half
    banked = bank_hbm.reshape((bank_hbm.shape[0] // KEY_CHUNK, KEY_CHUNK, latent))
    per_chunk = KEY_CHUNK // pool_size
    shift = _log2(pool_size)
    # Score column t * nck + ck is row t against key chunk ck.
    wide = rows * nck
    # Pools the step's own rows can stand on, from the first one's: they are read as
    # one run, its start clamped so the run stays inside the request's mask row.
    span = (rows - 1) // pool_size + 2
    sel_flat = sel_hbm.reshape((total * cands, 1))

    # ---- constants -----------------------------------------------------------------
    ones_col = _sb((KEY_CHUNK, _LINE), nl.float32)
    nisa.memset(dst=ones_col, value=1.0)
    ones_row = _sb((rows, KEY_CHUNK), nl.float32)
    nisa.memset(dst=ones_row, value=1.0)
    # eye[t, u] = (t == u); pair[k, t] = (k // 2 == t) sums MM2's hi and lo rows.
    eye = _between(_iota_f((rows, rows), [[1, rows]], 0, -1), 0, 0)
    pair = _between(_iota_f((2 * rows, rows), [[-2, rows]], 0, 1), 0, 1)
    # upto[t, u] = (u <= t): row t's own columns are rows 0 .. t of the step.
    upto = _between(_iota_f((rows, rows), [[1, rows]], 0, -1), -rows, 0)
    # spread[q, r] = (r // pool == q): pool q of a key chunk onto its pool tokens.
    spread_f = _between(_iota_f((per_chunk, KEY_CHUNK), [[1, KEY_CHUNK]], 0, -pool_size),
                        0, pool_size - 1)
    spread = _sb((per_chunk, KEY_CHUNK), nl.bfloat16)
    nisa.tensor_copy(dst=spread, src=spread_f)
    # Window row of key (r, ck): (ck0 + ck) * 128 + r.
    key_row = _iota_f((KEY_CHUNK, nck), [[KEY_CHUNK, nck]], ck0 * KEY_CHUNK, 1)
    # Row t's length offset t + 1, on the key partitions and on the rows.
    step_key = _sb((KEY_CHUNK, rows), nl.int32)
    nisa.iota(dst=step_key, pattern=[[1, rows]], offset=1, channel_multiplier=0)
    step_row = _sb((rows, 1), nl.int32)
    nisa.iota(dst=step_row, pattern=[[0, 1]], offset=1, channel_multiplier=1)
    # own_off[t, u] = u: the own column u of every row.
    own_off = _sb((rows, rows), nl.int32)
    nisa.iota(dst=own_off, pattern=[[1, rows]], offset=0, channel_multiplier=0)
    # The queries, each in its own column of a zero operand: the slice
    # [wide - ck, 2 * wide - ck) puts row t in score column t * nck + ck alone.
    hot = _sb((LATENT_TILE, n_lat, 2 * wide), q_hbm.dtype)
    nisa.memset(dst=hot, value=0.0)
    same, below = _request_matrices(rows, fold, request_major)

    for b in range(batch):
        first = b * rows
        # ---- the rows' pool scores, in the selection's fold layout ------------------
        # First on their DMA queue, so the search below runs while the pages arrive.
        scored = _sb((parts, chunk), nl.float32)
        if cands < fold * chunk:
            nisa.memset(dst=scored, value=BOUND_FILL)
        _fold_copy(scores_hbm, scored, first, rows, cands, chunk, fold, request_major, True)
        # ---- the request's start, its rows' tail starts ----------------------------
        start_i = _col(KEY_CHUNK, nl.int32)
        nisa.dma_copy(dst=start_i, src=pos_hbm.ap(pattern=[[0, KEY_CHUNK], [1, 1]],
                                                  offset=b))
        start_f = _col(KEY_CHUNK, nl.float32)
        nisa.tensor_copy(dst=start_f, src=start_i)
        # Row t's incomplete pool starts at (start + t + 1) & -pool.
        open_key_i = _sb((KEY_CHUNK, rows), nl.int32)
        nisa.tensor_scalar(dst=open_key_i, data=step_key, op0=nl.add, operand0=start_f)
        nisa.tensor_scalar(dst=open_key_i, data=open_key_i, op0=nl.bitwise_and,
                           operand0=-pool_size)
        open_key = _sb((KEY_CHUNK, rows), nl.float32)
        nisa.tensor_copy(dst=open_key, src=open_key_i)

        # ---- queries and own rows, latent on the partitions -------------------------
        q_nat = _sb((rows, latent), q_hbm.dtype)
        nisa.dma_copy(dst=q_nat, src=q_hbm.ap(pattern=[[heads * latent, rows], [1, latent]],
                                              offset=first * heads * latent))
        own = _sb((rows, latent), written_hbm.dtype)
        nisa.dma_copy(dst=own, src=written_hbm.ap(pattern=[[latent, rows], [1, latent]],
                                                  offset=first * latent))
        q_t = _latent_major(q_nat, rows, n_lat)
        own_t = _latent_major(own, rows, n_lat)
        nisa.tensor_copy(dst=hot.ap(pattern=[[n_lat * 2 * wide, LATENT_TILE], [2 * wide, n_lat],
                                             [nck, rows]], offset=wide),
                         src=q_t.ap(pattern=[[n_lat * rows, LATENT_TILE], [rows, n_lat],
                                             [1, rows]], offset=0))
        # own_score[t, u] = q_t . written_u.
        os_ps = nl.ndarray((rows, rows), dtype=nl.float32, buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_matmul(dst=os_ps, stationary=q_t[:, li * rows:(li + 1) * rows],
                           moving=own_t[:, li * rows:(li + 1) * rows], accumulate=(li > 0))

        # ---- this program's window rows, one page piece per DMA ---------------------
        raw = _sb((1, npg), nl.int32)
        nisa.dma_copy(dst=raw, src=table_hbm.ap(pattern=[[npg, 1], [1, npg]],
                                                offset=b * pages + ck0 // pieces))
        held = _sb((1, nck), nl.int32)
        # The -1 pad clamped onto page 0: its rows lie past the start and are masked.
        nisa.tensor_scalar(dst=held, data=raw.ap(pattern=[[npg, 1], [1, npg], [0, pieces]]),
                           op0=nl.maximum, operand0=0, op1=nl.multiply, operand1=pieces)
        if pieces > 1:
            piece_i = _sb((1, nck), nl.int32)
            nisa.iota(dst=piece_i, pattern=[[0, npg], [1, pieces]], offset=0)
            nisa.tensor_tensor(dst=held, data1=held, data2=piece_i, op=nl.add)
        c_rows = _sb((KEY_CHUNK, nck, latent), kv_dtype)
        for ck in range(nck):
            src = banked.ap(pattern=[[latent, KEY_CHUNK], [1, latent]], offset=0,
                            scalar_offset=held[0:1, ck:ck + 1], indirect_dim=0)
            # Issuing the page DMAs from one queue is the bound, so they rotate over
            # the Sync and Scalar hardware DGE queues and the GpSimd software DGE.
            if ck % 3 == 2:
                nisa.dma_copy(dst=c_rows[:, ck, :], src=src, dge_mode=nisa.dge_mode.swdge)
            else:
                nisa.dma_copy(dst=c_rows[:, ck, :], src=src, dge_mode=nisa.dge_mode.hwdge,
                              engine=nisa.engine.sync if ck % 3 == 0 else nisa.engine.scalar)

        # ---- the rows' selections, onto the scratch ---------------------------------
        picked = _select_mask(scored, same, below, parts, chunk, select_k)
        _fold_copy(sel_hbm, picked, first, rows, cands, chunk, fold, request_major, False)

        # ---- MM1: every row against every key chunk, keys on the partitions ---------
        s_ps = nl.ndarray((KEY_CHUNK, wide), dtype=nl.float32, buffer=nl.psum)
        for ck in range(nck):
            t_ps = nl.ndarray((LATENT_TILE, n_lat * KEY_CHUNK), dtype=kv_dtype,
                              buffer=nl.psum)
            for li in range(n_lat):
                nisa.nc_transpose(dst=t_ps[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK],
                                  data=c_rows[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE])
            c_t = _sb((LATENT_TILE, n_lat * KEY_CHUNK), kv_dtype)
            nisa.tensor_copy(dst=c_t, src=t_ps, engine=nisa.engine.vector)
            for li in range(n_lat):
                nisa.nc_matmul(dst=s_ps, stationary=c_t[:, li * KEY_CHUNK:(li + 1) * KEY_CHUNK],
                               moving=hot[:, li, wide - ck:2 * wide - ck],
                               accumulate=(ck > 0 or li > 0))

        # ---- the rows' pool masks on the key partitions ----------------------------
        # sel_k[r, ck * rows + t] = row t's mask at pool (ck0 + ck) * per_chunk + r // pool.
        sel_r = _sb((rows, nck * per_chunk), nl.float32)
        nisa.dma_copy(dst=sel_r, src=sel_hbm.ap(pattern=[[cands, rows], [1, nck * per_chunk]],
                                                offset=first * cands + ck0 * per_chunk))
        sel_ps = nl.ndarray((per_chunk, nck * rows), dtype=nl.float32, buffer=nl.psum)
        for ck in range(nck):
            nisa.nc_transpose(dst=sel_ps[:, ck * rows:(ck + 1) * rows],
                              data=sel_r[:, ck * per_chunk:(ck + 1) * per_chunk])
        sel_q = _sb((per_chunk, nck * rows), nl.bfloat16)
        nisa.tensor_copy(dst=sel_q, src=sel_ps)
        selk_ps = nl.ndarray((KEY_CHUNK, nck * rows), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=selk_ps, stationary=spread, moving=sel_q)
        sel_k = _sb((KEY_CHUNK, nck * rows), nl.float32)
        nisa.tensor_copy(dst=sel_k, src=selk_ps)

        # ---- which window columns each row attends ---------------------------------
        # Before the step's own rows (they join in the merge), and a selected pool or
        # the row's incomplete pool.
        before = _sb((KEY_CHUNK, nck), nl.float32)
        nisa.tensor_scalar(dst=before, data=key_row, op0=nl.less, operand0=start_f)
        in_tail = _sb((KEY_CHUNK, wide), nl.float32)
        nisa.tensor_tensor(dst=in_tail,
                           data1=key_row.ap(pattern=[[nck, KEY_CHUNK], [0, rows], [1, nck]]),
                           data2=open_key.ap(pattern=[[rows, KEY_CHUNK], [1, rows], [0, nck]]),
                           op=nl.greater_equal)
        hit = _sb((KEY_CHUNK, wide), nl.float32)
        nisa.tensor_tensor(dst=hit, data1=in_tail,
                           data2=sel_k.ap(pattern=[[nck * rows, KEY_CHUNK], [1, rows],
                                                   [rows, nck]]),
                           op=nl.maximum)
        valid = _sb((KEY_CHUNK, wide), nl.float32)
        nisa.tensor_tensor(dst=valid, data1=hit,
                           data2=before.ap(pattern=[[nck, KEY_CHUNK], [0, rows], [1, nck]]),
                           op=nl.multiply)
        pred = _sb((KEY_CHUNK, wide), nl.uint8)
        nisa.tensor_copy(dst=pred, src=valid)
        scores = _sb((KEY_CHUNK, wide), nl.float32)
        nisa.memset(dst=scores, value=MASKED_SCORE)
        nisa.tensor_copy_predicated(dst=scores, src=s_ps, predicate=pred)

        # ---- this program's softmax partials, every row at once ---------------------
        rmax = _sb((KEY_CHUNK, rows), nl.float32)
        nisa.tensor_reduce(dst=rmax, op=nl.maximum,
                           data=scores.reshape((KEY_CHUNK, rows, nck)), axis=2)
        rm_ps = nl.ndarray((rows, KEY_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=rm_ps, data=rmax)
        m_c = _col(rows, nl.float32)
        nisa.tensor_reduce(dst=m_c, op=nl.maximum, data=rm_ps, axis=1)
        nisa.tensor_scalar(dst=m_c, data=m_c, op0=nl.maximum, operand0=_MAX_FLOOR)
        diag = _sb((rows, rows), nl.float32)
        nisa.tensor_scalar(dst=diag, data=eye, op0=nl.multiply, operand0=m_c)
        mb_ps = nl.ndarray((KEY_CHUNK, rows), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=mb_ps, stationary=ones_row, moving=diag,
                       is_stationary_onezero=True)
        diff = _sb((KEY_CHUNK, wide), nl.float32)
        nisa.tensor_tensor(dst=diff, data1=scores,
                           data2=mb_ps.ap(pattern=[[rows, KEY_CHUNK], [1, rows], [0, nck]]),
                           op=nl.subtract)
        p = _sb((KEY_CHUNK, wide), nl.float32)
        nisa.activation(dst=p, op=nl.exp, data=diff, scale=softmax_scale)
        rsum = _sb((KEY_CHUNK, rows), nl.float32)
        nisa.tensor_reduce(dst=rsum, op=nl.add, data=p.reshape((KEY_CHUNK, rows, nck)),
                           axis=2)
        sum_ps = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sum_ps, stationary=rsum, moving=ones_col[:, 0:1],
                       is_moving_onezero=True)
        # hi = bf16(p), lo = bf16(p - hi), straight into MM2's stationary operand:
        # chunk ck holds row t's pair in columns 2t, 2t + 1.
        w2 = _sb((KEY_CHUNK, nck, 2 * rows), nl.bfloat16)
        hi = w2.ap(pattern=[[nck * 2 * rows, KEY_CHUNK], [2, rows], [2 * rows, nck]], offset=0)
        lo = w2.ap(pattern=[[nck * 2 * rows, KEY_CHUNK], [2, rows], [2 * rows, nck]], offset=1)
        p_v = p.reshape((KEY_CHUNK, rows, nck))
        nisa.tensor_copy(dst=hi, src=p_v)
        hi_f = _sb((KEY_CHUNK, rows, nck), nl.float32)
        nisa.tensor_copy(dst=hi_f, src=hi)
        nisa.tensor_tensor(dst=lo, data1=p_v, data2=hi_f, op=nl.subtract)

        # ---- MM2: value sums, rows on the partitions --------------------------------
        hl_ps = nl.ndarray((2 * rows, latent), dtype=nl.float32, buffer=nl.psum)
        for ck in range(nck):
            nisa.nc_matmul(dst=hl_ps, stationary=w2[:, ck, :], moving=c_rows[:, ck, :],
                           accumulate=(ck > 0))
        hl = _sb((2 * rows, latent), nl.float32)
        nisa.tensor_copy(dst=hl, src=hl_ps)
        a_ps = nl.ndarray((rows, latent), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=a_ps, stationary=pair, moving=hl, is_stationary_onezero=True)

        # ---- the own columns row t attends: rows u <= t of the step, by the rule ------
        # Row t at start + t, own row u at start + u: u counts when its pool is selected
        # for row t or it is a token of row t's incomplete pool.
        row_start_i = _col(rows, nl.int32)
        nisa.tensor_scalar(dst=row_start_i, data=step_row, op0=nl.add,
                           operand0=start_f[0:rows])
        nisa.tensor_scalar(dst=row_start_i, data=row_start_i, op0=nl.bitwise_and,
                           operand0=-pool_size)
        row_open = _col(rows, nl.float32)
        nisa.tensor_copy(dst=row_open, src=row_start_i)
        own_pos_i = _sb((rows, rows), nl.int32)
        nisa.tensor_scalar(dst=own_pos_i, data=own_off, op0=nl.add, operand0=start_f[0:rows])
        own_pos = _sb((rows, rows), nl.float32)
        nisa.tensor_copy(dst=own_pos, src=own_pos_i)
        in_open = _sb((rows, rows), nl.float32)
        nisa.tensor_scalar(dst=in_open, data=own_pos, op0=nl.greater_equal, operand0=row_open)
        # The masks of the own rows' pools: one run from the first own pool, clamped.
        run0 = _col(rows, nl.int32)
        nisa.tensor_scalar(dst=run0, data=start_i[0:rows], op0=nl.right_shift,
                           operand0=shift)
        nisa.tensor_scalar(dst=run0, data=run0, op0=nl.minimum, operand0=cands - span)
        own_sel = _sb((rows, span), nl.float32)
        nisa.dma_copy(dst=own_sel, src=sel_flat.ap(pattern=[[cands, rows], [1, span]],
                                                   offset=first * cands,
                                                   scalar_offset=run0[0:1, 0:1],
                                                   indirect_dim=0))
        run0_f = _col(rows, nl.float32)
        nisa.tensor_copy(dst=run0_f, src=run0)
        own_pool_i = _sb((rows, rows), nl.int32)
        nisa.tensor_scalar(dst=own_pool_i, data=own_pos_i, op0=nl.right_shift, operand0=shift)
        own_at = _sb((rows, rows), nl.float32)
        nisa.tensor_copy(dst=own_at, src=own_pool_i)
        nisa.tensor_scalar(dst=own_at, data=own_at, op0=nl.subtract, operand0=run0_f)
        att = _sb((rows, rows), nl.float32)
        nisa.memset(dst=att, value=0.0)
        for i in range(span):
            at_i = _sb((rows, rows), nl.float32)
            nisa.tensor_scalar(dst=at_i, data=own_at, op0=nl.equal, operand0=float(i))
            nisa.scalar_tensor_tensor(dst=att, data=at_i, op0=nl.multiply,
                                      operand0=own_sel[:, i:i + 1], op1=nl.add, operand1=att)
        nisa.tensor_tensor(dst=att, data1=att, data2=in_open, op=nl.maximum)
        nisa.tensor_tensor(dst=att, data1=att, data2=upto, op=nl.multiply)

        # ---- partials: [A for the other program's columns | M | T] ------------------
        t_m = _col(rows, nl.float32)
        nisa.tensor_copy(dst=t_m, src=sum_ps)
        a_mine = _sb((rows, half), nl.float32)
        nisa.tensor_copy(dst=a_mine, src=a_ps[:, my_c0:my_c0 + half])
        if n_prgs == 2:
            mine = _sb((rows, half + _LINE), nl.float32)
            nisa.tensor_copy(dst=mine[:, 0:half], src=a_ps[:, their_c0:their_c0 + half])
            nisa.tensor_copy(dst=mine[:, half:half + 1], src=m_c)
            nisa.tensor_copy(dst=mine[:, half + 1:half + 2], src=t_m)
            theirs = _sb((rows, half + _LINE), nl.float32)
            nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - prg, recv_from_rank=1 - prg,
                          pipe_id=0)
            m_t = theirs[:, half:half + 1]

        # ---- merge: both halves and the own columns ---------------------------------
        own_score = _sb((rows, rows), nl.float32)
        nisa.tensor_copy(dst=own_score, src=os_ps)
        # MASKED_SCORE where an own column does not count, exactly.
        own_off_score = _sb((rows, rows), nl.float32)
        nisa.tensor_scalar(dst=own_off_score, data=att, op0=nl.subtract, operand0=1.0,
                           op1=nl.multiply, operand1=-MASKED_SCORE)
        nisa.tensor_tensor(dst=own_score, data1=own_score, data2=att, op=nl.multiply)
        nisa.tensor_tensor(dst=own_score, data1=own_score, data2=own_off_score, op=nl.add)
        own_max = _col(rows, nl.float32)
        nisa.tensor_reduce(dst=own_max, op=nl.maximum, data=own_score, axis=1)
        top = _col(rows, nl.float32)
        nisa.tensor_tensor(dst=top, data1=m_c, data2=own_max, op=nl.maximum)
        if n_prgs == 2:
            nisa.tensor_tensor(dst=top, data1=top, data2=m_t, op=nl.maximum)
        nisa.tensor_scalar(dst=top, data=top, op0=nl.maximum, operand0=_MAX_FLOOR)
        neg_top = _col(rows, nl.float32)
        nisa.tensor_scalar(dst=neg_top, data=top, op0=nl.multiply, operand0=-softmax_scale)
        f_m = _col(rows, nl.float32)
        nisa.activation(dst=f_m, op=nl.exp, data=m_c, bias=neg_top, scale=softmax_scale)
        e_own = _sb((rows, rows), nl.float32)
        nisa.activation(dst=e_own, op=nl.exp, data=own_score, bias=neg_top,
                        scale=softmax_scale)
        p_own = _sb((rows, rows), nl.float32)
        own_sum = _col(rows, nl.float32)
        nisa.tensor_tensor(dst=p_own, data1=e_own, data2=att, op=nl.multiply)
        nisa.tensor_reduce(dst=own_sum, op=nl.add, data=p_own, axis=1)
        total_c = _col(rows, nl.float32)
        nisa.scalar_tensor_tensor(dst=total_c, data=t_m, op0=nl.multiply, operand0=f_m,
                                  op1=nl.add, operand1=own_sum)
        if n_prgs == 2:
            f_t = _col(rows, nl.float32)
            nisa.activation(dst=f_t, op=nl.exp, data=m_t, bias=neg_top, scale=softmax_scale)
            nisa.scalar_tensor_tensor(dst=total_c, data=theirs[:, half + 1:half + 2],
                                      op0=nl.multiply, operand0=f_t, op1=nl.add,
                                      operand1=total_c)
        nisa.tensor_scalar(dst=total_c, data=total_c, op0=nl.maximum, operand0=_TOTAL_FLOOR)
        recip = _col(rows, nl.float32)
        nisa.reciprocal(dst=recip, data=total_c)
        w_m = _col(rows, nl.float32)
        nisa.tensor_tensor(dst=w_m, data1=f_m, data2=recip, op=nl.multiply)
        w_o = _sb((rows, rows), nl.float32)
        nisa.tensor_scalar(dst=w_o, data=p_own, op0=nl.multiply, operand0=recip)
        res = _sb((rows, half), nl.float32)
        nisa.tensor_scalar(dst=res, data=a_mine, op0=nl.multiply, operand0=w_m)
        if n_prgs == 2:
            w_t = _col(rows, nl.float32)
            nisa.tensor_tensor(dst=w_t, data1=f_t, data2=recip, op=nl.multiply)
            nisa.scalar_tensor_tensor(dst=res, data=theirs[:, 0:half], op0=nl.multiply,
                                      operand0=w_t, op1=nl.add, operand1=res)
        for u in range(rows):
            # Own row u's half of the latent columns, on every row's partition.
            own_u = _sb((rows, half), written_hbm.dtype)
            nisa.dma_copy(dst=own_u, src=written_hbm.ap(pattern=[[0, rows], [1, half]],
                                                        offset=(first + u) * latent + my_c0))
            own_u_f = _sb((rows, half), nl.float32)
            nisa.tensor_copy(dst=own_u_f, src=own_u)
            nisa.scalar_tensor_tensor(dst=res, data=own_u_f, op0=nl.multiply,
                                      operand0=w_o[:, u:u + 1], op1=nl.add, operand1=res)
        nisa.dma_copy(dst=out_hbm.ap(pattern=[[heads * latent, rows], [1, half]],
                                     offset=first * heads * latent + my_c0),
                      src=res)


@nki.jit
def mla_decode_masked_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, scores_hbm,
                             softmax_scale, page_size, pool_size, select_k, fold_cap,
                             source_digest):
    """Selecting decode attention by pool mask, keys split over the programs (one head),
    the mask selected on chip from ``scores_hbm``.

    ``select_k`` pools per row; ``fold_cap`` is
    :func:`decode_select.select_fold_cap` of the candidate axis; ``source_digest`` is
    :data:`SOURCE_DIGEST`, which only keys the kernel cache.
    """
    total, heads, latent = q_hbm.shape
    out = nl.ndarray((total, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _masked_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, scores_hbm,
                 softmax_scale, page_size, pool_size, select_k, fold_cap, out)
    return out


def _programs() -> int:
    return 2 if lnc_pair() else 1


def mla_decode_masked_serves(*, heads: int, latent: int, page: int, window: int,
                             rows: int, pool: int) -> bool:
    """True when :func:`mla_decode_masked_attention` serves the geometry.

    One head, a latent of whole 128-column tiles per program, pages of whole 128-row
    chunks that split evenly over the programs, a power-of-two pool that divides a key
    chunk, every row's scores within one moving operand, a request's rows within one
    tile of the selection (:func:`decode_select.tile_requests`), and a program's window
    rows within :data:`mla_decode._ROW_BUDGET` bytes per partition (2-byte rows).
    """
    programs = _programs()
    chunks = window // KEY_CHUNK
    nck = chunks // programs
    return (heads == 1 and rows >= 1 and page % KEY_CHUNK == 0 and window % KEY_CHUNK == 0
            and latent % (LATENT_TILE * programs) == 0 and chunks % programs == 0
            and nck % (page // KEY_CHUNK) == 0 and nck * latent * 2 <= _ROW_BUDGET
            and pool >= 1 and pool & (pool - 1) == 0 and KEY_CHUNK % pool == 0
            and rows * nck <= MOVING_MAX and rows <= tile_requests(window // pool))


def mla_decode_masked_attention(q_lift: Tensor, bank: Tensor, block_table: Tensor,
                                position: Tensor, written: Tensor, softmax_scale: float,
                                page_size: int, pool_scores: Tensor, pool_size: int, *,
                                select_k: int) -> Tensor:
    """Selecting decode attention for ``B`` requests, ``T`` query rows each, by pool mask.
    ``[B * T, H, L]`` fp32.

    The module docstring's rule: row ``t`` of request ``b`` (row ``b * T + t`` of every
    row-major operand) sits at window row ``position[b] + t`` and attends the tokens of
    its ``select_k`` best candidate pools and of its incomplete pool up to its own row,
    which is the token set ``dsa_decode_select`` lists for it from the same scores.

    Args:
        q_lift: ``[B * T, 1, L]`` bf16/fp16, the absorbed queries, request-major.
        bank: ``[slots, L]`` the whole latent bank, same dtype.
        block_table: ``[B, pages]`` int, each request's pages (``-1`` pads).
        position: ``[B]`` int, the window row of each request's first row this step.
        written: ``[B * T, L]`` this step's own latent rows, request-major.
        softmax_scale: positive.
        page_size: rows per page, a trace-time int.
        pool_scores: ``[B * T, C]`` fp32, each row's bounded candidate pool scores (the
            score kernel's output: ``BOUND_FILL`` on a pool not complete at the row);
            ``C * pool_size`` is the window.
        pool_size: tokens per pool, a power of two that divides 128.
        select_k: pools each row selects, ``1 <= select_k < C``.

    Raises:
        MlaDecodeAttentionError: naming the first operand or geometry outside that
            contract (:func:`mla_decode_masked_serves`).
    """
    _require(q_lift, bank, block_table, position, written, page_size, None,
             float(softmax_scale))
    total, heads, latent = (int(d) for d in q_lift.shape)
    batch = int(block_table.shape[0])
    rows = total // batch
    window = int(block_table.shape[1]) * int(page_size)
    pool = int(pool_size)
    if pool < 1 or pool & (pool - 1) or window % pool:
        raise MlaDecodeAttentionError(
            f"pool_size must be a power of two that divides the {window}-row window; "
            f"got {pool_size}"
        )
    cands = window // pool
    if (pool_scores.ndim != 2 or tuple(pool_scores.shape) != (total, cands)
            or pool_scores.dtype != torch.float32):
        raise MlaDecodeAttentionError(
            f"pool_scores must be [batch * rows, window // pool_size] = {(total, cands)} "
            f"float32, one candidate pool per column; got {tuple(pool_scores.shape)} "
            f"{pool_scores.dtype}"
        )
    if not 1 <= int(select_k) < cands:
        raise MlaDecodeAttentionError(
            f"select_k must lie in [1, {cands}), below the {cands} candidate pools; got "
            f"{select_k}"
        )
    if not mla_decode_masked_serves(heads=heads, latent=latent, page=int(page_size),
                                    window=window, rows=rows, pool=pool):
        raise MlaDecodeAttentionError(
            f"the masked decode kernel does not serve heads={heads}, latent={latent}, "
            f"page_size={page_size}, window={window}, rows={rows}, pool_size={pool} on "
            f"{_programs()} program(s); see mla_decode_masked_serves"
        )
    programs = _programs()
    _count_dispatch(programs)
    call = wrap_nki(mla_decode_masked_kernel)
    if programs == 2:
        call = call[2]
    return call(q_lift.contiguous(), bank.contiguous(),
                block_table.contiguous().to(torch.int32),
                position.contiguous().to(torch.int32), written.contiguous(),
                pool_scores.contiguous(), float(softmax_scale), int(page_size), pool,
                int(select_k), int(select_fold_cap(cands)), SOURCE_DIGEST)


@nki.jit
def mla_decode_padded_split_kernel(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, topk_hbm,
                                   softmax_scale, page_size, padded, source_digest):
    """Selected decode attention, one query row per request, keys split over the programs,
    at a selected width whose 128-row chunks do not split evenly over them.

    The token indices go into a private scratch ``padded`` columns wide, ``-1`` (no token)
    past them, and :mod:`mla_decode`'s key-split body attends the scratch.
    ``source_digest`` is :data:`SOURCE_DIGEST`; it only keys the kernel cache.
    """
    batch, heads, latent = q_hbm.shape
    width = topk_hbm.shape[1]
    ids = _sb((batch, padded), nl.int32)
    nisa.memset(dst=ids, value=SENTINEL_INDEX)
    nisa.dma_copy(dst=ids[:, 0:width], src=topk_hbm)
    ids_hbm = nl.ndarray((batch, padded), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=ids_hbm, src=ids)
    out = nl.ndarray((batch, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _split_body(q_hbm, bank_hbm, table_hbm, pos_hbm, written_hbm, ids_hbm, softmax_scale,
                page_size, False, out, None, None)
    return out


def _padded_width(width: int, programs: int) -> int:
    """``width`` rounded up to a whole number of 128-row chunks per program."""
    step = KEY_CHUNK * programs
    return -(-int(width) // step) * step


def mla_decode_padded_split_serves(*, heads: int, latent: int, page: int,
                                   width: int) -> bool:
    """True when :func:`mla_decode_padded_split_attention` serves the geometry.

    An LNC2 pair; a selected width of whole 128-row chunks that do not split evenly over
    it (an even count is :mod:`mla_decode`'s key-split kernel's, and one program splits
    nothing); and a padded width the key-split body serves
    (:func:`mla_decode._split_serves`).
    """
    programs = _programs()
    padded = _padded_width(width, programs)
    return (programs == 2 and width % KEY_CHUNK == 0 and padded != width
            and _split_serves(heads, latent, page, padded, programs))


def mla_decode_padded_split_attention(q_lift: Tensor, bank: Tensor, block_table: Tensor,
                                      position: Tensor, written: Tensor,
                                      softmax_scale: float, page_size: int,
                                      topk_indices: Tensor) -> Tensor:
    """Selected decode attention for ``B`` requests, one query row each, keys split over an
    LNC2 pair at a width of an odd count of 128-row chunks. ``[B, H, L]`` fp32.

    The operands and the result are :func:`mla_decode.mla_decode_attention`'s; at such a
    width that function runs the general kernel on one program, and this one the
    key-split body on both, at the width padded with ``-1`` to whole chunks per program.

    Args:
        q_lift: ``[B, H, L]`` bf16/fp16, the absorbed queries.
        bank: ``[slots, L]`` the whole latent bank, same dtype.
        block_table: ``[B, pages]`` int, each request's pages (``-1`` pads).
        position: ``[B]`` int, each request's window row this step.
        written: ``[B, L]`` this step's own latent rows.
        softmax_scale: positive.
        page_size: rows per page, a trace-time int.
        topk_indices: ``[B, K]`` int window rows per request, ``-1`` for "no token".

    Raises:
        MlaDecodeAttentionError: naming the first operand or geometry outside that
            contract.
    """
    _require(q_lift, bank, block_table, position, written, page_size, topk_indices,
             float(softmax_scale))
    total, heads, latent = (int(d) for d in q_lift.shape)
    batch = int(block_table.shape[0])
    if total != batch:
        raise MlaDecodeAttentionError(
            f"the padded key-split form attends one query row per request; got {total} "
            f"rows over {batch} request(s)"
        )
    width = int(topk_indices.shape[1])
    if not mla_decode_padded_split_serves(heads=heads, latent=latent, page=int(page_size),
                                          width=width):
        raise MlaDecodeAttentionError(
            f"the padded key-split form serves one head on an LNC2 pair at a selected "
            f"width of an odd count of {KEY_CHUNK}-row chunks; got heads={heads}, "
            f"latent={latent}, page_size={page_size}, width={width}, "
            f"programs={_programs()}"
        )
    _count_padded_split()
    call = wrap_nki(mla_decode_padded_split_kernel)[2]
    return call(q_lift.contiguous(), bank.contiguous(),
                block_table.contiguous().to(torch.int32),
                position.contiguous().to(torch.int32), written.contiguous(),
                topk_indices.contiguous().to(torch.int32), float(softmax_scale),
                int(page_size), _padded_width(width, 2), SOURCE_DIGEST)
