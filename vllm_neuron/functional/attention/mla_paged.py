# SPDX-License-Identifier: Apache-2.0
"""Paged MLA attention for Kimi K3 in NKI: decode steps and 128-row prefill tiles.

One launch takes the all-reduced ``q_a`` and ``kv_a`` rows ``[T, 2112]`` fp32, the output
gate projection ``[T, H * 128]`` before its sigmoid, and the KV cache's int8 page buffer. It
writes the tokens' latent rows ``[T, 576]`` bf16 into the buffer at their slots and returns
the gated attention output and the buffer. Decode writes the output head dimension first,
``[128, T, H, 1]`` bf16, the input layout of the output projection; prefill writes it
``[T, H * 128]`` bf16. Per token, as the checkpoint's ``KimiMLAAttention``: the ``q_a`` and
``kv_a`` RMS norms, ``q_b``, attention with no rotary stage, and the sigmoid gate. The query
is absorbed into the latent:
``q_lift = [W_UK^T q_nope, q_rope]``, 576 wide in bf16, scores the latent rows directly, and
``W_UV`` applies once, in fp32, to the softmax-weighted sum of the rows. The normed q and
``q_nope`` enter ``q_b`` and ``W_UK`` as bf16 high and low parts, and the fp32 softmax
weights enter the weighted sum the same way, so ``q_lift`` is the only bf16 rounding on the
query side.

The buffer is ``[blocks, page bytes]`` int8, one page per block, shared with other layers'
caches. Each page holds the layer's latent block ``[block_size, 576]`` bf16, token-major, from
a byte offset; the kernels read it through a bf16 view, with no copy, and write only the new
rows, so the buffer is updated in place. PE transposes put the latent on the partitions for
the scores; the pages themselves move for the weighted sum. The softmax runs online over
512-key blocks in fp32. Rows past a sequence's length are masked by fill: their scores
become the fp32 minimum before the exp
and their latent becomes zero before the weighted sum, so no row a sequence does not own
reaches its output. A token's own latent row is one more key, computed in the launch.

At LNC 2 decode splits the sequences over the two programs, or the keys of a lone sequence,
merged over ``sendrecv``. Prefill takes one sequence: groups of 128-row query tiles
alternate between the programs, each over the paged prefix and the causal chunk.

Decode also runs in DCP mode, on one rank of a decode context parallel group that holds
every group-size-th block of each sequence's cache. The rank's heads' ``q_lift`` is
all-gathered over the group in bf16. The rank attends its own pages, with the new row a
key on its owner rank only, to an fp32 partial divided by its own sum and an fp32 LSE; a
rank with no key writes partial 0 and LSE -3.0e38 by select. The merge weighs each partial
by ``exp(LSE - group LSE)`` and returns each rank its own heads: an all-to-all of partials
and LSEs at 4 to 15 ranks, otherwise an all-gather of the LSEs and a reduce-scatter of the
weighted partials. The collectives run inside the kernel, on shared HBM.

Prefill in DCP mode reads the cached prefix from a second buffer, the sequence's blocks
gathered over the group, and stores each chunk row at its slot of the rank's own buffer, so
the rank's buffer is the one it returns.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
import torch.distributed as dist
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from libtorch_neuronx_lite.parallel.replica_mesh import resolve_full_mesh
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

_P = 128
_BLOCK = 512  # keys per online-softmax step: one fp32 PSUM bank of scores
_NOPE = 128
_ROPE = 64
_V = 128
_Q_LORA = 1536
_KV_LORA = 512
_LATENT = _KV_LORA + _ROPE
_ROW_BYTES = 2 * _LATENT
_QKV = _Q_LORA + _LATENT
_K_TILES = _Q_LORA // _P
_C_TILES = ((0, 128), (128, 128), (256, 128), (384, 128), (512, _ROPE))  # (first, width)
_V_TILES = _KV_LORA // _P
_MAX_FLOOR = -1.0e20  # running-max floor: keys filled below it weigh exactly 0
_FILL = -3.4028234663852886e38  # the fp32 minimum, range_select's only fill value
_Q_GROUP = 4  # prefill query tiles that share one pass over the keys
_KEYS_AHEAD = 3  # decode steps between a page's key transposes and its scores
_DUP = 32  # decode: partition of the low-part copy of each group's query rows
_DEAD_LSE = -3.0e38  # LSE of a rank with no key: its merge weight underflows to exactly 0


def _even(n):
    """``n`` rounded up to even: bf16 PSUM writes of a matmul start and end on 4-byte bounds."""
    return n + (n & 1)


def _zeros(shape, dtype):
    """An SBUF tile cleared to zero, for the pad row of an odd-sized bf16 transpose."""
    t = _sb(shape, dtype)
    nisa.memset(dst=t, value=0.0)
    return t


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _ps(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.psum)


def _load_weights(w_qb_t, w_uk, w_uv_t, heads):
    """SBUF copies of ``q_b^T`` by 128-row K tile, ``W_UK`` by head, ``W_UV^T`` by latent tile."""
    width = heads * (_NOPE + _ROPE)
    wqb = _sb((_P, _K_TILES, width), w_qb_t.dtype)
    nisa.dma_copy(dst=wqb, src=w_qb_t.ap(pattern=[[width, _P], [_P * width, _K_TILES],
                                                  [1, width]], offset=0))
    wuk = _sb((_P, heads, _KV_LORA), w_uk.dtype)
    nisa.dma_copy(dst=wuk, src=w_uk.ap(pattern=[[_KV_LORA, _P], [_P * _KV_LORA, heads],
                                                [1, _KV_LORA]], offset=0))
    wuv = _sb((_P, _V_TILES, heads, _V), w_uv_t.dtype)
    nisa.dma_copy(dst=wuv, src=w_uv_t.ap(pattern=[[_V, _P], [_P * _V, _V_TILES],
                                                  [_KV_LORA * _V, heads], [1, _V]], offset=0))
    return wqb, wuk, wuv


def _gain(weight, rows):
    """A norm gain ``[width]`` on ``rows`` partitions, in fp32."""
    gain = _sb((rows, weight.shape[0]), nl.float32)
    nisa.dma_copy(dst=gain, src=weight.ap(pattern=[[0, rows], [1, weight.shape[0]]], offset=0))
    return gain


def _rms_rows(dst, x, gain, eps):
    """``dst = x * rsqrt(mean(x^2) + eps) * gain`` per row, in fp32 math."""
    rows, width = x.shape
    sq = _sb((rows, width), nl.float32)
    ss = _sb((rows, 1), nl.float32)
    nisa.activation(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / width, op1=nl.add,
                       operand1=float(eps))
    rstd = _sb((rows, 1), nl.float32)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    nisa.scalar_tensor_tensor(dst=dst, data=x, op0=nl.multiply, operand0=rstd, op1=nl.multiply,
                              operand1=gain)


def _latent_rows(lat, x, kv_gain, eps):
    """``lat[:T] = [norm(kv_a[:512]) * gain, kv_a[512:]]`` in bf16 for the T qkv_a rows ``x``."""
    rows = x.shape[0]
    _rms_rows(lat[0:rows, 0:_KV_LORA], x[:, _Q_LORA:_Q_LORA + _KV_LORA], kv_gain, eps)
    nisa.tensor_copy(dst=lat[0:rows, _KV_LORA:_LATENT], src=x[:, _Q_LORA + _KV_LORA:_QKV])


def _slot(t, head, width, rows, by_token):
    """The ``[width, rows]`` columns of ``head`` in a query tile laid out by token or by head."""
    return t[0:width, 0:rows, head] if by_token else t[0:width, head, 0:rows]


def _split(src, rows):
    """bf16 high and low parts of an fp32 tile's ``rows`` rows, each padded to an even row count."""
    parts = []
    for _ in range(2):
        parts.append((_zeros if rows % 2 else _sb)((_even(rows), src.shape[1]), nl.bfloat16))
    nisa.tensor_copy(dst=parts[0][0:rows, :], src=src[0:rows, :])
    nisa.tensor_tensor(dst=parts[1][0:rows, :], data1=src[0:rows, :], data2=parts[0][0:rows, :],
                       op=nl.subtract)
    return parts


def _sum_parts(dst, ps, rows):
    """``dst = ps[:, :rows] + ps[:, rows:]``: a product's high-part and low-part columns summed."""
    low = _sb((ps.shape[0], rows), nl.float32)  # one operand of an add may sit in PSUM
    nisa.tensor_copy(dst=low, src=ps[:, rows:2 * rows])
    nisa.tensor_tensor(dst=dst, data1=ps[:, 0:rows], data2=low, op=nl.add)


def _absorbed_queries(ql, by_token, x, q_gain, wqb, wuk, eps, heads, k_outer=False):
    """``ql[i]``: ``q_lift^T`` of each head on latent tile ``i``, bf16, for the qkv_a rows ``x``.

    The normed q and ``q_nope`` enter their products unrounded, as bf16 high and low parts
    side by side in the moving operand, so each weight tile loads once; the parts' columns are
    summed after the product. With ``k_outer`` the ``q_b`` products run K tile by K tile over
    all heads, so each tile's products start when that tile of ``q_b`` lands.
    """
    rows = x.shape[0]
    n = 2 * rows
    qn = _sb((rows, _Q_LORA), nl.float32)
    _rms_rows(qn, x[:, 0:_Q_LORA], q_gain, eps)
    qnt = _sb((_P, _K_TILES, n), nl.bfloat16)
    for j, part in enumerate(_split(qn, rows)):
        for k0 in range(0, _K_TILES, 4):
            tp = _ps((_P, 4, _even(rows)), nl.bfloat16)
            for k in range(4):
                nisa.nc_transpose(dst=tp[:, k, :], data=part[:, (k0 + k) * _P:(k0 + k + 1) * _P])
            nisa.tensor_copy(dst=qnt[:, k0:k0 + 4, j * rows:(j + 1) * rows], src=tp[:, :, 0:rows])
    if k_outer:
        products = [(_ps((_NOPE, n), nl.float32), _ps((_ROPE, n), nl.float32))
                    for _ in range(heads)]
        for k in range(_K_TILES):
            for h, (qp, qr) in enumerate(products):
                base = h * (_NOPE + _ROPE)
                nisa.nc_matmul(dst=qp, stationary=wqb[:, k, base:base + _NOPE],
                               moving=qnt[:, k, :], accumulate=k > 0)
                nisa.nc_matmul(dst=qr, stationary=wqb[:, k, base + _NOPE:base + _NOPE + _ROPE],
                               moving=qnt[:, k, :], accumulate=k > 0)
    for h in range(heads):
        base = h * (_NOPE + _ROPE)
        if k_outer:
            qp, qr = products[h]
        else:
            qp = _ps((_NOPE, n), nl.float32)
            for k in range(_K_TILES):
                nisa.nc_matmul(dst=qp, stationary=wqb[:, k, base:base + _NOPE],
                               moving=qnt[:, k, :], accumulate=k > 0)
            qr = _ps((_ROPE, n), nl.float32)
            for k in range(_K_TILES):
                nisa.nc_matmul(dst=qr, stationary=wqb[:, k, base + _NOPE:base + _NOPE + _ROPE],
                               moving=qnt[:, k, :], accumulate=k > 0)
        _sum_parts(_slot(ql[4], h, _ROPE, rows, by_token), qr, rows)
        q_nope = _sb((_NOPE, rows), nl.float32)
        _sum_parts(q_nope, qp, rows)
        moving = _sb((_NOPE, n), nl.bfloat16)
        nisa.tensor_copy(dst=moving[:, 0:rows], src=q_nope)
        nisa.tensor_tensor(dst=moving[:, rows:n], data1=q_nope, data2=moving[:, 0:rows],
                           op=nl.subtract)
        for i in range(4):
            lp = _ps((_P, n), nl.float32)
            nisa.nc_matmul(dst=lp, stationary=wuk[:, h, i * _P:(i + 1) * _P], moving=moving)
            _sum_parts(_slot(ql[i], h, _P, rows, by_token), lp, rows)


def _cache_keys(table_row, length, blocks, pages, window, positions):
    """A sequence's block ids clamped into the cache, its length as a ``[128, 1]`` fp32 bound,
    and ``invalid[r, j]``: window row ``128 j + r`` lies at or past the length."""
    raw = _sb((1, pages), nl.int32)
    nisa.dma_copy(dst=raw, src=table_row)
    held = _sb((1, pages), nl.int32)
    nisa.tensor_scalar(dst=held, data=raw, op0=nl.maximum, operand0=0, op1=nl.minimum,
                       operand1=blocks - 1)
    n = _sb((_P, 1), nl.int32)
    nisa.dma_copy(dst=n, src=length)
    bound = _sb((_P, 1), nl.float32)
    nisa.tensor_scalar(dst=bound, data=n, op0=nl.maximum, operand0=0.0, op1=nl.minimum,
                       operand1=float(window))
    invalid = _sb((_P, positions.shape[1]), nl.uint8)
    nisa.tensor_scalar(dst=invalid, data=positions, op0=nl.greater_equal, operand0=bound)
    return held, bound, invalid


def _cache_pages(cache, held, first_tile, n_tiles, block_size, first, queue=None):
    """Window tiles ``first_tile ..`` of a sequence as one ``[128, n_tiles, 576]`` page tile.

    ``cache`` is the bf16 view ``[blocks, page elements]`` of the page buffer, its latent
    block from element ``first`` of each page. ``queue`` names the loads' DMA options.
    """
    per = block_size // _P
    pages = _sb((_P, n_tiles, _LATENT), cache.dtype)
    t = 0
    while t < n_tiles:
        b, r = (first_tile + t) // per, (first_tile + t) % per
        n = min(n_tiles - t, per - r)
        nisa.dma_copy(dst=pages[:, t:t + n, :],
                      src=cache.ap(pattern=[[_LATENT, _P], [_P * _LATENT, n], [1, _LATENT]],
                                   offset=first + r * _P * _LATENT,
                                   scalar_offset=held[0:1, b:b + 1], indirect_dim=0),
                      **(queue or {}))
        t += n
    return pages


def _after(inst, producer):
    """Schedule ``inst`` after ``producer``, with no hardware wait; no change when it is None."""
    if producer is not None:
        inst.depends_on(producer, kind=nisa.dependency.schedule)
    return inst


def _page_views(pages, latent_offset, block_size):
    """The page buffer as bf16 ``[blocks, page elements]`` for the page reads and as bf16 rows
    ``[blocks * page rows, 576]`` for the row writes, with the latent block's first element and
    first row in a page."""
    blocks, page_bytes = pages.shape
    kernel_assert(pages.dtype == nl.int8, "the page buffer is int8")
    kernel_assert(page_bytes % _ROW_BYTES == 0 and latent_offset % _ROW_BYTES == 0,
                  "pages and the latent offset are whole 1152-byte rows")
    kernel_assert(latent_offset + block_size * _ROW_BYTES <= page_bytes,
                  "the latent block lies inside a page")
    kernel_assert(block_size % _P == 0 and block_size & (block_size - 1) == 0,
                  "block_size is a power of two, a multiple of 128")
    by_page = pages.view(nl.bfloat16)
    by_row = by_page.reshape((blocks * (page_bytes // _ROW_BYTES), _LATENT))
    return by_page, by_row, latent_offset // 2, latent_offset // _ROW_BYTES


def _row_index(slots, first, rows, blocks, block_size, page_rows, first_row):
    """``[rows, 1]`` int32: the row of the row view that each of ``slots[first:first + rows]``
    names. A slot outside the cache, as the -1 of a padding token, names block 0, row 0."""
    s = _sb((rows, 1), nl.int32)
    nisa.dma_copy(dst=s, src=slots.ap(pattern=[[1, rows], [1, 1]], offset=first))
    high = _sb((rows, 1), nl.uint8)
    nisa.tensor_scalar(dst=high, data=s, op0=nl.greater_equal, operand0=float(blocks * block_size))
    outside = _sb((rows, 1), nl.uint8)
    nisa.scalar_tensor_tensor(dst=outside, data=s, op0=nl.less, operand0=0.0, op1=nl.maximum,
                              operand1=high)
    nisa.tensor_copy_predicated(dst=s, src=0, predicate=outside)
    # Row of slot s: (s // block_size) * page_rows + first_row + s % block_size.
    skip = _sb((rows, 1), nl.int32)
    nisa.tensor_scalar(dst=skip, data=s, op0=nl.right_shift, operand0=block_size.bit_length() - 1)
    nisa.tensor_scalar(dst=skip, data=skip, op0=nl.multiply, operand0=page_rows - block_size,
                       op1=nl.add, operand1=first_row)
    idx = _sb((rows, 1), nl.int32)
    nisa.tensor_tensor(dst=idx, data1=skip, data2=s, op=nl.add)
    return idx


def _write_rows(by_row, idx, lat, rows):
    """Store the latent rows ``lat[:rows]`` at the rows of ``by_row`` that ``idx`` names."""
    nisa.dma_copy(dst=by_row.ap(pattern=[[_LATENT, rows], [1, _LATENT]], offset=0,
                                vector_offset=idx, indirect_dim=0), src=lat[0:rows, :])


def _clear_unowned(pages, invalid, first_tile, after=None):
    """Zero the latent of rows the sequence does not own, so the weighted sum reads finite rows."""
    tiles = []
    for t in range(pages.shape[1]):
        j = first_tile + t
        _after(nisa.tensor_copy_predicated(dst=pages[:, t, 0:_KV_LORA], src=0.0,
                                           predicate=invalid[:, j:j + 1].broadcast(1, _KV_LORA)),
               after)
        tiles.append(pages[:, t, :])
    return tiles


def _keys_t(tiles, after=None):
    """The key pages with the latent on partitions: ``kt[i] [w_i, keys]`` bf16, by PE transposes."""
    rows_k = [t.shape[0] for t in tiles]
    width = _P * (len(tiles) - 1) + rows_k[-1]
    kt = []
    for i, (c0, w) in enumerate(_C_TILES):
        tp = _ps((w, width), nl.bfloat16)
        for t, page in enumerate(tiles):
            _after(nisa.nc_transpose(dst=tp[:, t * _P:t * _P + rows_k[t]], data=page[:, c0:c0 + w]),
                   after)
        k_sb = _sb((w, width), nl.bfloat16)
        nisa.tensor_copy(dst=k_sb, src=tp,
                         engine=nisa.engine.scalar if i % 2 == 0 else nisa.engine.vector)
        kt.append(k_sb)
    return kt


def _scores(g, kind, kt, rows_k, key0, scale, zero):
    """Group ``g``'s masked scores over one key step: its new running max and bf16 weights.

    Keys ``key0 + 128 t + r`` lie on row ``r`` of tile ``t``; the group's queries take the keys
    in ``[0, g[kind])``, and with ``kind == "own"`` only the first ``g["own_tiles"](key0)`` tiles.
    """
    n_t = len(rows_k) if kind == "cache" else g["own_tiles"](key0)
    cols = _P * (n_t - 1) + rows_k[n_t - 1]
    r = g["rows"]
    s_ps = _ps((r, cols), nl.float32)
    for i in range(len(_C_TILES)):
        nisa.nc_matmul(dst=s_ps, stationary=g["ql"][i], moving=kt[i][:, 0:cols], accumulate=i > 0)
    s_sb = _sb((r, cols), nl.float32)
    top = _sb((r, 1), nl.float32)
    nisa.range_select(dst=s_sb, on_true_tile=s_ps, comp_op0=nl.greater_equal,
                      comp_op1=nl.less, bound0=zero[0:r, :], bound1=g[kind][0:r, :],
                      reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=top,
                      reduce_op=nl.maximum, range_start=key0, on_false_value=_FILL)
    m = _sb((r, 1), nl.float32)
    if g["m"] is None:
        nisa.tensor_scalar(dst=m, data=top, op0=nl.maximum, operand0=_MAX_FLOOR)
    else:
        nisa.tensor_tensor(dst=m, data1=g["m"], data2=top, op=nl.maximum)
    neg = _sb((r, 1), nl.float32)
    nisa.tensor_scalar(dst=neg, data=m, op0=nl.multiply, operand0=-scale)
    p = _sb((r, cols), nl.float32)
    total = _sb((r, 1), nl.float32)
    nisa.activation(dst=p, op=nl.exp, data=s_sb, bias=neg, scale=scale, reduce_op=nl.add,
                    reduce_res=total, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    if g["split"] == "rows":
        # Rows from g["dup"] repeat the group's queries: they carry the low part in the same tile.
        dup = g["dup"]
        hi = _sb((r, cols), nl.bfloat16)
        nisa.tensor_copy(dst=hi, src=p)
        nisa.tensor_tensor(dst=hi[dup:r, :], data1=p[dup:r, :], data2=hi[dup:r, :],
                           op=nl.subtract)
        parts = [hi]
    else:
        parts = _split(p, r)
    corr = None
    if g["m"] is not None:
        corr = _sb((r, 1), nl.float32)
        nisa.activation(dst=corr, op=nl.exp, data=g["m"], bias=neg, scale=scale)
    g["m"] = m
    return {"g": g, "rows_k": rows_k[0:n_t], "parts": parts, "total": total, "corr": corr}


def _weights_t(sc):
    """``p^T`` of each bf16 part by key tile, ``[128, n_t, rows]``, the stationaries of the sum."""
    rows_k, r = sc["rows_k"], sc["g"]["rows"]
    sc["pt"] = []
    for part in sc["parts"]:
        pt_ps = _ps((_P, len(rows_k), r), nl.bfloat16)
        for t, k in enumerate(rows_k):
            nisa.nc_transpose(dst=pt_ps[0:k, t, :], data=part[0:r, t * _P:t * _P + k])
        pt = _sb((_P, len(rows_k), r), nl.bfloat16)
        for t, k in enumerate(rows_k):
            nisa.tensor_copy(dst=pt[0:k, t, :], src=pt_ps[0:k, t, :], engine=nisa.engine.scalar)
        sc["pt"].append(pt)


def _accumulate(sc, tiles):
    """Add one step's weighted latent rows into the group's fp32 online-softmax state."""
    g, rows_k = sc["g"], sc["rows_k"]
    o_ps = _ps((g["rows"], _KV_LORA), nl.float32)
    for t, k in enumerate(rows_k):
        for j, pt in enumerate(sc["pt"]):
            nisa.nc_matmul(dst=o_ps, stationary=pt[0:k, t, :], moving=tiles[t][:, 0:_KV_LORA],
                           accumulate=t > 0 or j > 0)
    acc = _sb((g["rows"], _KV_LORA), nl.float32)
    if sc["corr"] is None:
        nisa.tensor_copy(dst=acc, src=o_ps)
        g["l"] = sc["total"]
    else:
        lsum = _sb((g["rows"], 1), nl.float32)
        nisa.scalar_tensor_tensor(dst=lsum, data=g["l"], op0=nl.multiply, operand0=sc["corr"],
                                  op1=nl.add, operand1=sc["total"])
        nisa.scalar_tensor_tensor(dst=acc, data=g["acc"], op0=nl.multiply, operand0=sc["corr"],
                                  op1=nl.add, operand1=o_ps)
        g["l"] = lsum
    g["acc"] = acc


def _attend(groups, kind, tiles, key0, scale, zero, values=None):
    """One online-softmax step over the key ``tiles`` ``[rows, 576]``; the sum reads ``values``."""
    rows_k = [t.shape[0] for t in tiles]
    kt = _keys_t(tiles)
    for g in groups:
        sc = _scores(g, kind, kt, rows_k, key0, scale, zero)
        _weights_t(sc)
        _accumulate(sc, tiles if values is None else values)


def _pipelined(steps, scale, zero, queries):
    """Run single-group steps with each engine's work for later steps emitted ahead.

    A step is ``(group, kind, key0, load)``; ``load()`` issues its page DMA and returns a
    finisher that clears the unowned rows and returns the tiles. The emission order per
    iteration is: loads five steps ahead, clears four ahead, weights of this step, key
    transposes three ahead, scores one ahead, then this step's sum, so each engine has
    independent work queued while the dependent chain of a step completes. ``queries()``
    fills the groups' queries and is emitted just before the first scores.
    """
    n, st, k = len(steps), [dict() for _ in steps], _KEYS_AHEAD
    for t in range(-k - 2, n):
        if t == -1:
            queries()
        if t + k + 2 < n:
            st[t + k + 2]["finish"] = steps[t + k + 2][3]()
        if 0 <= t + k + 1 < n:
            st[t + k + 1]["tiles"] = st[t + k + 1].pop("finish")()
        if 0 <= t < n:
            _weights_t(st[t]["sc"])
        if 0 <= t + k < n:
            st[t + k]["kt"] = _keys_t(st[t + k]["tiles"])
        if 0 <= t + 1 < n:
            g, kind, key0, _ = steps[t + 1]
            rows_k = [x.shape[0] for x in st[t + 1]["tiles"]]
            st[t + 1]["sc"] = _scores(g, kind, st[t + 1].pop("kt"), rows_k, key0, scale, zero)
        if 0 <= t < n:
            _accumulate(st[t]["sc"], st[t]["tiles"])
            st[t].clear()


def _gated_pipeline(steps, scale, zero, gate, free, queries):
    """``_pipelined`` for steps whose page loads are issued, with queries from an all-gather.

    A step is ``(group, kind, key0, finish)``, ``finish(gate)`` the finisher of its load. The
    first ``free`` steps' clears and key transposes are emitted first, free of the gather, so
    they run as their pages land. ``queries()`` then fills the groups' queries from the gather,
    and the first scores follow. Every later step's clears and key transposes are scheduled
    after ``gate``, the gather, four and three steps ahead of their scores as in ``_pipelined``.
    """
    n, st, k = len(steps), [dict() for _ in steps], _KEYS_AHEAD
    free = n if n <= k + 1 else min(n, max(free, k + 1))  # the first scores' keys are free
    for i in range(-1, free):
        if i + 1 < free:
            st[i + 1]["tiles"] = steps[i + 1][3]()
        if i >= 0:
            st[i]["kt"] = _keys_t(st[i]["tiles"])
    queries()
    for t in range(-1, n):
        if free <= t + k + 1 < n:
            st[t + k + 1]["tiles"] = steps[t + k + 1][3](gate)
        if 0 <= t < n:
            _weights_t(st[t]["sc"])
        if free <= t + k < n:
            st[t + k]["kt"] = _keys_t(st[t + k]["tiles"], gate)
        if 0 <= t + 1 < n:
            g, kind, key0, _ = steps[t + 1]
            rows_k = [x.shape[0] for x in st[t + 1]["tiles"]]
            st[t + 1]["sc"] = _scores(g, kind, st[t + 1].pop("kt"), rows_k, key0, scale, zero)
        if 0 <= t < n:
            _accumulate(st[t]["sc"], st[t]["tiles"])
            st[t].clear()


def _merge(first, second, scale, rows):
    """Online-softmax states ``(m, l, acc)`` of the same queries over disjoint keys, combined."""
    m = _sb((rows, 1), nl.float32)
    nisa.tensor_tensor(dst=m, data1=first[0], data2=second[0], op=nl.maximum)
    neg = _sb((rows, 1), nl.float32)
    nisa.tensor_scalar(dst=neg, data=m, op0=nl.multiply, operand0=-scale)
    f0 = _sb((rows, 1), nl.float32)
    nisa.activation(dst=f0, op=nl.exp, data=first[0], bias=neg, scale=scale)
    f1 = _sb((rows, 1), nl.float32)
    nisa.activation(dst=f1, op=nl.exp, data=second[0], bias=neg, scale=scale)
    l1 = _sb((rows, 1), nl.float32)
    nisa.tensor_tensor(dst=l1, data1=second[1], data2=f1, op=nl.multiply)
    lsum = _sb((rows, 1), nl.float32)
    nisa.scalar_tensor_tensor(dst=lsum, data=first[1], op0=nl.multiply, operand0=f0, op1=nl.add,
                              operand1=l1)
    a1 = _sb((rows, _KV_LORA), nl.float32)
    nisa.tensor_scalar(dst=a1, data=second[2], op0=nl.multiply, operand0=f1)
    acc = _sb((rows, _KV_LORA), nl.float32)
    nisa.scalar_tensor_tensor(dst=acc, data=first[2], op0=nl.multiply, operand0=f0, op1=nl.add,
                              operand1=a1)
    return m, lsum, acc


def _normalized(l, acc, rows):
    """``acc / l`` of the first ``rows`` rows: the fp32 softmax-weighted latent."""
    recip = _sb((rows, 1), nl.float32)
    nisa.reciprocal(dst=recip, data=l[0:rows, :])
    o = _sb((rows, _KV_LORA), nl.float32)
    nisa.tensor_scalar(dst=o, data=acc[0:rows, :], op0=nl.multiply, operand0=recip)
    return o


def _gated_output(dst, o_t, wuv, sig, heads):
    """``dst[:, head cols] = (o @ W_UV^T) * sigmoid(gate)`` from ``o_t[c, tile, head, row]``."""
    for h in range(heads):
        op = _ps((dst.shape[0], _V), nl.float32)
        for t in range(_V_TILES):
            nisa.nc_matmul(dst=op, stationary=o_t[:, t, h, :], moving=wuv[:, t, h, :],
                           accumulate=t > 0)
        nisa.tensor_tensor(dst=dst[:, h * _V:(h + 1) * _V], data1=op,
                           data2=sig[:, h * _V:(h + 1) * _V], op=nl.multiply)


def _gated_output_t(dst, o_t, wuv, sig_t, heads):
    """``dst[d, row, head] = (o @ W_UV^T)[row, d] * sigmoid(gate)``: the head dimension first."""
    for h in range(heads):
        op = _ps((_V, dst.shape[1]), nl.float32)
        for t in range(_V_TILES):
            nisa.nc_matmul(dst=op, stationary=wuv[:, t, h, :], moving=o_t[:, t, h, :],
                           accumulate=t > 0)
        nisa.tensor_tensor(dst=dst[:, :, h], data1=op, data2=sig_t[:, :, h], op=nl.multiply)


def _gate_t(gate, row0, rows, heads):
    """``sigmoid(gate)`` of ``rows`` rows from ``row0`` as ``[128 d, rows, heads]`` fp32."""
    sig = _gate(gate, row0, rows, heads * _V)
    sig_t = _sb((_V, rows, heads), nl.float32)
    for h in range(heads):
        tp = _ps((_V, rows), nl.float32)
        nisa.nc_transpose(dst=tp, data=sig[:, h * _V:(h + 1) * _V])
        nisa.tensor_copy(dst=sig_t[:, :, h], src=tp)
    return sig_t


def _gate(gate, row0, rows, width):
    """``sigmoid(gate)`` of ``rows`` rows from ``row0``, fp32."""
    g = _sb((rows, width), nl.float32)
    nisa.dma_copy(dst=g, src=gate[row0:row0 + rows, :])
    sig = _sb((rows, width), nl.float32)
    nisa.activation(dst=sig, op=nl.sigmoid, data=g)
    return sig


def _transpose_into(o_t, o, rows):
    """``o_t[c, tile, row] = o[row, 128 tile + c]`` for an fp32 ``o [rows, 512]``."""
    tp = _ps((_P, _V_TILES, rows), nl.float32)
    for t in range(_V_TILES):
        nisa.nc_transpose(dst=tp[:, t, :], data=o[:, t * _P:(t + 1) * _P])
    nisa.tensor_copy(dst=o_t, src=tp)


def _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, slots):
    heads = w_uk.shape[0]
    kernel_assert(qkv_a.shape[1] == _QKV and qkv_a.dtype == nl.float32, "qkv_a is [T, 2112] fp32")
    kernel_assert(tuple(gate.shape) == (qkv_a.shape[0], heads * _V), "gate is [T, H * 128]")
    kernel_assert(tuple(q_norm.shape) == (_Q_LORA,) and tuple(kv_norm.shape) == (_KV_LORA,),
                  "norm gains are [1536] and [512]")
    kernel_assert(tuple(w_qb_t.shape) == (_Q_LORA, heads * (_NOPE + _ROPE)),
                  "q_b^T is [1536, H*192]")
    kernel_assert(tuple(w_uk.shape) == (heads, _NOPE, _KV_LORA), "W_UK is [H, 128, 512]")
    kernel_assert(tuple(w_uv_t.shape) == (heads, _KV_LORA, _V) and w_uv_t.dtype == nl.float32,
                  "W_UV^T is [H, 512, 128] fp32")
    kernel_assert(w_qb_t.dtype == nl.bfloat16 and w_uk.dtype == nl.bfloat16, "bf16 q_b and W_UK")
    kernel_assert(tuple(slots.shape) == (qkv_a.shape[0],), "slots is [T]")
    return heads


def _cache_step(cache, held, invalid, b, window, block_size, first_elem, queue=None):
    """The page load of a sequence's key block ``b``; its finisher clears the unowned rows,
    scheduled after the instruction it is given."""
    first_tile, n_t = b * (_BLOCK // _P), min(_BLOCK, window - b * _BLOCK) // _P

    def load():
        tiles = _cache_pages(cache, held, first_tile, n_t, block_size, first_elem, queue)
        return lambda gate=None: _clear_unowned(tiles, invalid, first_tile, gate)
    return load


def _own_step(lat, s):
    """The load of token ``s``'s own latent row as a one-key step."""
    def load():
        row = _zeros((2, _LATENT), nl.bfloat16)  # key 1 is past the bound and zero
        nisa.dma_copy(dst=row[0:1, :], src=lat[s:s + 1, :])
        return lambda gate=None: [row]
    return load


def _merge_programs(state, prg, scale, rows):
    """A lone sequence's states from the two programs' key halves, merged over ``sendrecv``."""
    m, l, acc = state
    mine = _sb((rows, _KV_LORA + 8), nl.float32)
    nisa.tensor_copy(dst=mine[:, 0:_KV_LORA], src=acc[0:rows, :])
    nisa.tensor_copy(dst=mine[:, _KV_LORA:_KV_LORA + 1], src=m[0:rows, :])
    nisa.tensor_copy(dst=mine[:, _KV_LORA + 1:_KV_LORA + 2], src=l[0:rows, :])
    theirs = _sb((rows, _KV_LORA + 8), nl.float32)
    nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - prg, recv_from_rank=1 - prg,
                  pipe_id=0)
    a, b = (mine, theirs) if prg == 0 else (theirs, mine)
    return _merge((a[:, _KV_LORA:_KV_LORA + 1], a[:, _KV_LORA + 1:_KV_LORA + 2], a[:, 0:_KV_LORA]),
                  (b[:, _KV_LORA:_KV_LORA + 1], b[:, _KV_LORA + 1:_KV_LORA + 2], b[:, 0:_KV_LORA]),
                  scale, rows)


def mla_paged_decode_kernel(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages,
                            block_table, context_lens, slots, EPS: float, LATENT_OFFSET: int,
                            BLOCK_SIZE: int):
    """One decode step of ``T`` sequences: ``(out [128, T, H, 1] bf16, pages)``.

    Args:
        qkv_a: ``[T, 2112]`` fp32, each sequence's new token: ``q_a`` then ``kv_a``.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm: ``[1536]`` ``q_a`` norm gain.
        kv_norm: ``[512]`` ``kv_a`` norm gain.
        w_qb_t: ``[1536, H * 192]`` bf16, the rank's ``q_b`` rows transposed.
        w_uk: ``[H, 128, 512]`` bf16, the ``k_nope`` rows of ``kv_b`` by head.
        w_uv_t: ``[H, 512, 128]`` fp32, the value rows of ``kv_b`` by head, transposed.
        pages: ``[blocks, page bytes]`` int8 page buffer, each page's latent block
            ``[BLOCK_SIZE, 576]`` bf16 from byte ``LATENT_OFFSET``; the new rows are written
            in place.
        block_table: ``[T, pages]`` int32 block ids; entries past a sequence's blocks are
            clamped into the cache and masked.
        context_lens: ``[T]`` int32 cached tokens per sequence; the new token follows them.
        slots: ``[T]`` int32 slot of each new token, ``block * BLOCK_SIZE + row``; -1 for a
            padding token, whose row goes to block 0.
        EPS: the norms' epsilon.
        LATENT_OFFSET: byte offset of the latent block in a page, a multiple of 1152.
        BLOCK_SIZE: tokens per block, a power of two and a multiple of 128.

    Returns:
        The gated attention output and the page buffer with the new rows.
    """
    tokens = qkv_a.shape[0]
    heads = _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, slots)
    cache, by_row, first_elem, first_row = _page_views(pages, LATENT_OFFSET, BLOCK_SIZE)
    blocks, block_size = pages.shape[0], BLOCK_SIZE
    page_rows = pages.shape[1] // _ROW_BYTES
    n_pages = block_table.shape[1]
    kernel_assert(tuple(block_table.shape) == (tokens, n_pages)
                  and tuple(context_lens.shape) == (tokens,),
                  "block_table is [T, pages] and context_lens is [T]")
    window = n_pages * block_size
    n_tiles = window // _P
    n_blocks = (window + _BLOCK - 1) // _BLOCK
    programs = nl.num_programs(0)
    prg = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    scale = float((_NOPE + _ROPE) ** -0.5)
    out = nl.ndarray((_V, tokens, heads, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # A lone sequence splits its key blocks; otherwise each program takes whole sequences.
    split = programs > tokens
    if split:
        first, count = 0, tokens
        b0 = prg * (n_blocks // 2)
        b1 = n_blocks // 2 if prg == 0 else n_blocks
        own = prg == 0
    else:
        base, extra = tokens // programs, tokens % programs
        first = prg * base + min(prg, extra)
        count = base + (1 if prg < extra else 0)
        b0, b1, own = 0, n_blocks, True

    wqb, wuk, wuv = _load_weights(w_qb_t, w_uk, w_uv_t, heads)
    x = _sb((count, _QKV), nl.float32)
    nisa.dma_copy(dst=x, src=qkv_a[first:first + count, :])
    lat = _sb((count, _LATENT), nl.bfloat16)
    _latent_rows(lat, x, _gain(kv_norm, count), EPS)
    if own:
        idx = _row_index(slots, first, count, blocks, block_size, page_rows, first_row)
    # Query rows 0.. and _DUP.. hold each head's query; the zero rows between weigh nothing.
    rows_q = _DUP + _even(heads)
    ql = [_zeros((_P, count, rows_q), nl.bfloat16) for _ in _C_TILES]
    q_gain = _gain(q_norm, count)

    positions = _sb((_P, n_tiles), nl.float32)
    nisa.iota(dst=positions, pattern=[[_P, n_tiles]], offset=0, channel_multiplier=1)
    zero = _sb((_P, 1), nl.float32)
    nisa.memset(dst=zero, value=0.0)
    one = _sb((_P, 1), nl.float32)
    nisa.memset(dst=one, value=1.0)
    seqs = []
    for s in range(count):
        seq = first + s
        held, bound, invalid = _cache_keys(
            block_table.ap(pattern=[[n_pages, 1], [1, n_pages]], offset=seq * n_pages),
            context_lens.ap(pattern=[[0, _P], [1, 1]], offset=seq),
            blocks, n_pages, window, positions)
        group = {"rows": rows_q, "split": "rows", "dup": _DUP,
                 "ql": [ql[i][0:w, s, :] for i, (_, w) in enumerate(_C_TILES)],
                 "cache": bound, "own": one, "own_tiles": lambda k: 1,
                 "m": None, "l": None, "acc": None}
        seqs.append((group, held, invalid))
    steps = [(group, "cache", b * _BLOCK,
              _cache_step(cache, held, invalid, b, window, block_size, first_elem))
             for b in range(b0, b1) for group, held, invalid in seqs]
    if own:
        steps += [(group, "own", 0, _own_step(lat, s)) for s, (group, _, _) in enumerate(seqs)]
    # The query chain is emitted after the first pages' loads and transposes, which do not need it.
    def queries():
        _absorbed_queries(ql, True, x, q_gain, wqb, wuk, EPS, heads)
        for i, (_, w) in enumerate(_C_TILES):
            nisa.tensor_copy(dst=ql[i][0:w, :, _DUP:_DUP + heads], src=ql[i][0:w, :, 0:heads])

    _pipelined(steps, scale, zero, queries)

    states = [(g["m"], g["l"], g["acc"]) for g, _, _ in seqs]
    if split:
        states = [_merge_programs(states[0], prg, scale, rows_q)]
    if split and prg != 0:
        return out, pages

    o_t = _sb((_P, _V_TILES, heads, count), nl.float32)
    for s, (_, l, acc) in enumerate(states):
        both = _sb((_P, _V_TILES, rows_q), nl.float32)
        _transpose_into(both, _normalized(l, acc, rows_q), rows_q)
        # The high and low parts' sums, on partitions 0.. and _DUP.., meet in the free dimension.
        nisa.tensor_tensor(dst=o_t[:, :, :, s], data1=both[:, :, 0:heads],
                           data2=both[:, :, _DUP:_DUP + heads], op=nl.add)
    res = _sb((_V, count, heads), nl.bfloat16)
    _gated_output_t(res, o_t, wuv, _gate_t(gate, first, count, heads), heads)
    nisa.dma_copy(dst=out.ap(pattern=[[tokens * heads, _V], [heads, count], [1, heads]],
                             offset=first * heads), src=res)
    # The new rows sit at or past their sequences' lengths, so every read above masks them.
    _write_rows(by_row, idx, lat, count)
    return out, pages


def _dup_row(rows):
    """First partition of the low-part copy of ``rows`` query rows, on a 32-row bound; None when
    two copies do not fit the 128 partitions."""
    dup = 32 * ((rows + 31) // 32)
    return dup if dup + rows <= _P else None


def _shared(shape, dtype, name):
    """A kernel-internal HBM tensor both programs share, under a fixed name: the programs'
    traces differ when their work does, and an unnamed tensor then gets one copy per program."""
    return nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm, name=name)


# DMA queues of the DCP mode: the query chain's loads and stores first, then the metadata, then
# the pages. A lower priority number is served first.
_QUERY_QUEUE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar, "priority": 0}
_META_QUEUE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync, "priority": 1}
_PAGE_QUEUE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync, "priority": 2}
_QB_CHUNK = 3  # K tiles of q_b per load
# Keys per program whose pages land, cleared and transposed, while the query chain runs: the
# page loads issue about 1.2 us apart, and the transposes and clears share the engines with it.
_FREE_KEYS = 2560


def _load_weights_first(w_qb_t, w_uk, heads):
    """``q_b^T`` by K tile and ``W_UK`` by head on the query queue, ``q_b`` in chunks of K tiles;
    also the ``W_UK`` load."""
    width = heads * (_NOPE + _ROPE)
    wqb = _sb((_P, _K_TILES, width), w_qb_t.dtype)
    for k0 in range(0, _K_TILES, _QB_CHUNK):
        nisa.dma_copy(dst=wqb[:, k0:k0 + _QB_CHUNK, :],
                      src=w_qb_t.ap(pattern=[[width, _P], [_P * width, _QB_CHUNK], [1, width]],
                                    offset=k0 * _P * width), **_QUERY_QUEUE)
    wuk = _sb((_P, heads, _KV_LORA), w_uk.dtype)
    last = nisa.dma_copy(dst=wuk, src=w_uk.ap(pattern=[[_KV_LORA, _P], [_P * _KV_LORA, heads],
                                                       [1, _KV_LORA]], offset=0), **_QUERY_QUEUE)
    return wqb, wuk, last


def _load_values(w_uv_t, heads, after):
    """``W_UV^T`` by latent tile on the page queue, scheduled after the instruction ``after``."""
    wuv = _sb((_P, _V_TILES, heads, _V), w_uv_t.dtype)
    _after(nisa.dma_copy(dst=wuv, src=w_uv_t.ap(pattern=[[_V, _P], [_P * _V, _V_TILES],
                                                         [_KV_LORA * _V, heads], [1, _V]],
                                                offset=0), **_PAGE_QUEUE), after)
    return wuv


def _gather_queries(qloc, first, count, tokens, heads, group, ranks, write):
    """The all-gather of every rank's ``q_lift`` for this program's tokens, and a function that
    reads it back, ``[128, 5, ranks, count * heads]`` bf16.

    ``qloc [128, 5, count, heads]`` holds the rank's own heads by latent tile. Each rank's
    chunk of the all-gather is its ``[128, 4, T, heads]`` nope tiles, then its rope tile.
    """
    per = tokens * heads
    nope = _V_TILES * _P * per
    size = nope + _ROPE * per
    src = _shared((1, size), nl.bfloat16, "query_gather_src")
    dst = _shared((ranks, size), nl.bfloat16, "query_gather_dst")
    if write:
        nisa.dma_copy(dst=src.ap(pattern=[[_V_TILES * per, _P], [per, _V_TILES], [heads, count],
                                          [1, heads]], offset=first * heads),
                      src=qloc[:, 0:_V_TILES, :, :], **_QUERY_QUEUE)
        nisa.dma_copy(dst=src.ap(pattern=[[per, _ROPE], [heads, count], [1, heads]],
                                 offset=nope + first * heads),
                      src=qloc[0:_ROPE, _V_TILES, :, :], **_QUERY_QUEUE)
    gather = ncc.all_gather(srcs=[src], dsts=[dst], replica_group=group, collective_dim=0)

    def read():
        q = _sb((_P, len(_C_TILES), ranks, count * heads), nl.bfloat16)
        nisa.dma_copy(dst=q[:, 0:_V_TILES, :, :],
                      src=dst.ap(pattern=[[_V_TILES * per, _P], [per, _V_TILES], [size, ranks],
                                          [1, count * heads]], offset=first * heads),
                      **_QUERY_QUEUE)
        nisa.dma_copy(dst=q[0:_ROPE, _V_TILES, :, :],
                      src=dst.ap(pattern=[[per, _ROPE], [size, ranks], [1, count * heads]],
                                 offset=nope + first * heads), **_QUERY_QUEUE)
        return q
    return gather, read


def _group_weights(lse, first, count, tokens, gathered, dup, group, ranks, write):
    """``exp(lse - group LSE)`` per query row and sequence, ``[rows, count]`` fp32, from an
    all-gather of every rank's LSE ``[gathered heads, T]``."""
    rows = lse.shape[0]
    src = _shared((1, gathered * tokens), nl.float32, "lse_gather_src")
    dst = _shared((ranks, gathered * tokens), nl.float32, "lse_gather_dst")
    if write:
        nisa.dma_copy(dst=src.ap(pattern=[[tokens, gathered], [1, count]], offset=first),
                      src=lse[0:gathered, :])
    ncc.all_gather(srcs=[src], dsts=[dst], replica_group=group, collective_dim=0)
    every = _sb((rows, ranks, count), nl.float32)
    nisa.memset(dst=every, value=0.0)  # the rows between the two copies stay finite
    for row0 in (0,) if dup is None else (0, dup):
        nisa.dma_copy(dst=every[row0:row0 + gathered, :, :],
                      src=dst.ap(pattern=[[tokens, gathered], [gathered * tokens, ranks],
                                          [1, count]], offset=first))
    top = _sb((rows, count), nl.float32)
    nisa.tensor_reduce(dst=top, op=nl.maximum, data=every.permute((0, 2, 1)), axis=2)
    shifted = _sb((rows, ranks, count), nl.float32)
    nisa.tensor_tensor(dst=shifted, data1=every, data2=top.expand_dim(1).broadcast(1, ranks),
                       op=nl.subtract)
    nisa.activation(dst=shifted, op=nl.exp, data=shifted)
    total = _sb((rows, count), nl.float32)
    nisa.tensor_reduce(dst=total, op=nl.add, data=shifted.permute((0, 2, 1)), axis=2)
    recip = _sb((rows, count), nl.float32)
    nisa.reciprocal(dst=recip, data=total)
    w = _sb((rows, count), nl.float32)
    nisa.tensor_tensor(dst=w, data1=lse, data2=top, op=nl.subtract)
    nisa.activation(dst=w, op=nl.exp, data=w)
    nisa.tensor_tensor(dst=w, data1=w, data2=recip, op=nl.multiply)
    return w


def _partials(states, factor, dead, ranks, heads, dup):
    """Each sequence's partial ``acc * factor`` with the latent on partitions,
    ``[128, ranks, count, 4, heads]`` fp32; a sequence with no key on the rank is 0, by select."""
    count, rows = len(states), states[0][2].shape[0]
    gathered = ranks * heads
    parts = _sb((_P, ranks, count, _V_TILES, heads), nl.float32)
    for s, (_, _, acc) in enumerate(states):
        o = _sb((rows, _KV_LORA), nl.float32)
        nisa.tensor_scalar(dst=o, data=acc[0:rows, :], op0=nl.multiply, operand0=factor[:, s:s + 1])
        tp = _ps((_P, _V_TILES, rows), nl.float32)
        for t in range(_V_TILES):
            nisa.nc_transpose(dst=tp[:, t, :], data=o[:, t * _P:(t + 1) * _P])
        dst = parts[:, :, s, :, :]
        hi = tp[:, :, 0:gathered].reshape_dim(2, (ranks, heads)).permute((0, 2, 1, 3))
        if dup is None:
            nisa.tensor_copy(dst=dst, src=hi)
        else:
            # The high and low parts' sums, on rows 0.. and dup.., meet in the free dimension.
            lo = _sb((_P, _V_TILES, gathered), nl.float32)
            nisa.tensor_copy(dst=lo, src=tp[:, :, dup:dup + gathered])
            nisa.tensor_tensor(dst=dst, data1=hi, op=nl.add,
                               data2=lo.reshape_dim(2, (ranks, heads)).permute((0, 2, 1, 3)))
        nisa.tensor_copy_predicated(
            dst=dst, src=0.0, predicate=dead[:, s:s + 1].broadcast(1, ranks * _V_TILES * heads)
            .reshape_dim(1, (ranks, _V_TILES, heads)))
    return parts


def _reduce_scatter(parts, first, count, tokens, heads, group, ranks, write):
    """The group's sum of the weighted partials of this rank's heads, ``[128, count, 4, heads]``.

    Rank ``r``'s chunk of the reduce-scatter is ``[128, T, 4, heads]`` for its heads."""
    per = _V_TILES * heads
    chunk = _P * tokens * per
    src = _shared((ranks, chunk), nl.float32, "partial_scatter_src")
    dst = _shared((1, chunk), nl.float32, "partial_scatter_dst")
    if write:
        nisa.dma_copy(dst=src.ap(pattern=[[tokens * per, _P], [chunk, ranks], [1, count * per]],
                                 offset=first * per), src=parts.flatten_dims(2, 4))
    ncc.reduce_scatter(srcs=[src], dsts=[dst], replica_group=group, collective_dim=0, op=nl.add)
    o = _sb((_P, count, _V_TILES, heads), nl.float32)
    nisa.dma_copy(dst=o.flatten_dims(1, 3),
                  src=dst.ap(pattern=[[tokens * per, _P], [1, count * per]], offset=first * per))
    return o


def _all_to_all(parts, lse, first, count, tokens, heads, group, ranks, write):
    """This rank's heads over the group: every rank's normalized partial weighed by
    ``exp(LSE - group LSE)`` and summed, ``[128, count, 4, heads]`` fp32.

    Rank ``r``'s chunk of the all-to-all is the partial ``[128, T, 4, heads]`` of its heads,
    then their LSE ``[T, heads]``."""
    per = _V_TILES * heads
    body = _P * tokens * per
    chunk = body + tokens * heads
    src = _shared((ranks, chunk), nl.float32, "partial_exchange_src")
    dst = _shared((ranks, chunk), nl.float32, "partial_exchange_dst")
    if write:
        nisa.dma_copy(dst=src.ap(pattern=[[tokens * per, _P], [chunk, ranks], [1, count * per]],
                                 offset=first * per), src=parts.flatten_dims(2, 4))
        for r in range(ranks):
            nisa.dma_copy(dst=src.ap(pattern=[[1, heads], [heads, count]],
                                     offset=r * chunk + body + first * heads),
                          src=lse[r * heads:(r + 1) * heads, :])
    ncc.all_to_all(srcs=[src], dsts=[dst], replica_group=group, collective_dim=0)
    got = _sb((_P, ranks, count * per), nl.float32)
    nisa.dma_copy(dst=got, src=dst.ap(pattern=[[tokens * per, _P], [chunk, ranks],
                                               [1, count * per]], offset=first * per))
    every = _sb((_P, ranks, count * heads), nl.float32)  # each rank's LSE on every partition
    nisa.dma_copy(dst=every, src=dst.ap(pattern=[[0, _P], [chunk, ranks], [1, count * heads]],
                                        offset=body + first * heads))
    top = _sb((_P, count * heads), nl.float32)
    nisa.tensor_reduce(dst=top, op=nl.maximum, data=every.permute((0, 2, 1)), axis=2)
    nisa.tensor_tensor(dst=every, data1=every, data2=top.expand_dim(1).broadcast(1, ranks),
                       op=nl.subtract)
    nisa.activation(dst=every, op=nl.exp, data=every)
    total = _sb((_P, count * heads), nl.float32)
    nisa.tensor_reduce(dst=total, op=nl.add, data=every.permute((0, 2, 1)), axis=2)
    recip = _sb((_P, count * heads), nl.float32)
    nisa.reciprocal(dst=recip, data=total)

    def by_tile(t):  # [P, count * heads] as [P, count, 4, heads], the same for each latent tile
        return t.reshape_dim(1, (count, 1, heads)).broadcast(2, _V_TILES)
    o = _sb((_P, count, _V_TILES, heads), nl.float32)
    term = _sb((_P, count, _V_TILES, heads), nl.float32)
    for r in range(ranks):
        part = got[:, r, :].reshape_dim(1, (count, _V_TILES, heads))
        nisa.tensor_tensor(dst=o if r == 0 else term, data1=part, data2=by_tile(every[:, r, :]),
                           op=nl.multiply)
        if r > 0:
            nisa.tensor_tensor(dst=o, data1=o, data2=term, op=nl.add)
    nisa.tensor_tensor(dst=o, data1=o, data2=by_tile(recip), op=nl.multiply)
    return o


def mla_paged_decode_dcp_kernel(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages,
                                block_table, context_lens, slots, owner, EPS: float,
                                LATENT_OFFSET: int, BLOCK_SIZE: int, DCP_GROUPS: tuple,
                                MERGE: str):
    """One decode step on one rank of a DCP group: ``(out [128, T, H, 1] bf16, pages)``.

    Args:
        qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages, slots, EPS, LATENT_OFFSET,
            BLOCK_SIZE: as ``mla_paged_decode_kernel``, for the rank's ``H`` heads.
        block_table: ``[T, pages]`` int32 ids of the rank's own blocks of each sequence.
        context_lens: ``[T]`` int32 cached tokens of each sequence that the rank holds.
        owner: ``[T]`` int32, 1 where the rank holds the new token's row, else 0.
        DCP_GROUPS: every DCP group's ranks, the same on every rank, so all groups share one
            graph; the collectives run within each group, and the rank at place ``r`` of its
            group holds gathered heads ``r H ..``.
        MERGE: ``"all_to_all"`` or ``"reduce_scatter"``.

    Returns:
        The gated attention output of the rank's heads and the page buffer with the new rows.
    """
    tokens = qkv_a.shape[0]
    heads = _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, slots)
    cache, by_row, first_elem, first_row = _page_views(pages, LATENT_OFFSET, BLOCK_SIZE)
    blocks, block_size = pages.shape[0], BLOCK_SIZE
    page_rows = pages.shape[1] // _ROW_BYTES
    n_pages = block_table.shape[1]
    kernel_assert(tuple(block_table.shape) == (tokens, n_pages)
                  and tuple(context_lens.shape) == (tokens,) and tuple(owner.shape) == (tokens,),
                  "block_table is [T, pages]; context_lens and owner are [T]")
    groups = [list(g) for g in DCP_GROUPS]
    ranks = len(groups[0])
    gathered = ranks * heads
    kernel_assert(all(len(g) == ranks for g in groups), "every DCP group has the same size")
    kernel_assert(ranks >= 2 and gathered <= _P, "2 or more ranks, at most 128 gathered heads")
    kernel_assert(MERGE in ("all_to_all", "reduce_scatter"), "MERGE names a merge collective")
    group = ncc.ReplicaGroup(groups)
    window = n_pages * block_size
    n_tiles = window // _P
    n_blocks = (window + _BLOCK - 1) // _BLOCK
    programs = nl.num_programs(0)
    prg = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    scale = float((_NOPE + _ROPE) ** -0.5)
    out = nl.ndarray((_V, tokens, heads, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    split = programs > tokens
    if split:
        first, count = 0, tokens
        b0 = prg * (n_blocks // 2)
        b1 = n_blocks // 2 if prg == 0 else n_blocks
        lead = prg == 0
    else:
        base, extra = tokens // programs, tokens % programs
        first = prg * base + min(prg, extra)
        count = base + (1 if prg < extra else 0)
        b0, b1, lead = 0, n_blocks, True

    # The input, the gains and the metadata load first, ahead of the weights, so the query
    # chain starts while the weights land.
    x = _sb((count, _QKV), nl.float32)
    nisa.dma_copy(dst=x, src=qkv_a[first:first + count, :])
    raw = _sb((1, count * n_pages), nl.int32)
    nisa.dma_copy(dst=raw, src=block_table.ap(pattern=[[count * n_pages, 1], [1, count * n_pages]],
                                              offset=first * n_pages), **_META_QUEUE)
    q_gain = _gain(q_norm, count)
    held_n = _sb((_P, count), nl.int32)
    nisa.dma_copy(dst=held_n, src=context_lens.ap(pattern=[[0, _P], [1, count]], offset=first),
                  **_META_QUEUE)
    flag = _sb((_P, count), nl.int32)
    nisa.dma_copy(dst=flag, src=owner.ap(pattern=[[0, _P], [1, count]], offset=first),
                  **_META_QUEUE)
    kv_gain = _gain(kv_norm, count)
    wqb, wuk, last = _load_weights_first(w_qb_t, w_uk, heads)
    # The program's sequences' block ids, clamped into the cache. The slots load before the
    # pages, since each page load holds its engine about 1.2 us.
    ids = _sb((1, count * n_pages), nl.int32)
    nisa.tensor_scalar(dst=ids, data=raw, op0=nl.maximum, operand0=0, op1=nl.minimum,
                       operand1=blocks - 1)
    if lead:
        idx = _row_index(slots, first, count, blocks, block_size, page_rows, first_row)
    # Query rows 0.. hold the gathered heads in rank order, and rows dup.. again where both fit.
    dup = _dup_row(gathered)
    rows_q = gathered if dup is None else dup + gathered
    qloc = _sb((_P, len(_C_TILES), count, heads), nl.bfloat16)
    ql = [_sb((_P, count, rows_q), nl.bfloat16) for _ in _C_TILES]
    # own[:, s] is 1 where the new row is a key here; dead[:, s] where the rank has no key;
    # invalid[r, j] where window row 128 j + r lies at or past the rank's cached length.
    own = _sb((_P, count), nl.float32)
    bound = _sb((_P, count), nl.float32)
    seqs = []
    for s in range(count):
        group_s = {"rows": rows_q, "split": "passes" if dup is None else "rows", "dup": dup,
                   "ql": [ql[i][0:w, s, :] for i, (_, w) in enumerate(_C_TILES)],
                   "cache": bound[:, s:s + 1], "own": own[:, s:s + 1], "own_tiles": lambda k: 1,
                   "m": None, "l": None, "acc": None}
        seqs.append((group_s, ids[0:1, s * n_pages:(s + 1) * n_pages],
                     _sb((_P, n_tiles), nl.uint8)))
    steps = [(g, "cache", b * _BLOCK,
              _cache_step(cache, held, invalid, b, window, block_size, first_elem, _PAGE_QUEUE)())
             for b in range(b0, b1) for g, held, invalid in seqs]
    step_keys = [min(_BLOCK, window - b * _BLOCK) for b in range(b0, b1) for _ in seqs]
    wuv = _load_values(w_uv_t, heads, last)

    _absorbed_queries([qloc[:, i, :, :] for i in range(len(_C_TILES))], True, x, q_gain, wqb, wuk,
                      EPS, heads, k_outer=True)
    lat = _sb((count, _LATENT), nl.bfloat16)
    _latent_rows(lat, x, kv_gain, EPS)
    positions = _sb((_P, n_tiles), nl.float32)
    nisa.iota(dst=positions, pattern=[[_P, n_tiles]], offset=0, channel_multiplier=1)
    zero = _sb((_P, 1), nl.float32)
    nisa.memset(dst=zero, value=0.0)
    nisa.tensor_scalar(dst=own, data=flag, op0=nl.maximum, operand0=0.0, op1=nl.minimum,
                       operand1=1.0)
    keys = _sb((_P, count), nl.float32)
    nisa.scalar_tensor_tensor(dst=keys, data=held_n, op0=nl.maximum, operand0=0.0, op1=nl.add,
                              operand1=own)
    dead = _sb((_P, count), nl.uint8)
    nisa.tensor_scalar(dst=dead, data=keys, op0=nl.less, operand0=0.5)
    nisa.tensor_scalar(dst=bound, data=held_n, op0=nl.maximum, operand0=0.0, op1=nl.minimum,
                       operand1=float(window))
    for s, (_, _, invalid) in enumerate(seqs):
        nisa.tensor_scalar(dst=invalid, data=positions, op0=nl.greater_equal,
                           operand0=bound[:, s:s + 1])
    if lead:
        steps += [(g, "own", 0, _own_step(lat, s)()) for s, (g, _, _) in enumerate(seqs)]
        step_keys += [1] * count
    gather, read = _gather_queries(qloc, first, count, tokens, heads, group, ranks, lead)

    def queries():
        q = read()
        for i, (_, w) in enumerate(_C_TILES):
            mine = ql[i][0:w, :, 0:gathered].reshape_dim(2, (ranks, heads))
            nisa.tensor_copy(dst=mine, src=q[0:w, i, :, :].reshape_dim(2, (count, heads))
                             .permute((0, 2, 1, 3)))
            if dup is not None:
                nisa.memset(dst=ql[i][0:w, :, gathered:dup], value=0.0)
                nisa.tensor_copy(dst=ql[i][0:w, :, dup:rows_q], src=ql[i][0:w, :, 0:gathered])

    free = 0
    while free < len(steps) and sum(step_keys[:free + 1]) <= _FREE_KEYS:
        free += 1
    _gated_pipeline(steps, scale, zero, gather, free, queries)

    states = [(g["m"], g["l"], g["acc"]) for g, _, _ in seqs]
    if split:
        states = [_merge_programs(states[0], prg, scale, rows_q)]
    # LSE = scale m + ln l per query row; a rank with no key takes the dead LSE and sum 1.
    stats = _sb((rows_q, 2, count), nl.float32)
    for s, (m, l, _) in enumerate(states):
        nisa.tensor_copy(dst=stats[:, 0, s:s + 1], src=m[0:rows_q, :])
        nisa.tensor_copy(dst=stats[:, 1, s:s + 1], src=l[0:rows_q, :])
    lse = _sb((rows_q, count), nl.float32)
    nisa.activation(dst=lse, op=nl.log, data=stats[:, 1, :])
    nisa.scalar_tensor_tensor(dst=lse, data=stats[:, 0, :], op0=nl.multiply, operand0=scale,
                              op1=nl.add, operand1=lse)
    nisa.tensor_copy_predicated(dst=lse, src=_DEAD_LSE, predicate=dead[0:rows_q, :])
    total = stats[:, 1, :]
    nisa.tensor_copy_predicated(dst=total, src=1.0, predicate=dead[0:rows_q, :])
    factor = _sb((rows_q, count), nl.float32)
    nisa.reciprocal(dst=factor, data=total)
    if MERGE == "reduce_scatter":
        w = _group_weights(lse, first, count, tokens, gathered, dup, group, ranks, lead)
        nisa.tensor_tensor(dst=factor, data1=factor, data2=w, op=nl.multiply)
    parts = _partials(states, factor, dead, ranks, heads, dup)
    if MERGE == "reduce_scatter":
        o = _reduce_scatter(parts, first, count, tokens, heads, group, ranks, lead)
    else:
        o = _all_to_all(parts, lse, first, count, tokens, heads, group, ranks, lead)
    if split and prg != 0:
        return out, pages

    res = _sb((_V, count, heads), nl.bfloat16)
    _gated_output_t(res, o.permute((0, 2, 3, 1)), wuv, _gate_t(gate, first, count, heads), heads)
    nisa.dma_copy(dst=out.ap(pattern=[[tokens * heads, _V], [heads, count], [1, heads]],
                             offset=first * heads), src=res)
    # The new rows sit at or past the rank's cached rows, so every read above masks them.
    _write_rows(by_row, idx, lat, count)
    return out, pages


def _finite_values(lat):
    """The value columns of latent rows ``[rows, 512]``, each non-finite element set to 0.

    A masked key weighs exactly 0 in the weighted sum, but 0 times a NaN value is NaN. The
    scores still read the rows as they are, so a query that sees a non-finite key turns NaN.
    """
    rows = lat.shape[0]
    bad = _sb((rows, _KV_LORA), nl.uint8)
    nisa.tensor_scalar(dst=bad, data=lat[:, 0:_KV_LORA], op0=nl.multiply, operand0=0.0,
                       op1=nl.not_equal, operand1=0.0)
    val = _sb((rows, _KV_LORA), nl.bfloat16)
    nisa.tensor_copy(dst=val, src=lat[:, 0:_KV_LORA])
    nisa.tensor_copy_predicated(dst=val, src=0.0, predicate=bad)
    return val


def _prefill_chunk(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, source, source_offset,
                   pages, block_table, prefix_len, slots, EPS, LATENT_OFFSET, BLOCK_SIZE):
    """One prefill chunk that reads its cached prefix from the page buffer ``source`` and stores
    its latent rows at their slots of the page buffer ``pages``: ``(out, pages)``."""
    tokens = qkv_a.shape[0]
    heads = _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, slots)
    cache, _, first_elem, _ = _page_views(source, source_offset, BLOCK_SIZE)
    _, by_row, _, first_row = _page_views(pages, LATENT_OFFSET, BLOCK_SIZE)
    blocks, block_size = pages.shape[0], BLOCK_SIZE
    page_rows = pages.shape[1] // _ROW_BYTES
    n_pages = block_table.shape[0]
    kernel_assert(tuple(prefix_len.shape) == (1,), "prefix_len is [1]")
    width = heads * _V
    window = n_pages * block_size
    n_tiles = window // _P
    n_blocks = (window + _BLOCK - 1) // _BLOCK
    n_q = (tokens + _P - 1) // _P
    programs = nl.num_programs(0)
    prg = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    scale = float((_NOPE + _ROPE) ** -0.5)
    out = nl.ndarray((tokens, width), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_groups = [list(range(g, min(n_q, g + _Q_GROUP))) for g in range(0, n_q, _Q_GROUP)]
    mine = q_groups[prg::programs]
    if not mine:
        return out, pages
    my_tiles = [i for grp in mine for i in grp]

    wqb, wuk, wuv = _load_weights(w_qb_t, w_uk, w_uv_t, heads)
    q_gain = _gain(q_norm, _P)
    kv_gain = _gain(kv_norm, _P)

    def rows_of(i):
        return min(_P, tokens - i * _P)

    def load_rows(i):
        x = _sb((rows_of(i), _QKV), nl.float32)
        nisa.dma_copy(dst=x, src=qkv_a[i * _P:i * _P + rows_of(i), :])
        return x

    # The chunk's latent rows, the causal keys, up to this program's last query tile.
    lats, idx = [], {}
    for i in range(my_tiles[-1] + 1):
        rows = rows_of(i)
        lat = (_zeros if rows % 2 else _sb)((_even(rows), _LATENT), nl.bfloat16)
        _latent_rows(lat, load_rows(i), kv_gain[0:rows, :], EPS)
        if i in my_tiles:
            idx[i] = _row_index(slots, i * _P, rows, blocks, block_size, page_rows, first_row)
        lats.append(lat)

    positions = _sb((_P, n_tiles), nl.float32)
    nisa.iota(dst=positions, pattern=[[_P, n_tiles]], offset=0, channel_multiplier=1)
    zero = _sb((_P, 1), nl.float32)
    nisa.memset(dst=zero, value=0.0)
    held, bound, invalid = _cache_keys(
        block_table.ap(pattern=[[n_pages, 1], [1, n_pages]], offset=0),
        prefix_len.ap(pattern=[[0, _P], [1, 1]], offset=0), source.shape[0], n_pages, window,
        positions)
    tiles_per_block = _BLOCK // _P

    for grp in mine:
        groups = []
        for i in grp:
            rows = rows_of(i)
            ql = [(_zeros if rows % 2 else _sb)((_P, heads, _even(rows)), nl.bfloat16)
                  for _ in _C_TILES]
            _absorbed_queries(ql, False, load_rows(i), q_gain[0:rows, :], wqb, wuk, EPS, heads)
            causal = _sb((_P, 1), nl.float32)
            nisa.iota(dst=causal, pattern=[[0, 1]], offset=i * _P + 1, channel_multiplier=1)
            for h in range(heads):
                groups.append({
                    "rows": _even(rows), "tile": i, "head": h, "split": "passes",
                    "ql": [ql[c][0:w, h, :] for c, (_, w) in enumerate(_C_TILES)],
                    "cache": bound, "own": causal,
                    "own_tiles": (lambda k, i=i: min(tiles_per_block, i + 1 - k // _P)),
                    "m": None, "l": None, "acc": None})
        for b in range(n_blocks):
            n_t = min(_BLOCK, window - b * _BLOCK) // _P
            tiles = _clear_unowned(_cache_pages(cache, held, b * tiles_per_block, n_t, block_size,
                                                first_elem), invalid, b * tiles_per_block)
            _attend(groups, "cache", tiles, b * _BLOCK, scale, zero)
        if grp is mine[-1]:
            # The chunk's rows sit past the prefix, so the reads mask them; they go out after
            # the last page reads, under this group's causal steps.
            for i in my_tiles:
                _write_rows(by_row, idx[i], lats[i], rows_of(i))
        for c0 in range(0, grp[-1] + 1, tiles_per_block):
            c1 = min(grp[-1] + 1, c0 + tiles_per_block)
            # A query tile masks keys only in its own tile, so only there the sum needs finite rows.
            values = [_finite_values(lats[j]) if j in grp else lats[j] for j in range(c0, c1)]
            _attend([g for g in groups if g["tile"] >= c0], "own", lats[c0:c1], c0 * _P,
                    scale, zero, values)
        for i in grp:
            rows = rows_of(i)
            o_t = _sb((_P, _V_TILES, heads, rows), nl.float32)
            for g in groups:
                if g["tile"] == i:
                    _transpose_into(o_t[:, :, g["head"], :], _normalized(g["l"], g["acc"], rows),
                                    rows)
            res = _sb((rows, width), nl.bfloat16)
            _gated_output(res, o_t, wuv, _gate(gate, i * _P, rows, width), heads)
            nisa.dma_copy(dst=out[i * _P:i * _P + rows, :], src=res)
    return out, pages


def mla_paged_prefill_kernel(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages,
                             block_table, prefix_len, slots, EPS: float, LATENT_OFFSET: int,
                             BLOCK_SIZE: int):
    """One prefill chunk of one sequence: ``(out [T, H * 128] bf16, pages)``.

    Args:
        qkv_a: ``[T, 2112]`` fp32 rows of the chunk's tokens: ``q_a`` then ``kv_a``.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages: as ``mla_paged_decode_kernel``.
        block_table: ``[pages]`` int32 block ids of the sequence's cached prefix.
        prefix_len: ``[1]`` int32 cached tokens before the chunk.
        slots: ``[T]`` int32 slot of each chunk token; -1 for a padding token.
        EPS, LATENT_OFFSET, BLOCK_SIZE: as ``mla_paged_decode_kernel``.

    Returns:
        The gated attention output and the page buffer with the chunk's rows.
    """
    return _prefill_chunk(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages,
                          LATENT_OFFSET, pages, block_table, prefix_len, slots, EPS,
                          LATENT_OFFSET, BLOCK_SIZE)


def mla_paged_prefill_dcp_kernel(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages,
                                 window, window_table, prefix_len, slots, EPS: float,
                                 LATENT_OFFSET: int, BLOCK_SIZE: int):
    """One prefill chunk of one sequence on one rank of a DCP group: ``(out, pages)``.

    Args:
        pages: the rank's own page buffer, as ``mla_paged_decode_kernel``'s; the chunk's rows
            go to their slots in it.
        window: ``[window blocks, BLOCK_SIZE * 1152]`` int8, the sequence's latent blocks
            gathered over the group, from byte 0 of each page; read only.
        window_table: ``[window blocks]`` int32 window block of each block of the sequence.
        slots: ``[T]`` int32 slot of each chunk token in ``pages``.
        Others: as ``mla_paged_prefill_kernel``.

    Returns:
        The gated attention output and the rank's page buffer with the chunk's rows.
    """
    return _prefill_chunk(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, window, 0, pages,
                          window_table, prefix_len, slots, EPS, LATENT_OFFSET, BLOCK_SIZE)


_DECODE = wrap_nki(mla_paged_decode_kernel)
_DECODE_DCP = wrap_nki(mla_paged_decode_dcp_kernel)
_PREFILL = wrap_nki(mla_paged_prefill_kernel)
_PREFILL_DCP = wrap_nki(mla_paged_prefill_dcp_kernel)


def prepare_mla_weights(q_b_weight: torch.Tensor, kv_b_weight: torch.Tensor):
    """Kernel layouts of a rank's ``q_b`` and ``kv_b`` slices, built once at load.

    Args:
        q_b_weight: ``[H * 192, 1536]`` bf16 ``q_b_proj`` rows of the rank's heads.
        kv_b_weight: ``[H * 256, 512]`` bf16 ``kv_b_proj`` rows of the rank's heads.

    Returns:
        ``(w_qb_t [1536, H * 192] bf16, w_uk [H, 128, 512] bf16, w_uv_t [H, 512, 128] fp32)``.
    """
    heads = kv_b_weight.shape[0] // (_NOPE + _V)
    kv = kv_b_weight.reshape(heads, _NOPE + _V, _KV_LORA)
    return (q_b_weight.t().contiguous(), kv[:, :_NOPE].contiguous(),
            kv[:, _NOPE:].transpose(1, 2).float().contiguous())


def _check_pages(pages, latent_offset, block_size, name="pages"):
    # The dispatch drops strides, so a strided view would reach the kernel as a copy.
    if pages.dim() != 2 or pages.dtype != torch.int8 or not pages.is_contiguous():
        raise ValueError(f"{name} must be the contiguous int8 [blocks, page bytes] page buffer, "
                         f"got {tuple(pages.shape)} {pages.dtype} strides {pages.stride()}")
    if pages.shape[1] % _ROW_BYTES or latent_offset % _ROW_BYTES or latent_offset < 0:
        raise ValueError(f"page bytes {pages.shape[1]} and latent offset {latent_offset} must be "
                         f"whole {_ROW_BYTES}-byte rows")
    if block_size % _P or block_size & (block_size - 1):
        raise ValueError(f"block_size {block_size} must be a power of two, a multiple of {_P}")
    if latent_offset + block_size * _ROW_BYTES > pages.shape[1]:
        raise ValueError(f"a {block_size}-row latent block from byte {latent_offset} does not fit "
                         f"a {pages.shape[1]}-byte page")


def _check_inputs(qkv_a, gate, pages, latent_offset, block_size, slots, w_uk, w_uv_t):
    heads = w_uk.shape[0]
    if qkv_a.dim() != 2 or qkv_a.shape[1] != _QKV or qkv_a.dtype != torch.float32:
        raise ValueError(f"qkv_a must be [T, {_QKV}] fp32, got {tuple(qkv_a.shape)} {qkv_a.dtype}")
    if tuple(gate.shape) != (qkv_a.shape[0], heads * _V) or gate.dtype != torch.bfloat16:
        raise ValueError(f"gate must be [T, {heads * _V}] bf16, "
                         f"got {tuple(gate.shape)} {gate.dtype}")
    _check_pages(pages, latent_offset, block_size)
    if slots.dim() != 1 or slots.shape[0] != qkv_a.shape[0]:
        raise ValueError(f"slots must be [T], got {tuple(slots.shape)}")
    if w_uv_t.dtype != torch.float32:
        raise ValueError(f"w_uv_t must be fp32, got {w_uv_t.dtype}")


def mla_paged_decode(qkv_a: torch.Tensor, gate: torch.Tensor, q_norm_weight: torch.Tensor,
                     kv_norm_weight: torch.Tensor, w_qb_t: torch.Tensor, w_uk: torch.Tensor,
                     w_uv_t: torch.Tensor, pages: torch.Tensor, latent_offset: int,
                     block_size: int, block_table: torch.Tensor, context_lens: torch.Tensor,
                     slots: torch.Tensor, eps: float):
    """Paged MLA decode step on the NKI kernel: ``(out [128, T, H, 1] bf16, pages)``.

    Args:
        qkv_a: ``[T, 2112]`` fp32 all-reduced ``q_a`` and ``kv_a`` of each sequence's new token.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm_weight: ``[1536]`` ``q_a_layernorm`` gain.
        kv_norm_weight: ``[512]`` ``kv_a_layernorm`` gain.
        w_qb_t, w_uk, w_uv_t: the layouts from :func:`prepare_mla_weights`.
        pages: ``[blocks, page bytes]`` contiguous int8 page buffer of the KV caches, updated in
            place: each page's latent block ``[block_size, 576]`` bf16 starts at
            ``latent_offset``.
        latent_offset: byte offset of the latent block in a page, a multiple of 1152.
        block_size: tokens per block, a power of two and a multiple of 128.
        block_table: ``[T, pages]`` int32 block ids per sequence.
        context_lens: ``[T]`` int32 tokens already cached per sequence.
        slots: ``[T]`` slot of each new token; -1 for a padding token, which writes block 0.
        eps: the norms' epsilon.

    Returns:
        The output for ``o_proj`` and the page buffer with each new token's latent row.

    Raises:
        ValueError: an input outside the kernel's contract.
    """
    _check_inputs(qkv_a, gate, pages, latent_offset, block_size, slots, w_uk, w_uv_t)
    if block_table.dim() != 2 or block_table.shape[0] != qkv_a.shape[0]:
        raise ValueError(f"block_table must be [T, pages], got {tuple(block_table.shape)}")
    return _DECODE[get_logical_nc_config()](
        qkv_a=qkv_a.contiguous(), gate=gate.contiguous(), q_norm=q_norm_weight.contiguous(),
        kv_norm=kv_norm_weight.contiguous(), w_qb_t=w_qb_t, w_uk=w_uk, w_uv_t=w_uv_t,
        pages=pages, block_table=block_table.to(torch.int32).contiguous(),
        context_lens=context_lens.to(torch.int32).contiguous(),
        slots=slots.to(torch.int32).contiguous(), EPS=float(eps),
        LATENT_OFFSET=int(latent_offset), BLOCK_SIZE=int(block_size))


def mla_paged_prefill(qkv_a: torch.Tensor, gate: torch.Tensor, q_norm_weight: torch.Tensor,
                      kv_norm_weight: torch.Tensor, w_qb_t: torch.Tensor, w_uk: torch.Tensor,
                      w_uv_t: torch.Tensor, pages: torch.Tensor, latent_offset: int,
                      block_size: int, block_table: torch.Tensor, prefix_len: torch.Tensor,
                      slots: torch.Tensor, eps: float):
    """Paged MLA prefill chunk of one sequence on the NKI kernel, in 128-row query tiles.

    Args:
        qkv_a: ``[T, 2112]`` fp32 all-reduced ``q_a`` and ``kv_a`` of the chunk's tokens.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, pages, latent_offset,
            block_size: as :func:`mla_paged_decode`.
        block_table: ``[pages]`` int32 block ids of the cached prefix.
        prefix_len: ``[1]`` int32 tokens cached before the chunk.
        slots: ``[T]`` slot of each chunk token; -1 for a padding token.
        eps: the norms' epsilon.

    Returns:
        ``(out [T, H * 128] bf16, pages)``, the buffer with the chunk's latent rows.

    Raises:
        ValueError: an input outside the kernel's contract.
    """
    _check_inputs(qkv_a, gate, pages, latent_offset, block_size, slots, w_uk, w_uv_t)
    if block_table.dim() != 1 or prefix_len.numel() != 1:
        raise ValueError(f"block_table must be [pages] and prefix_len [1], got "
                         f"{tuple(block_table.shape)} and {tuple(prefix_len.shape)}")
    return _PREFILL[get_logical_nc_config()](
        qkv_a=qkv_a.contiguous(), gate=gate.contiguous(), q_norm=q_norm_weight.contiguous(),
        kv_norm=kv_norm_weight.contiguous(), w_qb_t=w_qb_t, w_uk=w_uk, w_uv_t=w_uv_t,
        pages=pages, block_table=block_table.to(torch.int32).contiguous(),
        prefix_len=prefix_len.to(torch.int32).reshape(1).contiguous(),
        slots=slots.to(torch.int32).contiguous(), EPS=float(eps),
        LATENT_OFFSET=int(latent_offset), BLOCK_SIZE=int(block_size))


def _torch_prefill_dcp(qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, pages,
                       latent_offset, block_size, window, window_table, prefix_len, slots, eps):
    """The DCP prefill chunk in torch, for the CPU venue where the kernel does not run."""
    tokens, heads = qkv_a.shape[0], w_uk.shape[0]
    x = qkv_a.float()

    def norm(v, weight):
        return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * weight.float()

    q = (norm(x[:, :_Q_LORA], q_norm_weight) @ w_qb_t.float()).view(tokens, heads, -1)
    q_lift = torch.cat((torch.einsum("thn,hnl->thl", q[..., :_NOPE], w_uk.float()),
                        q[..., _NOPE:]), dim=-1).bfloat16()
    lat = torch.cat((norm(x[:, _Q_LORA:_Q_LORA + _KV_LORA], kv_norm_weight),
                     x[:, _Q_LORA + _KV_LORA:]), dim=-1).bfloat16()
    prefix = int(prefix_len)
    ids = window_table.long().clamp(0, window.shape[0] - 1)
    cached = window.view(torch.bfloat16).view(window.shape[0], -1, _LATENT)[ids, :block_size]
    keys = torch.cat((cached.reshape(-1, _LATENT)[:prefix], lat)).float()
    scores = torch.einsum("thl,kl->thk", q_lift.float(), keys) * (_NOPE + _ROPE) ** -0.5
    seen = torch.arange(keys.shape[0]) <= prefix + torch.arange(tokens).unsqueeze(1)
    weights = scores.masked_fill(~seen.unsqueeze(1), float("-inf")).softmax(-1)
    o = torch.einsum("thk,kl->thl", weights, keys[:, :_KV_LORA])
    out = torch.einsum("thl,hlv->thv", o, w_uv_t.float())
    out = out * torch.sigmoid(gate.float()).view(tokens, heads, _V)
    s = slots.long()
    s = torch.where((s < 0) | (s >= pages.shape[0] * block_size), 0, s)
    first = latent_offset // _ROW_BYTES
    target = s // block_size * (pages.shape[1] // _ROW_BYTES) + first + s % block_size
    pages.view(torch.bfloat16).view(-1, _LATENT)[target] = lat
    return out.reshape(tokens, -1).bfloat16(), pages


def mla_paged_prefill_dcp(qkv_a: torch.Tensor, gate: torch.Tensor, q_norm_weight: torch.Tensor,
                          kv_norm_weight: torch.Tensor, w_qb_t: torch.Tensor, w_uk: torch.Tensor,
                          w_uv_t: torch.Tensor, pages: torch.Tensor, latent_offset: int,
                          block_size: int, window: torch.Tensor, window_table: torch.Tensor,
                          prefix_len: torch.Tensor, slots: torch.Tensor, eps: float):
    """Paged MLA prefill chunk on one rank of a DCP group: ``(out [T, H * 128] bf16, pages)``.

    The chunk attends its cached prefix in ``window`` and its own rows, as
    :func:`mla_paged_prefill`, and stores each of its latent rows at its slot of ``pages``.

    Args:
        qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, pages, latent_offset,
            block_size, eps: as :func:`mla_paged_prefill`; ``pages`` is the rank's own buffer.
        window: ``[blocks, block_size * 1152]`` contiguous int8, the sequence's latent blocks
            gathered over the group, rank by rank; read only.
        window_table: ``[blocks]`` int32 window block of each block of the sequence.
        prefix_len: ``[1]`` int32 tokens cached before the chunk.
        slots: ``[T]`` slot of each chunk token in ``pages``: its own slot on the rank that
            holds it, the null block on the others, -1 for a padding token.

    Returns:
        The output for ``o_proj`` and the rank's page buffer with the chunk's latent rows.

    Raises:
        ValueError: an input outside the kernel's contract.
    """
    _check_inputs(qkv_a, gate, pages, latent_offset, block_size, slots, w_uk, w_uv_t)
    _check_pages(window, 0, block_size, "window")
    if window_table.dim() != 1 or prefix_len.numel() != 1:
        raise ValueError(f"window_table must be [blocks] and prefix_len [1], got "
                         f"{tuple(window_table.shape)} and {tuple(prefix_len.shape)}")
    if not can_run_kernel(qkv_a):
        return _torch_prefill_dcp(qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk,
                                  w_uv_t, pages, latent_offset, block_size, window, window_table,
                                  prefix_len, slots, eps)
    return _PREFILL_DCP[get_logical_nc_config()](
        qkv_a=qkv_a.contiguous(), gate=gate.contiguous(), q_norm=q_norm_weight.contiguous(),
        kv_norm=kv_norm_weight.contiguous(), w_qb_t=w_qb_t, w_uk=w_uk, w_uv_t=w_uv_t,
        pages=pages, window=window, window_table=window_table.to(torch.int32).contiguous(),
        prefix_len=prefix_len.to(torch.int32).reshape(1).contiguous(),
        slots=slots.to(torch.int32).contiguous(), EPS=float(eps),
        LATENT_OFFSET=int(latent_offset), BLOCK_SIZE=int(block_size))


def _torch_decode_dcp(qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, pages,
                      latent_offset, block_size, block_table, context_lens, slots, owner, eps,
                      dcp_group):
    """The DCP decode step in torch, for the CPU venue where the kernel does not run."""
    tokens, heads = qkv_a.shape[0], w_uk.shape[0]
    x = qkv_a.float()

    def norm(v, weight):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * weight.float()

    q = (norm(x[:, :_Q_LORA], q_norm_weight) @ w_qb_t.float()).view(tokens, heads, -1)
    q_lift = torch.cat((torch.einsum("thn,hnl->thl", q[..., :_NOPE], w_uk.float()),
                        q[..., _NOPE:]), dim=-1).bfloat16()
    lat = torch.cat((norm(x[:, _Q_LORA:_Q_LORA + _KV_LORA], kv_norm_weight),
                     x[:, _Q_LORA + _KV_LORA:]), dim=-1).bfloat16()
    queries = dcp_group.all_gather(q_lift.contiguous(), dim=1).float()
    rows = pages.view(torch.bfloat16).view(pages.shape[0], -1, _LATENT)
    first = latent_offset // _ROW_BYTES
    part = queries.new_zeros(tokens, queries.shape[1], _KV_LORA)
    lse = queries.new_full(queries.shape[:2], _DEAD_LSE)
    for t in range(tokens):
        blocks = block_table[t].long().clamp(0, pages.shape[0] - 1)
        keys = rows[blocks, first:first + block_size].reshape(-1, _LATENT)[:int(context_lens[t])]
        if int(owner[t]) > 0:
            keys = torch.cat((keys, lat[t:t + 1]))
        if keys.shape[0]:
            scores = queries[t] @ keys.float().T * (_NOPE + _ROPE) ** -0.5
            lse[t] = torch.logsumexp(scores, dim=-1)
            part[t] = torch.softmax(scores, dim=-1) @ keys[:, :_KV_LORA].float()
    group_lse = torch.logsumexp(dcp_group.all_gather(lse.contiguous(), dim=0)
                                .view(-1, *lse.shape), dim=0)
    o = dcp_group.reduce_scatter((part * torch.exp(lse - group_lse)[..., None]).contiguous(),
                                 dim=1)
    out = torch.einsum("thl,hlv->thv", o, w_uv_t) * torch.sigmoid(gate.float()).view(o.shape[0],
                                                                                   heads, _V)
    s = slots.long()
    s = torch.where((s < 0) | (s >= pages.shape[0] * block_size), 0, s)
    rows[s // block_size, first + s % block_size] = lat
    return out.permute(2, 0, 1).unsqueeze(-1).bfloat16().contiguous(), pages


def mla_paged_decode_dcp(qkv_a: torch.Tensor, gate: torch.Tensor, q_norm_weight: torch.Tensor,
                         kv_norm_weight: torch.Tensor, w_qb_t: torch.Tensor, w_uk: torch.Tensor,
                         w_uv_t: torch.Tensor, pages: torch.Tensor, latent_offset: int,
                         block_size: int, block_table: torch.Tensor, context_lens: torch.Tensor,
                         slots: torch.Tensor, owner: torch.Tensor, eps: float, dcp_group):
    """Paged MLA decode step of one rank of a DCP group: ``(out [128, T, H, 1] bf16, pages)``.

    The rank holds every group-size-th block of each sequence's cache. The merge returns the
    rank's ``H`` heads over the whole cache: an all-to-all from 4 to 15 ranks, otherwise an
    all-gather of the LSEs and a reduce-scatter of the partials.

    Args:
        qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, pages, latent_offset,
            block_size, slots, eps: as :func:`mla_paged_decode`, for the rank's heads; ``slots``
            names the rank's own slot of the new row, or a slot of the null block.
        block_table: ``[T, pages]`` int32 ids of the rank's own blocks per sequence.
        context_lens: ``[T]`` int32 tokens of each sequence that the rank holds.
        owner: ``[T]`` int32, 1 where the rank holds the new token's row, else 0.
        dcp_group: the DCP group coordinator; the kernel reads its ``ranks`` and every group of
            its mesh, and the CPU venue its ``all_gather`` and ``reduce_scatter``.

    Returns:
        The output for ``o_proj`` and the page buffer with each new token's latent row.

    Raises:
        ValueError: an input outside the kernel's contract, or a DCP mesh that does not split
            the world into groups of the group's size.
    """
    _check_inputs(qkv_a, gate, pages, latent_offset, block_size, slots, w_uk, w_uv_t)
    tokens = qkv_a.shape[0]
    if block_table.dim() != 2 or block_table.shape[0] != tokens:
        raise ValueError(f"block_table must be [T, pages], got {tuple(block_table.shape)}")
    if tuple(owner.shape) != (tokens,) or tuple(context_lens.shape) != (tokens,):
        raise ValueError(f"owner and context_lens must be [T], got {tuple(owner.shape)} and "
                         f"{tuple(context_lens.shape)}")
    ranks = tuple(int(r) for r in dcp_group.ranks)
    if not can_run_kernel(qkv_a):
        return _torch_decode_dcp(qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk,
                                 w_uv_t, pages, latent_offset, block_size, block_table,
                                 context_lens, slots, owner, eps, dcp_group)
    # Every group's ranks, the same on every rank, so all groups lower one graph.
    groups = tuple(tuple(int(r) for r in g) for g in resolve_full_mesh(dcp_group))
    flat = sorted([r for g in groups for r in g])
    # With no process group, as in a one-process test, the world is the mesh's own ranks.
    world = dist.get_world_size() if dist.is_initialized() else len(flat)
    if (ranks not in groups or any(len(g) != len(ranks) for g in groups)
            or flat != list(range(world))):
        raise ValueError(f"the DCP mesh {groups} must hold the group's ranks {ranks} and cover "
                         f"the world's {world} ranks once, in groups of one size")
    # All-to-all needs 4 ranks and fails at 16 or more.
    merge = "all_to_all" if 4 <= len(ranks) < 16 else "reduce_scatter"
    return _DECODE_DCP[get_logical_nc_config()](
        qkv_a=qkv_a.contiguous(), gate=gate.contiguous(), q_norm=q_norm_weight.contiguous(),
        kv_norm=kv_norm_weight.contiguous(), w_qb_t=w_qb_t, w_uk=w_uk, w_uv_t=w_uv_t,
        pages=pages, block_table=block_table.to(torch.int32).contiguous(),
        context_lens=context_lens.to(torch.int32).contiguous(),
        slots=slots.to(torch.int32).contiguous(), owner=owner.to(torch.int32).contiguous(),
        EPS=float(eps), LATENT_OFFSET=int(latent_offset), BLOCK_SIZE=int(block_size),
        DCP_GROUPS=groups, MERGE=merge)
