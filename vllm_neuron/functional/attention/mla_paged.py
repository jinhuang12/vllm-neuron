# SPDX-License-Identifier: Apache-2.0
"""Paged MLA attention for Kimi K3 in NKI: decode steps and 128-row prefill tiles.

One launch takes the all-reduced ``q_a`` and ``kv_a`` rows ``[T, 2112]`` fp32, the output
gate projection ``[T, H * 128]`` before its sigmoid, and the paged latent cache. It returns
the gated attention output and the tokens' latent rows ``[T, 576]`` bf16 for the cache
write. Decode writes the output head dimension first, ``[128, T, H, 1]`` bf16, the input
layout of the output projection; prefill writes it ``[T, H * 128]`` bf16. Per token, as the
checkpoint's ``KimiMLAAttention``: the ``q_a`` and ``kv_a`` RMS norms, ``q_b``, attention with
no rotary stage, and the sigmoid gate. The query is absorbed into the latent:
``q_lift = [W_UK^T q_nope, q_rope]``, 576 wide in bf16, scores the latent rows directly, and
``W_UV`` applies once, in fp32, to the softmax-weighted sum of the rows. The normed q and
``q_nope`` enter ``q_b`` and ``W_UK`` as bf16 high and low parts, and the fp32 softmax
weights enter the weighted sum the same way, so ``q_lift`` is the only bf16 rounding on the
query side.

Cache pages are token-major ``[blocks, block_size, 576]`` bf16, ``block_size`` a multiple of
128. PE transposes put the latent on the partitions for the scores; the pages themselves
move for the weighted sum. The softmax runs online over 512-key blocks in fp32. Rows past a
sequence's length are masked by fill: their scores become the fp32 minimum before the exp
and their latent becomes zero before the weighted sum, so no row a sequence does not own
reaches its output. A token's own latent row is one more key, computed in the launch.

At LNC 2 decode splits the sequences over the two programs, or the keys of a lone sequence,
merged over ``sendrecv``. Prefill takes one sequence: groups of 128-row query tiles
alternate between the programs, each over the paged prefix and the causal chunk.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config

_P = 128
_BLOCK = 512  # keys per online-softmax step: one fp32 PSUM bank of scores
_NOPE = 128
_ROPE = 64
_V = 128
_Q_LORA = 1536
_KV_LORA = 512
_LATENT = _KV_LORA + _ROPE
_QKV = _Q_LORA + _LATENT
_K_TILES = _Q_LORA // _P
_C_TILES = ((0, 128), (128, 128), (256, 128), (384, 128), (512, _ROPE))  # (first, width)
_V_TILES = _KV_LORA // _P
_MAX_FLOOR = -1.0e20  # running-max floor: keys filled below it weigh exactly 0
_FILL = -3.4028234663852886e38  # the fp32 minimum, range_select's only fill value
_Q_GROUP = 4  # prefill query tiles that share one pass over the keys
_KEYS_AHEAD = 3  # decode steps between a page's key transposes and its scores
_DUP = 32  # decode: partition of the low-part copy of each group's query rows


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


def _absorbed_queries(ql, by_token, x, q_gain, wqb, wuk, eps, heads):
    """``ql[i]``: ``q_lift^T`` of each head on latent tile ``i``, bf16, for the qkv_a rows ``x``.

    The normed q and ``q_nope`` enter their products unrounded, as bf16 high and low parts
    side by side in the moving operand, so each weight tile loads once; the parts' columns are
    summed after the product.
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
    for h in range(heads):
        base = h * (_NOPE + _ROPE)
        qp = _ps((_NOPE, n), nl.float32)
        for k in range(_K_TILES):
            nisa.nc_matmul(dst=qp, stationary=wqb[:, k, base:base + _NOPE], moving=qnt[:, k, :],
                           accumulate=k > 0)
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


def _cache_pages(cache, held, first_tile, n_tiles):
    """Window tiles ``first_tile ..`` of a sequence as one ``[128, n_tiles, 576]`` page tile."""
    block_size = cache.shape[1]
    per = block_size // _P
    pages = _sb((_P, n_tiles, _LATENT), cache.dtype)
    t = 0
    while t < n_tiles:
        b, r = (first_tile + t) // per, (first_tile + t) % per
        n = min(n_tiles - t, per - r)
        nisa.dma_copy(dst=pages[:, t:t + n, :],
                      src=cache.ap(pattern=[[_LATENT, _P], [_P * _LATENT, n], [1, _LATENT]],
                                   offset=r * _P * _LATENT, scalar_offset=held[0:1, b:b + 1],
                                   indirect_dim=0))
        t += n
    return pages


def _clear_unowned(pages, invalid, first_tile):
    """Zero the latent of rows the sequence does not own, so the weighted sum reads finite rows."""
    tiles = []
    for t in range(pages.shape[1]):
        j = first_tile + t
        nisa.tensor_copy_predicated(dst=pages[:, t, 0:_KV_LORA], src=0.0,
                                    predicate=invalid[:, j:j + 1].broadcast(1, _KV_LORA))
        tiles.append(pages[:, t, :])
    return tiles


def _keys_t(tiles):
    """The key pages with the latent on partitions: ``kt[i] [w_i, keys]`` bf16, by PE transposes."""
    rows_k = [t.shape[0] for t in tiles]
    width = _P * (len(tiles) - 1) + rows_k[-1]
    kt = []
    for i, (c0, w) in enumerate(_C_TILES):
        tp = _ps((w, width), nl.bfloat16)
        for t, page in enumerate(tiles):
            nisa.nc_transpose(dst=tp[:, t * _P:t * _P + rows_k[t]], data=page[:, c0:c0 + w])
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
        # Rows from _DUP repeat the group's queries: they carry the low part in the same tile.
        hi = _sb((r, cols), nl.bfloat16)
        nisa.tensor_copy(dst=hi, src=p)
        nisa.tensor_tensor(dst=hi[_DUP:r, :], data1=p[_DUP:r, :], data2=hi[_DUP:r, :],
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


def _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, cache):
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
    kernel_assert(cache.shape[2] == _LATENT and cache.dtype == nl.bfloat16,
                  "cache is [blocks, block_size, 576] bf16")
    kernel_assert(cache.shape[1] % _P == 0, "block_size is a multiple of 128")
    return heads


def mla_paged_decode_kernel(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, cache,
                            block_table, context_lens, EPS: float):
    """One decode step of ``T`` sequences: ``(out [128, T, H, 1] bf16, latent [T, 576] bf16)``.

    Args:
        qkv_a: ``[T, 2112]`` fp32, each sequence's new token: ``q_a`` then ``kv_a``.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm: ``[1536]`` ``q_a`` norm gain.
        kv_norm: ``[512]`` ``kv_a`` norm gain.
        w_qb_t: ``[1536, H * 192]`` bf16, the rank's ``q_b`` rows transposed.
        w_uk: ``[H, 128, 512]`` bf16, the ``k_nope`` rows of ``kv_b`` by head.
        w_uv_t: ``[H, 512, 128]`` fp32, the value rows of ``kv_b`` by head, transposed.
        cache: ``[blocks, block_size, 576]`` bf16 latent pages.
        block_table: ``[T, pages]`` int32 block ids; entries past a sequence's blocks are
            clamped into the cache and masked.
        context_lens: ``[T]`` int32 cached tokens per sequence; the new token follows them.
        EPS: the norms' epsilon.

    Returns:
        The gated attention output and the new tokens' latent rows.
    """
    tokens = qkv_a.shape[0]
    heads = _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, cache)
    blocks, block_size = cache.shape[0], cache.shape[1]
    pages = block_table.shape[1]
    kernel_assert(tuple(block_table.shape) == (tokens, pages)
                  and tuple(context_lens.shape) == (tokens,),
                  "block_table is [T, pages] and context_lens is [T]")
    width = heads * _V
    window = pages * block_size
    n_tiles = window // _P
    n_blocks = (window + _BLOCK - 1) // _BLOCK
    programs = nl.num_programs(0)
    prg = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    scale = float((_NOPE + _ROPE) ** -0.5)
    out = nl.ndarray((_V, tokens, heads, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    lat_out = nl.ndarray((tokens, _LATENT), dtype=nl.bfloat16, buffer=nl.shared_hbm)

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
            block_table.ap(pattern=[[pages, 1], [1, pages]], offset=seq * pages),
            context_lens.ap(pattern=[[0, _P], [1, 1]], offset=seq),
            blocks, pages, window, positions)
        group = {"rows": rows_q, "split": "rows",
                 "ql": [ql[i][0:w, s, :] for i, (_, w) in enumerate(_C_TILES)],
                 "cache": bound, "own": one, "own_tiles": lambda k: 1,
                 "m": None, "l": None, "acc": None}
        seqs.append((group, held, invalid))
    def cache_step(held, invalid, b):
        first_tile, n_t = b * (_BLOCK // _P), min(_BLOCK, window - b * _BLOCK) // _P

        def load():
            pages = _cache_pages(cache, held, first_tile, n_t)
            return lambda: _clear_unowned(pages, invalid, first_tile)
        return load

    def own_step(s):
        def load():
            row = _zeros((2, _LATENT), nl.bfloat16)  # key 1 is past the bound and zero
            nisa.dma_copy(dst=row[0:1, :], src=lat[s:s + 1, :])
            return lambda: [row]
        return load

    steps = [(group, "cache", b * _BLOCK, cache_step(held, invalid, b))
             for b in range(b0, b1) for group, held, invalid in seqs]
    if own:
        steps += [(group, "own", 0, own_step(s)) for s, (group, _, _) in enumerate(seqs)]
    # The query chain is emitted after the first pages' loads and transposes, which do not need it.
    def queries():
        _absorbed_queries(ql, True, x, q_gain, wqb, wuk, EPS, heads)
        for i, (_, w) in enumerate(_C_TILES):
            nisa.tensor_copy(dst=ql[i][0:w, :, _DUP:_DUP + heads], src=ql[i][0:w, :, 0:heads])

    _pipelined(steps, scale, zero, queries)

    states = [(g["m"], g["l"], g["acc"]) for g, _, _ in seqs]
    if split:
        m, l, acc = states[0]
        mine = _sb((rows_q, _KV_LORA + 8), nl.float32)
        nisa.tensor_copy(dst=mine[:, 0:_KV_LORA], src=acc[0:rows_q, :])
        nisa.tensor_copy(dst=mine[:, _KV_LORA:_KV_LORA + 1], src=m[0:rows_q, :])
        nisa.tensor_copy(dst=mine[:, _KV_LORA + 1:_KV_LORA + 2], src=l[0:rows_q, :])
        theirs = _sb((rows_q, _KV_LORA + 8), nl.float32)
        nisa.sendrecv(src=mine, dst=theirs, send_to_rank=1 - prg, recv_from_rank=1 - prg,
                      pipe_id=0)
        a, b = (mine, theirs) if prg == 0 else (theirs, mine)
        states = [_merge((a[:, _KV_LORA:_KV_LORA + 1], a[:, _KV_LORA + 1:_KV_LORA + 2],
                          a[:, 0:_KV_LORA]),
                         (b[:, _KV_LORA:_KV_LORA + 1], b[:, _KV_LORA + 1:_KV_LORA + 2],
                          b[:, 0:_KV_LORA]), scale, rows_q)]
    if split and prg != 0:
        return out, lat_out

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
    nisa.dma_copy(dst=lat_out[first:first + count, :], src=lat)
    return out, lat_out


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


def mla_paged_prefill_kernel(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, cache,
                             block_table, prefix_len, EPS: float):
    """One prefill chunk of one sequence: ``(out [T, H * 128] bf16, latent [T, 576] bf16)``.

    Args:
        qkv_a: ``[T, 2112]`` fp32 rows of the chunk's tokens: ``q_a`` then ``kv_a``.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, cache: as ``mla_paged_decode_kernel``.
        block_table: ``[pages]`` int32 block ids of the sequence's cached prefix.
        prefix_len: ``[1]`` int32 cached tokens before the chunk.
        EPS: the norms' epsilon.

    Returns:
        The gated attention output and the chunk's latent rows.
    """
    tokens = qkv_a.shape[0]
    heads = _check_common(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, cache)
    blocks, block_size = cache.shape[0], cache.shape[1]
    pages = block_table.shape[0]
    kernel_assert(tuple(prefix_len.shape) == (1,), "prefix_len is [1]")
    width = heads * _V
    window = pages * block_size
    n_tiles = window // _P
    n_blocks = (window + _BLOCK - 1) // _BLOCK
    n_q = (tokens + _P - 1) // _P
    programs = nl.num_programs(0)
    prg = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    scale = float((_NOPE + _ROPE) ** -0.5)
    out = nl.ndarray((tokens, width), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    lat_out = nl.ndarray((tokens, _LATENT), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_groups = [list(range(g, min(n_q, g + _Q_GROUP))) for g in range(0, n_q, _Q_GROUP)]
    mine = q_groups[prg::programs]
    if not mine:
        return out, lat_out
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
    lats = []
    for i in range(my_tiles[-1] + 1):
        rows = rows_of(i)
        lat = (_zeros if rows % 2 else _sb)((_even(rows), _LATENT), nl.bfloat16)
        _latent_rows(lat, load_rows(i), kv_gain[0:rows, :], EPS)
        if i in my_tiles:
            nisa.dma_copy(dst=lat_out[i * _P:i * _P + rows, :], src=lat[0:rows, :])
        lats.append(lat)

    positions = _sb((_P, n_tiles), nl.float32)
    nisa.iota(dst=positions, pattern=[[_P, n_tiles]], offset=0, channel_multiplier=1)
    zero = _sb((_P, 1), nl.float32)
    nisa.memset(dst=zero, value=0.0)
    held, bound, invalid = _cache_keys(block_table.ap(pattern=[[pages, 1], [1, pages]], offset=0),
                                       prefix_len.ap(pattern=[[0, _P], [1, 1]], offset=0),
                                       blocks, pages, window, positions)
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
            tiles = _clear_unowned(_cache_pages(cache, held, b * tiles_per_block, n_t), invalid,
                                   b * tiles_per_block)
            _attend(groups, "cache", tiles, b * _BLOCK, scale, zero)
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
    return out, lat_out


_DECODE = wrap_nki(mla_paged_decode_kernel)
_PREFILL = wrap_nki(mla_paged_prefill_kernel)


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


def _check_inputs(qkv_a, gate, kv_cache, w_uk, w_uv_t):
    heads = w_uk.shape[0]
    if qkv_a.dim() != 2 or qkv_a.shape[1] != _QKV or qkv_a.dtype != torch.float32:
        raise ValueError(f"qkv_a must be [T, {_QKV}] fp32, got {tuple(qkv_a.shape)} {qkv_a.dtype}")
    if tuple(gate.shape) != (qkv_a.shape[0], heads * _V) or gate.dtype != torch.bfloat16:
        raise ValueError(f"gate must be [T, {heads * _V}] bf16, "
                         f"got {tuple(gate.shape)} {gate.dtype}")
    if kv_cache.dim() != 3 or kv_cache.shape[2] != _LATENT or kv_cache.dtype != torch.bfloat16:
        raise ValueError(f"kv_cache must be [blocks, block_size, {_LATENT}] bf16, got "
                         f"{tuple(kv_cache.shape)} {kv_cache.dtype}")
    if kv_cache.shape[1] % _P:
        raise ValueError(f"block_size {kv_cache.shape[1]} must be a multiple of {_P}")
    if w_uv_t.dtype != torch.float32:
        raise ValueError(f"w_uv_t must be fp32, got {w_uv_t.dtype}")


def mla_paged_decode(qkv_a: torch.Tensor, gate: torch.Tensor, q_norm_weight: torch.Tensor,
                     kv_norm_weight: torch.Tensor, w_qb_t: torch.Tensor, w_uk: torch.Tensor,
                     w_uv_t: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor,
                     context_lens: torch.Tensor, eps: float):
    """Paged MLA decode step on the NKI kernel: ``(out [128, T, H, 1] bf16, latent [T, 576] bf16)``.

    Args:
        qkv_a: ``[T, 2112]`` fp32 all-reduced ``q_a`` and ``kv_a`` of each sequence's new token.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm_weight: ``[1536]`` ``q_a_layernorm`` gain.
        kv_norm_weight: ``[512]`` ``kv_a_layernorm`` gain.
        w_qb_t, w_uk, w_uv_t: the layouts from :func:`prepare_mla_weights`.
        kv_cache: ``[blocks, block_size, 576]`` bf16 latent cache, ``block_size`` a multiple
            of 128.
        block_table: ``[T, pages]`` int32 block ids per sequence.
        context_lens: ``[T]`` int32 tokens already cached per sequence.
        eps: the norms' epsilon.

    Returns:
        The output for ``o_proj`` and the latent rows to write at each sequence's position.

    Raises:
        ValueError: an input outside the kernel's contract.
    """
    _check_inputs(qkv_a, gate, kv_cache, w_uk, w_uv_t)
    if block_table.dim() != 2 or block_table.shape[0] != qkv_a.shape[0]:
        raise ValueError(f"block_table must be [T, pages], got {tuple(block_table.shape)}")
    return _DECODE[get_logical_nc_config()](
        qkv_a=qkv_a.contiguous(), gate=gate.contiguous(), q_norm=q_norm_weight.contiguous(),
        kv_norm=kv_norm_weight.contiguous(), w_qb_t=w_qb_t, w_uk=w_uk, w_uv_t=w_uv_t,
        cache=kv_cache, block_table=block_table.to(torch.int32).contiguous(),
        context_lens=context_lens.to(torch.int32).contiguous(), EPS=float(eps))


def mla_paged_prefill(qkv_a: torch.Tensor, gate: torch.Tensor, q_norm_weight: torch.Tensor,
                      kv_norm_weight: torch.Tensor, w_qb_t: torch.Tensor, w_uk: torch.Tensor,
                      w_uv_t: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor,
                      prefix_len: torch.Tensor, eps: float):
    """Paged MLA prefill chunk of one sequence on the NKI kernel, in 128-row query tiles.

    Args:
        qkv_a: ``[T, 2112]`` fp32 all-reduced ``q_a`` and ``kv_a`` of the chunk's tokens.
        gate: ``[T, H * 128]`` bf16 output-gate projection, before its sigmoid.
        q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, kv_cache: as
            :func:`mla_paged_decode`.
        block_table: ``[pages]`` int32 block ids of the cached prefix.
        prefix_len: ``[1]`` int32 tokens cached before the chunk.
        eps: the norms' epsilon.

    Returns:
        ``(out [T, H * 128] bf16, latent [T, 576] bf16)``.

    Raises:
        ValueError: an input outside the kernel's contract.
    """
    _check_inputs(qkv_a, gate, kv_cache, w_uk, w_uv_t)
    if block_table.dim() != 1 or prefix_len.numel() != 1:
        raise ValueError(f"block_table must be [pages] and prefix_len [1], got "
                         f"{tuple(block_table.shape)} and {tuple(prefix_len.shape)}")
    return _PREFILL[get_logical_nc_config()](
        qkv_a=qkv_a.contiguous(), gate=gate.contiguous(), q_norm=q_norm_weight.contiguous(),
        kv_norm=kv_norm_weight.contiguous(), w_qb_t=w_qb_t, w_uk=w_uk, w_uv_t=w_uv_t,
        cache=kv_cache, block_table=block_table.to(torch.int32).contiguous(),
        prefix_len=prefix_len.to(torch.int32).reshape(1).contiguous(), EPS=float(eps))
