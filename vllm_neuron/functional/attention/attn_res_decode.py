# SPDX-License-Identifier: Apache-2.0
"""Attention-residual (AttnRes) mix at decode, with the producer's tail and a fused RMSNorm.

A call mixes each token's bank rows and its running stream with softmax weights taken
from each source's RMS-normed dot with the score weight. Before the mix it can fold
the producer's tail into the stream: a bf16 residual add, or the fp32 MoE tail pack
``[up | shared | sum of squares]``, whose latent norm scale it applies. It can store
the new stream as a bank row, and emit this rank's slice of the RMS-normed mix in fp32
for a K-sharded projection, at its replica's rows of a buffer whose other rows are zero,
in the output projection's ``[D, rows, N, 1]`` input layout.

The hidden dim lies on the partitions (``h = p * H / 128 + j``). One ones-matmul sums
the per-partition dots and squares and broadcasts them to every partition, so the
softmax runs on every partition and the mix needs no second broadcast.
"""

from typing import NamedTuple

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config


class AttnResDecodeOutput(NamedTuple):
    """Outputs of ``attn_res_decode``; the optional ones are None when not asked for."""

    mix: torch.Tensor
    new_stream: torch.Tensor
    k_slice: torch.Tensor | None
    bank: torch.Tensor | None


def attn_res_score_weight(norm_weight: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """Return the fp32 score weight ``norm.weight * proj.weight`` that the kernel takes."""
    return (norm_weight.float() * proj_weight.reshape(-1).float()).contiguous()


def attn_res_decode(
    bank: torch.Tensor,
    stream: torch.Tensor,
    score_weight: torch.Tensor,
    num_sources: int,
    *,
    residual: torch.Tensor | None = None,
    moe_tail: torch.Tensor | None = None,
    latent_size: int | None = None,
    bank_slot: int | None = None,
    norm_weight: torch.Tensor | None = None,
    tp_rank: torch.Tensor | None = None,
    tp_size: int = 1,
    dp_rank: torch.Tensor | None = None,
    dp_size: int = 1,
    eps: float = 1e-5,
) -> AttnResDecodeOutput:
    """Mix ``num_sources`` sources per token: the first ``num_sources - 1`` bank rows and the stream.

    Args:
        bank: ``[T, NB, H]`` bf16 bank rows; row ``bank_slot`` is written in place.
        stream: ``[T, H]`` bf16 running stream.
        score_weight: ``[H]`` fp32, from ``attn_res_score_weight``.
        num_sources: sources per token, bank rows plus the stream (2 to NB + 1).
        residual: ``[T, H]`` bf16 added to the stream before the mix.
        moe_tail: ``[T, 2H + 1]`` fp32 MoE tail pack; the stream gains
            ``bf16(up * rsqrt(ss / latent_size + eps) + shared)``.
        latent_size: the latent width the pack's sum of squares covers.
        bank_slot: bank row that receives the new stream (at or after the sources).
        norm_weight: ``[H]`` bf16 RMSNorm weight; when given, emit the K slice.
        tp_rank: ``[1]`` int32 rank in the TP group, picks the K slice's columns.
        tp_size: TP group size; the K slice is ``H / tp_size`` wide.
        dp_rank: ``[1]`` int32 replica index, picks the K slice's rows.
        dp_size: replicas; the K slice has ``dp_size * T`` rows.
        eps: epsilon of every RMS norm here.

    Returns:
        The bf16 mix ``[T, H]``; the new stream (``stream`` itself with no tail);
        the normed fp32 K slice ``[D, dp_size * T, N, 1]``, whose element
        ``(d, row, n)`` is column ``n * D + d`` of the row's ``H / tp_size``
        columns, with ``N = ceil(H / tp_size / 128)`` and the other replicas'
        rows zero; and ``bank`` when ``bank_slot`` is set.
    """
    tokens, slots, hidden = bank.shape
    assert bank.dtype == stream.dtype == torch.bfloat16, "bank and stream must be bf16"
    assert tuple(stream.shape) == (tokens, hidden), (
        f"stream {tuple(stream.shape)} must be [{tokens}, {hidden}]")
    assert score_weight.dtype == torch.float32 and score_weight.numel() == hidden, (
        f"score_weight must be fp32 with {hidden} elements")
    assert 2 <= num_sources <= slots + 1, f"num_sources {num_sources} must be in [2, {slots + 1}]"
    assert residual is None or moe_tail is None, "fold one producer tail, not two"
    if residual is not None:
        assert residual.dtype == torch.bfloat16 and tuple(residual.shape) == (tokens, hidden), (
            f"residual must be bf16 [{tokens}, {hidden}]")
    if moe_tail is not None:
        assert moe_tail.dtype == torch.float32 and tuple(moe_tail.shape) == (
            tokens, 2 * hidden + 1), f"moe_tail must be fp32 [{tokens}, {2 * hidden + 1}]"
        assert latent_size, "latent_size is required with moe_tail"
    if bank_slot is not None:
        assert residual is not None or moe_tail is not None, "a bank row stores a new stream"
        assert num_sources - 1 <= bank_slot < slots, (
            f"bank_slot {bank_slot} must be in [{num_sources - 1}, {slots})")
    if norm_weight is not None:
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.numel() == hidden, (
            f"norm_weight must be bf16 with {hidden} elements")
        assert hidden % tp_size == 0, f"hidden {hidden} must split over {tp_size} ranks"
        assert tp_size == 1 or tp_rank is not None, "tp_rank is required with tp_size > 1"
        assert dp_size == 1 or dp_rank is not None, "dp_rank is required with dp_size > 1"

    def rank(r):
        return None if r is None else r.reshape(1, 1).to(torch.int32)

    outs = wrap_nki(_attn_res_decode_kernel)[get_logical_nc_config()](
        bank=bank,
        stream=stream,
        score_weight=score_weight.reshape(1, hidden),
        residual=residual,
        moe_tail=moe_tail,
        norm_weight=None if norm_weight is None else norm_weight.reshape(1, hidden),
        tp_rank=rank(tp_rank) if tp_size > 1 else None,
        dp_rank=rank(dp_rank) if dp_size > 1 else None,
        num_sources=num_sources,
        bank_slot=bank_slot,
        tp_size=tp_size,
        dp_size=dp_size,
        latent_size=latent_size or 1,
        eps=eps,
    )
    outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
    mix = outs.pop(0)
    new_stream = outs.pop(0) if residual is not None or moe_tail is not None else stream
    k_slice = outs.pop(0) if norm_weight is not None else None
    new_bank = outs.pop(0) if bank_slot is not None else None
    return AttnResDecodeOutput(mix, new_stream, k_slice, new_bank)


#: Static transfers take runtime-built descriptors; the two dynamic ones (the
#: rank's columns of the mix) take the Sync engine's descriptor generator.
_STATIC = dict(dge_mode=nisa.dge_mode.none)
_DYNAMIC = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _rows(t0, tn, row_stride, cols, offset=0):
    """HBM pattern of ``tn`` rows of ``cols`` elements spread over the 128 partitions."""
    return [[cols, nl.tile_size.pmax], [row_stride, tn], [1, cols]], offset + t0 * row_stride


def _load(dst, src, pattern, offset):
    nisa.dma_copy(dst=dst, src=src.ap(pattern=pattern, offset=offset), **_STATIC)


def _store(dst, src, pattern, offset):
    nisa.dma_copy(dst=dst.ap(pattern=pattern, offset=offset), src=src, **_STATIC)


@nki.jit
def _attn_res_decode_kernel(bank, stream, score_weight, residual, moe_tail, norm_weight,
                            tp_rank, dp_rank, num_sources, bank_slot, tp_size, dp_size,
                            latent_size, eps):
    """Each core mixes its share of the tokens end to end."""
    T, NB, H = bank.shape
    P = nl.tile_size.pmax
    J = H // P
    C = num_sources
    nb = C - 1
    has_tail = moe_tail is not None
    has_prologue = residual is not None or has_tail
    has_norm = norm_weight is not None
    kernel_assert(H % P == 0, f"Expected hidden divisible by {P}, got {H}")
    kernel_assert(T <= P, f"Expected at most {P} tokens, got {T}")

    n_prog = nl.num_programs(axes=0)
    prog = nl.program_id(axis=0)
    per_core = div_ceil(T, n_prog)
    t0 = prog * per_core
    tn = min(per_core, T - t0)

    mix = nl.ndarray((T, H), dtype=bank.dtype, buffer=nl.shared_hbm)
    outs = [mix]
    if has_prologue:
        new_stream = nl.ndarray((T, H), dtype=bank.dtype, buffer=nl.shared_hbm)
        outs.append(new_stream)
    if has_norm:
        KW = H // tp_size
        KN = div_ceil(KW, P)
        KD = KW // KN
        kernel_assert(KW % KN == 0, f"Expected the K slice width {KW} to split into {KN} equal blocks")
        k_slice = nl.ndarray((KD, dp_size * T, KN, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        outs.append(k_slice)
    if bank_slot is not None:
        outs.append(bank)
    if tn <= 0:
        return tuple(outs)
    TC = tn * C

    # Loads first: the bank rows, the stream, the producer's tail and the norm weight.
    v = nl.ndarray((P, tn, C, J), dtype=bank.dtype, buffer=nl.sbuf)
    for t in range(tn):
        _load(v[0:P, t, 0:nb, 0:J], bank, [[J, P], [H, nb], [1, J]], (t0 + t) * NB * H)
    sw = nl.ndarray((P, J), dtype=nl.float32, buffer=nl.sbuf)
    _load(sw, score_weight, [[J, P], [1, J]], 0)
    pat, off = _rows(t0, tn, H, J)
    stream_row = v[0:P, 0:tn, nb, 0:J]
    if not has_prologue:
        _load(stream_row, stream, pat, off)
    else:
        st = nl.ndarray((P, tn, J), dtype=bank.dtype, buffer=nl.sbuf)
        _load(st, stream, pat, off)
        addend = nl.ndarray((P, tn, J), dtype=bank.dtype, buffer=nl.sbuf)
        if not has_tail:
            _load(addend, residual, pat, off)
        else:
            W = moe_tail.shape[1]
            pat_w, off_w = _rows(t0, tn, W, J)
            up = nl.ndarray((P, tn, J), dtype=nl.float32, buffer=nl.sbuf)
            shared = nl.ndarray((P, tn, J), dtype=nl.float32, buffer=nl.sbuf)
            tail_ss = nl.ndarray((P, tn), dtype=nl.float32, buffer=nl.sbuf)
            _load(up, moe_tail, pat_w, off_w)
            _load(shared, moe_tail, pat_w, off_w + H)
            _load(tail_ss, moe_tail, [[0, P], [W, tn]], t0 * W + 2 * H)

    # Constants, and a first activation that loads the Scalar table while the
    # loads are in flight; its zero result is the squares' bias.
    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0, engine=nisa.gpsimd_engine)
    zero_col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_col, value=0.0, engine=nisa.gpsimd_engine)
    no_bias = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=no_bias, op=nl.square, data=zero_col)

    # The norm's inputs, ahead of the mix so none of them waits on it: the weight
    # slice and this rank's column offset, and for several replicas one select
    # mask per replica, true only in this replica's block, with D on the partitions.
    if has_norm:
        w = nl.ndarray((tn, KW), dtype=norm_weight.dtype, buffer=nl.sbuf)
        col = None
        if tp_rank is None:
            _load(w, norm_weight, [[0, tn], [1, KW]], 0)
        else:
            col = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=col, src=tp_rank, **_STATIC)
            nisa.tensor_scalar(dst=col, data=col, op0=nl.multiply, operand0=KW,
                               engine=nisa.gpsimd_engine)
            nisa.dma_copy(dst=w, src=norm_weight.ap(pattern=[[0, tn], [1, KW]], offset=0,
                                                    scalar_offset=col, indirect_dim=1), **_DYNAMIC)
        if dp_rank is not None:
            rep_i = nl.ndarray((KD, 1), dtype=nl.int32, buffer=nl.sbuf)
            _load(rep_i, dp_rank, [[0, KD], [1, 1]], 0)
            rep = nl.ndarray((KD, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rep, src=rep_i, engine=nisa.gpsimd_engine)
            masks = []
            for r in range(dp_size):
                mask = nl.ndarray((KD, tn, KN), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mask, data=rep.ap(pattern=[[1, KD], [0, tn], [0, KN]], offset=0),
                                   op0=nl.equal, operand0=float(r), engine=nisa.gpsimd_engine)
                masks.append(mask)
            k_rows = nl.ndarray((KD, dp_size, tn, KN), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=k_rows, value=0.0, engine=nisa.gpsimd_engine)

    # The prologue: the new stream, rounded once to bf16 as the reference's add.
    if has_prologue:
        if has_tail:
            scale = nl.ndarray((P, tn), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scale, data=tail_ss, op0=nl.multiply, operand0=1.0 / latent_size,
                               op1=nl.add, operand1=eps, engine=nisa.gpsimd_engine)
            nisa.tensor_scalar(dst=scale, data=scale, op0=nl.rsqrt, operand0=0.0,
                               engine=nisa.gpsimd_engine)
            for t in range(tn):
                nisa.scalar_tensor_tensor(dst=addend[0:P, t, 0:J], data=up[0:P, t, 0:J],
                                          op0=nl.multiply, operand0=scale[0:P, t:t + 1],
                                          op1=nl.add, operand1=shared[0:P, t, 0:J])
        nisa.tensor_tensor(dst=stream_row, data1=st, data2=addend, op=nl.add)
        _store(new_stream, stream_row, pat, off)
        if bank_slot is not None:
            _store(bank, stream_row, [[J, P], [NB * H, tn], [1, J]], (t0 * NB + bank_slot) * H)

    # Per-partition dots (Vector multiplies) and squares (Scalar), reduced over j in
    # one pass: terms = [dot terms (t, c, j) | squares (t, c, j)].
    terms = nl.ndarray((P, 2, tn, C, J), dtype=nl.float32, buffer=nl.sbuf)
    for lo, n in ((0, nb), (nb, 1)):
        nisa.tensor_tensor(dst=terms[0:P, 0, 0:tn, lo:lo + n, 0:J],
                           data1=v[0:P, 0:tn, lo:lo + n, 0:J],
                           data2=sw.ap(pattern=[[J, P], [0, tn], [0, n], [1, J]], offset=0),
                           op=nl.multiply)
        nisa.activation(dst=terms[0:P, 1, 0:tn, lo:lo + n, 0:J], op=nl.square,
                        data=v[0:P, 0:tn, lo:lo + n, 0:J], bias=no_bias)
    part = nl.ndarray((P, 2 * TC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=part, data=terms.ap(pattern=[[2 * TC * J, P], [J, 2 * TC], [1, J]],
                                                offset=0), op=nl.add, axis=2)
    sums = nl.ndarray((P, 2 * TC), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=sums, stationary=ones, moving=part, is_stationary_onezero=True)

    # Softmax over each token's sources, on every partition. Every rsqrt runs on
    # GpSimd, the engine whose rsqrt is accurate to 1 ULP.
    rs = nl.ndarray((P, TC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rs, data=sums[0:P, TC:2 * TC], op0=nl.multiply, operand0=1.0 / H,
                       op1=nl.add, operand1=eps, engine=nisa.vector_engine)
    nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.gpsimd_engine)
    score = nl.ndarray((P, TC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=score, data1=rs, data2=sums[0:P, 0:TC], op=nl.multiply)
    by_token = [[TC, P], [C, tn], [1, C]]
    neg_top = nl.ndarray((P, tn), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_top, data=score.ap(pattern=by_token, offset=0), op=nl.maximum,
                       axis=2, negate=True)
    expo = nl.ndarray((P, TC), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((P, tn), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(tn):
        nisa.activation_reduce(dst=expo[0:P, t * C:(t + 1) * C], op=nl.exp,
                               data=score[0:P, t * C:(t + 1) * C], bias=neg_top[0:P, t:t + 1],
                               reduce_op=nl.add, reduce_res=den[0:P, t:t + 1])
    rden = nl.ndarray((P, tn), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    prob = nl.ndarray((P, TC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prob.ap(pattern=by_token, offset=0),
                       data1=expo.ap(pattern=by_token, offset=0),
                       data2=rden.ap(pattern=[[tn, P], [1, tn], [0, C]], offset=0), op=nl.multiply)

    # The mix: one multiply by the weights, then the sources summed by halving;
    # the last add writes the bf16 output.
    weighted = nl.ndarray((P, tn, C, J), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=weighted, data1=v,
                       data2=prob.ap(pattern=[[TC, P], [C, tn], [1, C], [0, J]], offset=0),
                       op=nl.multiply)
    out = nl.ndarray((P, tn, J), dtype=bank.dtype, buffer=nl.sbuf)
    n = C
    while n > 1:
        h = n // 2
        if n == 2:
            nisa.tensor_tensor(dst=out, data1=weighted[0:P, 0:tn, 0, 0:J],
                               data2=weighted[0:P, 0:tn, 1, 0:J], op=nl.add)
        else:
            nisa.tensor_tensor(dst=weighted[0:P, 0:tn, 0:h, 0:J], data1=weighted[0:P, 0:tn, 0:h, 0:J],
                               data2=weighted[0:P, 0:tn, n - h:n, 0:J], op=nl.add)
        n -= h
    _store(mix, out, pat, off)
    if not has_norm:
        return tuple(outs)

    # The K slice: the token's sum of squares lands on its own partition, and its
    # slice of the mix comes back from HBM at this rank's column offset.
    sq = nl.ndarray((P, tn, J), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=out, bias=no_bias)
    norm_ss = nl.ndarray((P, tn), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=norm_ss, data=sq, op=nl.add, axis=2)
    ones_col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0, engine=nisa.gpsimd_engine)
    tok_ss = nl.ndarray((tn, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tok_ss, stationary=norm_ss, moving=ones_col)
    rnorm = nl.ndarray((tn, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rnorm, data=tok_ss, op0=nl.multiply, operand0=1.0 / H,
                       op1=nl.add, operand1=eps, engine=nisa.vector_engine)
    nisa.tensor_scalar(dst=rnorm, data=rnorm, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.gpsimd_engine)
    kx = nl.ndarray((tn, KW), dtype=bank.dtype, buffer=nl.sbuf)
    if col is None:
        _load(kx, mix, [[H, tn], [1, KW]], t0 * H)
    else:
        nisa.dma_copy(dst=kx, src=mix.ap(pattern=[[H, tn], [1, KW]], offset=t0 * H,
                                         scalar_offset=col, indirect_dim=1), **_DYNAMIC)
    ks = nl.ndarray((tn, KW), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=ks, data=kx, op0=nl.multiply, operand0=rnorm,
                              op1=nl.multiply, operand1=w)
    # D onto the partitions: one Tensor Engine transpose per block of D columns,
    # bit-exact (non-finite values included), read back in (d, t, n) order.
    kt = nl.ndarray((KD, KN, tn), dtype=nl.float32, buffer=nl.psum)
    for n in range(KN):
        nisa.nc_transpose(dst=kt[0:KD, n, 0:tn], data=ks[0:tn, n * KD:(n + 1) * KD],
                          engine=nisa.engine.tensor)
    by_row = kt.ap(pattern=[[KN * tn, KD], [1, tn], [tn, KN]], offset=0)
    if dp_rank is None:
        kst = nl.ndarray((KD, tn, KN), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kst, src=by_row)
        _store(k_slice, kst, [[T * KN, KD], [1, tn * KN]], t0 * KN)
        return tuple(outs)
    # Each replica block is this rank's slice where its mask is true and the zero
    # fill elsewhere: a select, so a non-finite row never reaches another block.
    for r in range(dp_size):
        nisa.tensor_copy_predicated(dst=k_rows[0:KD, r, 0:tn, 0:KN], src=by_row,
                                    predicate=masks[r])
    _store(k_slice, k_rows, [[dp_size * T * KN, KD], [T * KN, dp_size], [1, tn * KN]], t0 * KN)
    return tuple(outs)
