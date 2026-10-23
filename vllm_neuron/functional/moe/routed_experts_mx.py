# SPDX-License-Identifier: Apache-2.0
"""Routed experts on MXFP4 weights: sigmoid top-k selection and a distinct-expert loop.

At decode one kernel selects each token's top-k experts from fp32 router logits,
keeps the affinities of this rank's local experts, and visits each local expert
that at least one token chose exactly once, in ascending local order. Every visit
streams all tokens: a token that did not choose the expert has affinity 0 there.
The tokens are MXFP8 stationaries ``[128, D/512, T]``; the weights move in
512-column chunks. Gate and up products feed SiTU in fp32, the bf16 result is
transposed and quantized to MXFP8 for the down product, and the fp32 down
product is weighted and summed per token. Under LNC 2 each core takes half of
the expert width and the halves meet in one send-receive. The partial is stored
in the combine's chunk order ``[R, C, T/R, D/C]``.

A token whose logits or latent row hold a non-finite value chooses no expert,
streams as zeros, and gets a NaN output row; all three use selects.

Weight bank, per local expert ``e`` and program ``g`` of ``G`` (the loader's
repack): ``gate_up_weight [E, G, 2NI, 128, NK*512]`` uint16, four fp4 values per
element, partition ``p`` of a K tile holding K values ``4p`` to ``4p + 3``;
chunk ``n < NI`` is gate rows ``g*I/G + n*512 + c``, chunk ``n >= NI`` the up
rows; ``gate_up_scale [E, G, 16, 2NI*NK*512]`` uint8, scale row ``r`` covering K
values ``32r`` to ``32r + 31`` at free index ``(n*NK + k)*512 + c``;
``down_weight [E, G, D/512, 128, NI*512]`` and ``down_scale
[E, G, 16, (D/512)*NI*512]`` likewise over the program's expert-width columns.
``NK = D/512``, ``NI = I/(512G)``.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config

TOP_K = 16
#: Added to the sum of the chosen scores before the renormalizing division.
RENORM_GUARD = 1e-20
_KT = 512  # contraction values per MX K tile
_Q = 4  # values per x4 element
_MIN_FLOAT = -3.4028235e38
_NAN = float("nan")


def router_select(
    logits: torch.Tensor,
    score_bias: torch.Tensor,
    scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each token's top-k expert indices and renormalized sigmoid weights.

    Args:
        logits: ``[T, E]`` fp32 router logits, ``T <= 128``.
        score_bias: ``[E]`` fp32 score correction bias, added for the choice only.
        scaling: routed scaling factor applied after renormalization.

    Returns:
        ``[T, 16]`` int32 indices in descending choice order and ``[T, 16]`` fp32
        weights. A token with a non-finite logit gets in-range indices and NaN weights.
    """
    tokens, experts = logits.shape
    assert logits.dtype == score_bias.dtype == torch.float32, "logits and score_bias must be fp32"
    assert 1 <= tokens <= 128, f"router_select takes at most 128 tokens, got {tokens}"
    assert tuple(score_bias.shape) == (experts,), f"score_bias must be [{experts}]"
    return wrap_nki(_router_select_kernel)[get_logical_nc_config()](
        logits=logits, score_bias=score_bias, scaling=scaling)


def routed_experts_mx(
    routed: torch.Tensor,
    score_bias: torch.Tensor,
    expert_base: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    beta: float,
    linear_beta: float,
    scaling: float = 1.0,
    replicas: int = 2,
    chunks: int = 32,
) -> torch.Tensor:
    """Return this rank's routed-expert partial of ``T`` decode tokens in combine order.

    Args:
        routed: ``[T, D + E]`` fp32: the latent rows, then the router logits.
        score_bias: ``[E]`` fp32 score correction bias.
        expert_base: ``[1, 1]`` int32 global id of this rank's first local expert.
        gate_up_weight, gate_up_scale, down_weight, down_scale: the local bank.
        beta: SiTU gate beta.
        linear_beta: SiTU up beta.
        scaling: routed scaling factor.
        replicas: token groups ``R`` in ``routed``, ``T/R`` rows each.
        chunks: column chunks ``C`` per replica in the combine.

    Returns:
        ``[R, C, T/R, D/C]`` fp32: entry ``[r, c, t, j]`` is token ``r*T/R + t``,
        latent column ``c*D/C + j``.
    """
    tokens = routed.shape[0]
    experts, programs, _, _, packed = gate_up_weight.shape
    latent = packed
    assert routed.dtype == score_bias.dtype == torch.float32, "routed and score_bias must be fp32"
    assert expert_base.dtype == torch.int32 and tuple(expert_base.shape) == (1, 1), (
        "expert_base must be [1, 1] int32"
    )
    assert gate_up_weight.dtype == down_weight.dtype == torch.uint16, "MXFP4 weights are uint16"
    assert gate_up_scale.dtype == down_scale.dtype == torch.uint8, "MX scales are uint8"
    assert routed.shape[1] == latent + score_bias.shape[0], (
        f"routed {tuple(routed.shape)} must be [T, {latent} + {score_bias.shape[0]}]"
    )
    assert 1 <= tokens <= 64 and tokens % replicas == 0, (
        f"T={tokens} must be in [1, 64] and divide into {replicas} replicas"
    )
    assert latent % (chunks * programs) == 0, f"D={latent} must split into {chunks} chunks per core"
    assert programs == get_logical_nc_config(), (
        f"the bank is repacked for {programs} programs, LNC is {get_logical_nc_config()}"
    )
    return wrap_nki(_routed_experts_kernel)[programs](
        routed=routed,
        score_bias=score_bias,
        expert_base=expert_base,
        gate_up_weight=gate_up_weight,
        gate_up_scale=gate_up_scale,
        down_weight=down_weight,
        down_scale=down_scale,
        beta=beta,
        linear_beta=linear_beta,
        scaling=scaling,
        replicas=replicas,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------- #
# Kernel pieces.
# ---------------------------------------------------------------------------- #


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols)."""
    return nl.ndarray((rows, cols), dtype=dtype, buffer=nl.sbuf)


def _row_broadcast(src, rows, cols):
    """A [1, cols] fp32 SBUF row copied to ``rows`` partitions by a K=1 ones product (exact)."""
    ones = _tile(1, rows)
    nisa.memset(dst=ones, value=1.0)
    spread = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=spread, stationary=ones, moving=src)
    out = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=spread)
    return out


def _nonfinite_count(rows, width, dst):
    """``dst [P, 1]`` = number of non-finite values in each row of ``rows [P, width]``."""
    flags = nl.ndarray((rows.shape[0], width), dtype=nl.float32, buffer=nl.sbuf)
    # x * 0 is 0 for finite x and NaN for Inf or NaN; NaN != 0 holds.
    nisa.tensor_scalar(dst=flags, data=rows, op0=nl.multiply, operand0=0.0,
                       op1=nl.not_equal, operand1=0.0)
    nisa.tensor_reduce(dst=dst, data=flags, op=nl.add, axis=(1,))


def _select(logits, bias, keep, scaling):
    """Top-k of ``logits [P, E]`` by sigmoid score plus ``bias``; rows with ``keep`` 0 choose zeros.

    Returns (indices ``[P, 16]`` uint32, weights ``[P, 16]`` fp32).
    """
    rows, experts = logits.shape
    scores = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scores, op=nl.sigmoid, data=logits)
    biased = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=biased, data1=scores, data2=bias, op=nl.add)
    choice = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=choice, value=0.0)
    nisa.tensor_copy_predicated(dst=choice, src=biased, predicate=keep.ap(
        pattern=[[keep.shape[1], rows], [0, experts]]))
    top = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    index = nl.ndarray((rows, TOP_K), dtype=nl.uint32, buffer=nl.sbuf)
    first = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top[:, 0:8], src=choice)
    nisa.nc_match_replace8(dst=first, data=choice, vals=top[:, 0:8], imm=_MIN_FLOAT,
                           dst_idx=index[:, 0:8])
    second = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top[:, 8:16], src=first)
    nisa.nc_match_replace8(dst=second, data=first, vals=top[:, 8:16], imm=_MIN_FLOAT,
                           dst_idx=index[:, 8:16])
    chosen = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=chosen, data=scores, indices=index)
    total = _tile(rows, 1)
    nisa.tensor_reduce(dst=total, data=chosen, op=nl.add, axis=(1,))
    nisa.tensor_scalar(dst=total, data=total, op0=nl.add, operand0=RENORM_GUARD)
    inverse = _tile(rows, 1)
    nisa.reciprocal(dst=inverse, data=total)
    weights = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    if scaling == 1.0:
        nisa.tensor_scalar(dst=weights, data=chosen, op0=nl.multiply, operand0=inverse)
    else:
        nisa.tensor_scalar(dst=weights, data=chosen, op0=nl.multiply, operand0=inverse,
                           op1=nl.multiply, operand1=scaling)
    return index, weights


def _keep_rows(bad, rows):
    """uint8 ``[rows, 1]``: 1 where ``bad`` (a non-finite count) is 0."""
    keep = _tile(rows, 1, nl.uint8)
    nisa.tensor_scalar(dst=keep, data=bad, op0=nl.equal, operand0=0.0)
    return keep


@nki.jit
def _router_select_kernel(logits, score_bias, scaling):
    """Top-k indices and weights of ``[T, E]`` logits; a non-finite row gets NaN weights.

    Under LNC 2 each program selects for its half of the rows.
    """
    tokens, experts = logits.shape
    programs = nl.num_programs(0) if nl.program_ndim() != 0 else 1
    program = nl.program_id(0) if programs > 1 else 0
    index_out = nl.ndarray((tokens, TOP_K), dtype=nl.int32, buffer=nl.shared_hbm)
    weight_out = nl.ndarray((tokens, TOP_K), dtype=nl.float32, buffer=nl.shared_hbm)
    share = (tokens + programs - 1) // programs
    first = program * share
    count = min(share, tokens - first)
    if count > 0:
        rows = nl.ndarray((count, experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=logits[first:first + count, :])
        bias = nl.ndarray((count, experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias, src=score_bias.ap(pattern=[[0, count], [1, experts]], offset=0))
        bad = _tile(count, 1)
        _nonfinite_count(rows, experts, bad)
        keep = _keep_rows(bad, count)
        index, weights = _select(rows, bias, keep, scaling)
        marked = nl.ndarray((count, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=marked, value=_NAN)
        nisa.tensor_copy_predicated(dst=marked, src=weights, predicate=keep.ap(
            pattern=[[keep.shape[1], count], [0, TOP_K]]))
        index_i32 = nl.ndarray((count, TOP_K), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=index_i32, src=index)
        nisa.dma_copy(dst=index_out[first:first + count, :], src=index_i32)
        nisa.dma_copy(dst=weight_out[first:first + count, :], src=marked)
    return index_out, weight_out


def _local_affinity(index, weights, keep, base, local):
    """Dense ``[P, local]`` affinities and 0/1 choices of the experts ``base .. base + local - 1``."""
    rows = index.shape[0]
    index_f = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=index_f, src=index)
    offset = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=offset, data=index_f, op0=nl.subtract, operand0=base)
    slot = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=slot, pattern=[[1, local]], offset=0)
    hit = nl.ndarray((rows, local, TOP_K), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=hit, data1=offset.ap(pattern=[[TOP_K, rows], [0, local], [1, TOP_K]]),
                       data2=slot.ap(pattern=[[local, rows], [1, local], [0, TOP_K]]), op=nl.equal)
    spread = nl.ndarray((rows, local, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=spread, value=0.0)
    nisa.tensor_copy_predicated(dst=spread, src=weights.ap(
        pattern=[[TOP_K, rows], [0, local], [1, TOP_K]]), predicate=hit)
    picked = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=picked, data=spread, op=nl.add, axis=(2,))
    hits = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=hits, data=hit, op=nl.add, axis=(2,))
    affinity = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=affinity, value=0.0)
    chosen = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=chosen, value=0.0)
    for dst, src in ((affinity, picked), (chosen, hits)):
        nisa.tensor_copy_predicated(dst=dst, src=src, predicate=keep.ap(
            pattern=[[keep.shape[1], rows], [0, local]]))
    return affinity, chosen


def _latent_mx(latent_bf16, rows, latent):
    """MXFP8 stationaries ``[128, D/512, P]`` and scales of bf16 rows ``[P, D]``.

    Partition ``p`` of K tile ``k`` holds latent values ``k*512 + 4p .. + 3``; each
    group of 8 partitions by 4 values is one token's 32 consecutive values.
    """
    tiles = latent // _KT
    flipped = nl.ndarray((128, tiles, _Q, rows), dtype=nl.bfloat16, buffer=nl.psum)
    for k in range(tiles):
        for q in range(_Q):
            nisa.nc_transpose(dst=flipped[:, k, q, :], data=latent_bf16.ap(
                pattern=[[latent, rows], [_Q, 128]], offset=k * _KT + q))
    natural = nl.ndarray((128, tiles, rows, _Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=natural, src=flipped.ap(
        pattern=[[tiles * _Q * rows, 128], [_Q * rows, tiles], [1, rows], [rows, _Q]]))
    quant = nl.ndarray((128, tiles, rows), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scale = nl.ndarray((128, tiles, rows), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=quant, src=natural, dst_scale=scale)
    return quant, scale


def _operand_tiles(nk, ni, nd):
    """SBUF tiles for one expert's operands on one program, allocated once."""
    gate_up = nl.ndarray((128, 2 * ni, nk * _KT), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    down = nl.ndarray((128, nd, ni * _KT), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    scales = []
    for blocks in (2 * ni * nk, nd * ni):
        scale = nl.ndarray((128, blocks * _KT), dtype=nl.uint8, buffer=nl.sbuf)
        # The products read scales from partitions 32q to 32q + 3 only, which every load
        # writes; a small memset defines the tile ahead of the loop.
        nisa.memset(dst=scale[0:1, 0:4], value=0, engine=nisa.engine.gpsimd)
        scales.append(scale)
    return gate_up, scales[0], down, scales[1]


def _load_operands(bank, staged, program, tiles, dge):
    """DMA the expert in the [1, 1] uint32 tile ``staged`` of ``bank`` for ``program``: ten DMAs.

    All go on the queue of ``dge`` in program order, each set of scales ahead of its weights.
    """
    gate_up_weight, gate_up_scale, down_weight, down_scale = bank
    gate_up, gate_up_s, down, down_s = tiles
    _, programs, two_ni, _, gu_cols = gate_up_weight.shape
    _, _, nd, _, dn_cols = down_weight.shape
    with nl.no_reorder():
        expert = nisa.register_alloc()
        nisa.register_load(dst=expert, src=staged)
        for scale_src, scale_dst, weight_src, weight_dst, count, cols in (
                (gate_up_scale, gate_up_s, gate_up_weight, gate_up, two_ni, gu_cols),
                (down_scale, down_s, down_weight, down, nd, dn_cols)):
            scale_cols = scale_src.shape[3]
            for quad in range(4):
                nisa.dma_copy(dst=scale_dst[32 * quad:32 * quad + 4, :], src=scale_src.ap(
                    pattern=[[scale_cols, 4], [1, scale_cols]],
                    offset=(program * 16 + 4 * quad) * scale_cols, scalar_offset=expert,
                    indirect_dim=0), **dge)
            nisa.dma_copy(dst=weight_dst, src=weight_src.ap(
                pattern=[[cols, 128], [128 * cols, count], [1, cols]],
                offset=program * count * 128 * cols, scalar_offset=expert, indirect_dim=0),
                **dge)


def _expert_rows(stationary, stationary_scale, tiles, affinity, acc, beta, linear_beta):
    """``acc [P, D] += affinity [P, 1] * expert(rows)`` for one loaded expert's program half."""
    gate_up, gate_up_s, down, down_s = tiles
    nk = stationary.shape[1]
    rows = stationary.shape[2]
    ni = gate_up.shape[1] // 2
    nd = down.shape[1]
    act = nl.ndarray((rows, ni, _KT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(ni):
        products = []
        for n in (j, j + ni):
            product = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.psum)
            for k in range(nk):
                nisa.nc_matmul_mx(dst=product, stationary=stationary[:, k, :],
                                  moving=gate_up[:, n, k * _KT:(k + 1) * _KT],
                                  stationary_scale=stationary_scale[:, k, :],
                                  moving_scale=gate_up_s[:, (n * nk + k) * _KT:(n * nk + k + 1) * _KT],
                                  accumulate=(k > 0))
            products.append(product)
        gate, up = products
        # SiTU in fp32 in the reference's order: (beta * tanh(g / beta)) * sigmoid(g)
        # times linear_beta * tanh(u / linear_beta), rounded once to bf16.
        tanh_gate = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=tanh_gate, op=nl.tanh, data=gate, scale=1.0 / beta)
        sig_gate = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sig_gate, op=nl.sigmoid, data=gate)
        tanh_up = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=tanh_up, op=nl.tanh, data=up, scale=1.0 / linear_beta)
        situ = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=situ, data=tanh_gate, op0=nl.multiply, operand0=beta,
                                  op1=nl.multiply, operand1=sig_gate)
        nisa.scalar_tensor_tensor(dst=act[:, j, :], data=tanh_up, op0=nl.multiply,
                                  operand0=linear_beta, op1=nl.multiply, operand1=situ)
    flipped = nl.ndarray((128, ni, _Q, rows), dtype=nl.bfloat16, buffer=nl.psum)
    for j in range(ni):
        for q in range(_Q):
            nisa.nc_transpose(dst=flipped[:, j, q, :], data=act.ap(
                pattern=[[ni * _KT, rows], [_Q, 128]], offset=j * _KT + q))
    natural = nl.ndarray((128, ni, rows, _Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=natural, src=flipped.ap(
        pattern=[[ni * _Q * rows, 128], [_Q * rows, ni], [1, rows], [rows, _Q]]))
    act_q = nl.ndarray((128, ni, rows), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    act_s = nl.ndarray((128, ni, rows), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=act_q, src=natural, dst_scale=act_s)
    for d in range(nd):
        out = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.psum)
        for k in range(ni):
            nisa.nc_matmul_mx(dst=out, stationary=act_q[:, k, :], moving=down[:, d, k * _KT:(k + 1) * _KT],
                              stationary_scale=act_s[:, k, :],
                              moving_scale=down_s[:, (d * ni + k) * _KT:(d * ni + k + 1) * _KT],
                              accumulate=(k > 0))
        span = acc[:, d * _KT:(d + 1) * _KT]
        nisa.scalar_tensor_tensor(dst=span, data=out, op0=nl.multiply, operand0=affinity,
                                  op1=nl.add, operand1=span)


def _stage(table, p, dst, dge):
    """DMA column ``p`` (a register) of the SBUF ``table`` into ``dst`` on the queue of ``dge``.

    A DMA stages the value, so what it feeds waits on no compute engine.
    """
    rows, width = table.shape
    nisa.dma_copy(dst=dst, src=table.ap(pattern=[[width, rows], [1, 1]], scalar_offset=p,
                                        indirect_dim=1), **dge)
    return dst


def _visit_experts(chosen, affinity, stationary, stationary_scale, bank, program, acc,
                   beta, linear_beta):
    """Run each local expert with a chosen token once, ascending, next expert's loads in flight.

    Items run in pairs (2p, 2p + 1) on two operand buffers swapped statically: item
    2p + 1 loads while 2p computes, and 2p + 2 while 2p + 1 computes. With an even count
    the last pair loads nothing after it; with an odd count the last item runs alone. Each
    of the ``count`` items loads once; with no item, expert 0 loads and never computes.
    """
    rows, local = chosen.shape
    nk = stationary.shape[1]
    two_ni = bank[0].shape[2]
    nd = bank[2].shape[2]
    pairs = (local + 1) // 2
    padded = 2 * pairs + 2

    # Experts with at least one chosen token, compacted, and their count.
    ones = _tile(rows, 1)
    nisa.memset(dst=ones, value=1.0)
    counts = nl.ndarray((1, local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=counts, stationary=ones, moving=chosen)
    load = _tile(1, local)
    nisa.tensor_copy(dst=load, src=counts)
    visit = _tile(1, local + 1, nl.int32)
    nisa.nonzero_with_count(dst=visit, src=load, index_offset=0, padding_val=0)
    # Loop counts, read from uint32 tiles: pairs that load the item after them, a last pair
    # that loads nothing after it (an even count), and a last odd item.
    count = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=count, src=visit[:, local:local + 1])
    pairs_n = _tile(1, 1, nl.uint32)
    nisa.tensor_scalar(dst=pairs_n, data=count, op0=nl.right_shift, operand0=1)
    loops = _tile(1, 3, nl.uint32)
    nisa.tensor_scalar(dst=loops[:, 2:3], data=count, op0=nl.bitwise_and, operand0=1)
    even_tail = _tile(1, 2, nl.uint32)
    nisa.tensor_scalar(dst=even_tail[:, 0:1], data=pairs_n, op0=nl.minimum, operand0=1)
    nisa.tensor_scalar(dst=even_tail[:, 1:2], data=loops[:, 2:3], op0=nl.bitwise_xor, operand0=1)
    nisa.tensor_tensor(dst=loops[:, 1:2], data1=even_tail[:, 0:1], data2=even_tail[:, 1:2],
                       op=nl.bitwise_and)
    nisa.tensor_tensor(dst=loops[:, 0:1], data1=pairs_n, data2=loops[:, 1:2], op=nl.subtract)
    leading, last, odd = (nisa.register_alloc() for _ in range(3))
    for i, reg in enumerate((leading, last, odd)):
        nisa.register_load(dst=reg, src=loops[:, i:i + 1])
    flat = _tile(1, padded, nl.int32)
    nisa.memset(dst=flat, value=0)
    nisa.tensor_copy(dst=flat[:, 0:local], src=visit[:, 0:local])

    # Item i's affinity column, aff[:, flat[i]], by a select and an exact sum.
    flat_f = _tile(1, padded)
    nisa.tensor_copy(dst=flat_f, src=flat)
    item_expert = _row_broadcast(flat_f, rows, padded)
    slot = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=slot, pattern=[[1, local]], offset=0)
    match = nl.ndarray((rows, padded, local), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=match, data1=item_expert.ap(pattern=[[padded, rows], [1, padded], [0, local]]),
                       data2=slot.ap(pattern=[[local, rows], [0, padded], [1, local]]), op=nl.equal)
    spread = nl.ndarray((rows, padded, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=spread, value=0.0)
    nisa.tensor_copy_predicated(dst=spread, src=affinity.ap(
        pattern=[[local, rows], [0, padded], [1, local]]), predicate=match)
    item_aff = nl.ndarray((rows, padded), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=item_aff, data=spread, op=nl.add, axis=(2,))

    # Per pair position q: item 2p + q's expert and affinity column at loop register p.
    expert_at, aff_at = [], []
    for q in range(3):
        e_q = _tile(1, pairs, nl.int32)
        nisa.tensor_copy(dst=e_q, src=flat.ap(pattern=[[padded, 1], [2, pairs]], offset=q))
        a_q = nl.ndarray((rows, pairs), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=a_q, src=item_aff.ap(pattern=[[padded, rows], [2, pairs]], offset=q))
        expert_at.append(e_q)
        aff_at.append(a_q)

    buffers = [_operand_tiles(nk, two_ni // 2, nd) for _ in range(2)]
    first = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=first, src=flat[:, 0:1])
    loads = dict(dge_mode=nisa.dge_mode.hwdge)
    # Table entries go on the GpSimd queue, so they wait behind no expert load.
    entries = dict(dge_mode=nisa.dge_mode.swdge)
    _load_operands(bank, first, program, buffers[0], loads)
    held = [_tile(1, 1, nl.uint32), _tile(1, 1, nl.uint32)]

    def compute(q, p, tiles):
        # A full-height tile starts at partition 0, as the accumulator does.
        column = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)[0:rows, :]
        _stage(aff_at[q], p, column, entries)
        # In program order, so no engine starts the next expert ahead of this one's products.
        with nl.no_reorder():
            _expert_rows(stationary, stationary_scale, tiles, column, acc, beta, linear_beta)

    def run_pair(p):
        _load_operands(bank, _stage(expert_at[1], p, held[1], entries), program, buffers[1], loads)
        compute(0, p, buffers[0])
        _load_operands(bank, _stage(expert_at[2], p, held[0], entries), program, buffers[0], loads)
        compute(1, p, buffers[1])

    def last_pair(_):
        _load_operands(bank, _stage(expert_at[1], leading, held[1], entries), program, buffers[1],
                       loads)
        compute(0, leading, buffers[0])
        compute(1, leading, buffers[1])

    nl.fori_loop(0, leading, run_pair)
    nl.fori_loop(0, last, last_pair)
    nl.fori_loop(0, odd, lambda _: compute(0, leading, buffers[0]))


@nki.jit
def _routed_experts_kernel(routed, score_bias, expert_base, gate_up_weight, gate_up_scale,
                           down_weight, down_scale, beta, linear_beta, scaling, replicas, chunks):
    """Selection, the distinct-expert loop, the LNC 2 merge and the combine-order store."""
    tokens, width = routed.shape
    local, programs, two_ni, _, gu_cols = gate_up_weight.shape
    latent = gu_cols
    experts = width - latent
    per_replica = tokens // replicas
    kernel_assert(programs == (nl.num_programs(0) if nl.program_ndim() != 0 else 1),
                  "one program per bank half")
    program = nl.program_id(0) if programs > 1 else 0
    rows = tokens + tokens % 2  # the stationary free size must be even
    bank = (gate_up_weight.view(nl.float4_e2m1fn_x4), gate_up_scale,
            down_weight.view(nl.float4_e2m1fn_x4), down_scale)

    # Rows past T read as NaN logits, so they choose nothing and stream as zeros.
    logits = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    if rows > tokens:
        nisa.memset(dst=logits, value=_NAN)
    nisa.dma_copy(dst=logits[0:tokens, :], src=routed[:, latent:width], dge_mode=nisa.dge_mode.hwdge)
    hidden = nl.ndarray((rows, latent), dtype=nl.float32, buffer=nl.sbuf)
    if rows > tokens:
        nisa.memset(dst=hidden, value=0.0)
    nisa.dma_copy(dst=hidden[0:tokens, :], src=routed[:, 0:latent], dge_mode=nisa.dge_mode.hwdge)
    bias = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias, src=score_bias.ap(pattern=[[0, rows], [1, experts]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    base_i = _tile(rows, 1, nl.int32)
    nisa.dma_copy(dst=base_i, src=expert_base.ap(pattern=[[0, rows], [1, 1]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    base = _tile(rows, 1)
    nisa.tensor_copy(dst=base, src=base_i)

    bad_logit = _tile(rows, 1)
    _nonfinite_count(logits, experts, bad_logit)
    bad_hidden = _tile(rows, 1)
    _nonfinite_count(hidden, latent, bad_hidden)
    bad = _tile(rows, 1)
    nisa.tensor_tensor(dst=bad, data1=bad_logit, data2=bad_hidden, op=nl.add)
    keep = _keep_rows(bad, rows)

    index, weights = _select(logits, bias, keep, scaling)
    affinity, chosen = _local_affinity(index, weights, keep, base, local)

    rounded = nl.ndarray((rows, latent), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rounded, src=hidden)
    clean = nl.ndarray((rows, latent), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=clean, value=0.0)
    nisa.tensor_copy_predicated(dst=clean, src=rounded, predicate=keep.ap(
        pattern=[[keep.shape[1], rows], [0, latent]]))
    stationary, stationary_scale = _latent_mx(clean, rows, latent)

    acc = nl.ndarray((rows, latent), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    _visit_experts(chosen, affinity, stationary, stationary_scale, bank, program, acc,
                   beta, linear_beta)

    own = latent // programs
    if programs == 2:
        other = 1 - program
        theirs = nl.ndarray((rows, own), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=acc[:, other * own:(other + 1) * own], dst=theirs,
                      send_to_rank=other, recv_from_rank=other, pipe_id=0)
        total = nl.ndarray((rows, own), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=total, data1=acc[:, program * own:(program + 1) * own],
                           data2=theirs, op=nl.add)
    else:
        total = acc
    marked = nl.ndarray((rows, own), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=marked, value=_NAN)
    nisa.tensor_copy_predicated(dst=marked, src=total, predicate=keep.ap(
        pattern=[[keep.shape[1], rows], [0, own]]))

    width_c = latent // chunks
    own_chunks = chunks // programs
    out = nl.ndarray((replicas, chunks, per_replica, width_c), dtype=nl.float32,
                     buffer=nl.shared_hbm)
    for r in range(replicas):
        nisa.dma_copy(
            dst=out.ap(pattern=[[width_c, per_replica], [per_replica * width_c, own_chunks],
                                [1, width_c]],
                       offset=(r * chunks + program * own_chunks) * per_replica * width_c),
            src=marked[r * per_replica:(r + 1) * per_replica, :])
    return out
