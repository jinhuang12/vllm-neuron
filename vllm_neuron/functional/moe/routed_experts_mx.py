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

A token whose logits or latent row hold a non-finite value chooses no expert and
gets a NaN output row; its non-finite latent values stream as zeros. All three use
selects.

At prefill a second kernel runs each local expert on the rows gathered for it, in
128-row tiles, with every expert at a static address; under LNC 2 the programs take
alternate experts, each the full expert width.

Weight bank, per local expert ``e`` and program ``g`` of ``G`` (the loader's
repack): ``gate_up_weight [E, G, 2NI, 128, NK*512]`` uint16, four fp4 values per
element, partition ``p`` of a K tile holding K values ``4p`` to ``4p + 3``;
chunk ``n < NI`` is gate rows ``g*I/G + n*512 + c``, chunk ``n >= NI`` the up
rows; ``down_weight [E, G, D/512, 128, NI*512]`` likewise over the program's
expert-width columns. The scales are E8M0 bytes, one per 32 contraction values, in
3 slots per quadrant: ``gate_up_scale [E, G, 48, 2NI*NK*512/3]`` and ``down_scale
[E, G, 48, (D/512)*NI*512/3]``. Block ``b`` of ``B`` (``b = n*NK + k`` for gate/up,
``d*NI + k`` for down) sits in slot ``s`` at local block ``l``, ``s, l = divmod(b,
B/3)``; row ``12q + 4s + r``, column ``l*512 + c`` holds the scale of output column
``c`` over K values ``32(4q + r)`` to ``32(4q + r) + 31`` of that block. Each
quadrant's 12 rows load in one copy. ``NK = D/512``, ``NI = I/(512G)``.
"""

import contextlib

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
_SLOTS = 3  # scale blocks side by side in each quadrant's rows
_GROUP = 16  # tokens per latent group: 16 tokens by 7 K tiles fill 112 partitions
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
    expose_codes: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        expose_codes: a test mode that also returns each visited expert's MXFP8
            intermediate, the input of its down product.

    Returns:
        ``[R, C, T/R, D/C]`` fp32: entry ``[r, c, t, j]`` is token ``r*T/R + t``,
        latent column ``c*D/C + j``. With ``expose_codes``, also the codes ``[E, G,
        128, NI*P]`` int32 and the block scales ``[E, G, 128, NI*P]`` uint8 of each
        local expert the loop visits (other experts' entries are undefined), ``P`` the
        rows rounded up to even: byte ``j`` of code ``[e, g, p, k*P + t]`` is the e4m3
        code of row ``t`` at intermediate index ``g*I/G + k*512 + 4p + j``, and scale
        ``[e, g, 32q + r, k*P + t]`` is the E8M0 scale of that row's 32 values from
        ``g*I/G + k*512 + 32(4q + r)``.
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
    assert gate_up_scale.shape[2] == down_scale.shape[2] == 4 * 4 * _SLOTS, (
        "MX scales hold 3 slots of 4 rows per quadrant"
    )
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
        expose_codes=expose_codes,
    )


# ---------------------------------------------------------------------------- #
# Kernel pieces.
# ---------------------------------------------------------------------------- #


def routed_expert_tiles_mx(
    rows: torch.Tensor,
    weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    beta: float,
    linear_beta: float,
    expose_codes: bool = False,
):
    """Weighted outputs ``[N, R, D]`` fp32 of local expert ``n`` on the rows gathered for it.

    ``rows [N, R, D]`` fp32 holds expert ``n``'s gathered latent rows, ``R`` a multiple
    of 128, and ``weights [N, R]`` fp32 their routing weights, 0 on padding rows. Each
    item runs its expert at a static address, every bank half, in 128-row tiles; under
    LNC 2 the programs take alternate items. A row with a non-finite latent value gets a
    NaN output row. With ``expose_codes`` the call also returns the MXFP8 intermediate
    codes ``[N, G, 128, NI*R]`` int32 and scales uint8, laid out as ``routed_experts_mx``
    lays them out with ``R`` rows.
    """
    items, per_item, latent = rows.shape
    experts, _, _, _, packed = gate_up_weight.shape
    assert rows.dtype == weights.dtype == torch.float32, "rows and weights must be fp32"
    assert tuple(weights.shape) == (items, per_item), f"weights must be [{items}, {per_item}]"
    assert per_item % 128 == 0, f"R={per_item} must be a multiple of 128"
    assert items <= experts and latent == packed, (
        f"rows {tuple(rows.shape)} must be at most [{experts}, R, {packed}]")
    assert gate_up_weight.dtype == down_weight.dtype == torch.uint16, "MXFP4 weights are uint16"
    assert gate_up_scale.dtype == down_scale.dtype == torch.uint8, "MX scales are uint8"
    assert gate_up_scale.shape[2] == down_scale.shape[2] == 4 * 4 * _SLOTS, (
        "MX scales hold 3 slots of 4 rows per quadrant"
    )
    return wrap_nki(_routed_expert_tiles_kernel)[get_logical_nc_config()](
        rows=rows,
        weights=weights,
        gate_up_weight=gate_up_weight,
        gate_up_scale=gate_up_scale,
        down_weight=down_weight,
        down_scale=down_scale,
        beta=beta,
        linear_beta=linear_beta,
        expose_codes=expose_codes,
    )


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


def _latent_groups(routed, tokens, rows, latent):
    """The latent of ``routed`` in bf16 groups of up to 16 rows, partition ``k*n + t`` holding K tile ``k`` of row ``t``.

    Each K tile of a group lands in one DMA, ``n`` partitions of one 512-value run each,
    so the loads and the checks spread over up to 112 partitions. A non-finite value has
    its partition's tile zeroed, so no transpose carries it to another token. Rows past T
    read as zeros. Returns the groups as (first row, row count ``n``, tile) and
    ``bad [rows, 1]``, each row's count of non-finite values.
    """
    width = routed.shape[1]
    tiles = latent // _KT
    bad = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.psum)
    groups = []
    for g, first in enumerate(range(0, rows, _GROUP)):
        count = min(_GROUP, rows - first)
        real = min(count, tokens - first)
        parts = tiles * count
        part = nl.ndarray((parts, _KT), dtype=nl.float32, buffer=nl.sbuf)
        if real < count:
            nisa.memset(dst=part, value=0.0)
        for k in range(tiles):
            nisa.dma_copy(dst=part[k * count:k * count + real, :], src=routed.ap(
                pattern=[[width, real], [1, _KT]], offset=first * width + k * _KT),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync if k % 2 == 0 else nisa.engine.scalar)
        rounded = nl.ndarray((parts, _KT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rounded, src=part)
        flags = _tile(parts, 1)
        _nonfinite_count(rounded, _KT, flags)
        clean = nl.ndarray((parts, _KT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=clean, value=0.0)
        finite = _tile(parts, 1, nl.uint8)
        nisa.tensor_scalar(dst=finite, data=flags, op0=nl.equal, operand0=0.0)
        nisa.tensor_copy_predicated(dst=clean, src=rounded, predicate=finite.ap(
            pattern=[[1, parts], [0, _KT]]))
        # owner[k*n + t, r] = 1 where r is row first + t, built transposed and flipped on the PE.
        lag = _tile(rows, parts)
        nisa.iota(dst=lag, pattern=[[0, tiles], [1, count]], offset=first, channel_multiplier=-1)
        match = _tile(rows, parts)
        nisa.tensor_scalar(dst=match, data=lag, op0=nl.equal, operand0=0.0)
        flipped = nl.ndarray((parts, rows), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=flipped, data=match)
        owner = _tile(parts, rows)
        nisa.tensor_copy(dst=owner, src=flipped)
        nisa.nc_matmul(dst=bad, stationary=owner, moving=flags, accumulate=g > 0)
        groups.append((first, count, clean))
    return groups, bad


def _flip(src, rows, tiles, natural=None):
    """``natural [128, tiles, P, 4]`` bf16 of ``src [P, tiles*512]``: ``natural[p, k, t, q] = src[t, 512k + 4p + q]``.

    PE transposes into PSUM tiles of at most 2 KB per partition.
    """
    if natural is None:
        natural = nl.ndarray((128, tiles, rows, _Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    span = max(1, 1024 // (_Q * rows))
    width = tiles * _KT
    for k0 in range(0, tiles, span):
        ks = min(span, tiles - k0)
        flipped = nl.ndarray((128, ks, _Q, rows), dtype=nl.bfloat16, buffer=nl.psum)
        for k in range(ks):
            for q in range(_Q):
                nisa.nc_transpose(dst=flipped[:, k, q, :], data=src.ap(
                    pattern=[[width, rows], [_Q, 128]], offset=(k0 + k) * _KT + q))
        nisa.tensor_copy(dst=natural[:, k0:k0 + ks, :, :], src=flipped.ap(
            pattern=[[ks * _Q * rows, 128], [_Q * rows, ks], [1, rows], [rows, _Q]]))
    return natural


def _tile_work(latent):
    """The working tiles one 128-row tile's quantization reuses: fp32 rows, bf16 rows, natural."""
    return (_tile(128, latent), nl.ndarray((128, latent), dtype=nl.bfloat16, buffer=nl.sbuf),
            nl.ndarray((128, latent // _KT, 128, _Q), dtype=nl.bfloat16, buffer=nl.sbuf))


def _tile_state(latent):
    """One 128-row tile's state across the bank halves: stationaries, scales, keep, weight, sum."""
    tiles = latent // _KT
    return (nl.ndarray((128, tiles, 128), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf),
            nl.ndarray((128, tiles, 128), dtype=nl.uint8, buffer=nl.sbuf),
            _tile(128, 1, nl.uint8), _tile(128, 1), _tile(128, latent))


def _tile_mx(rows, item, first, work, state):
    """Quantize 128 rows of ``rows[item]`` into ``state``; zero its sum and read its weights.

    A row with a non-finite value streams as zeros, so no transpose carries it to
    another row; its keep entry is 0.
    """
    per_item, latent = rows.shape[1:]
    part, rounded, natural = work
    quant, scale, keep, _, acc = state
    nisa.dma_copy(dst=part, src=rows.ap(pattern=[[latent, 128], [1, latent]],
                                        offset=(item * per_item + first) * latent),
                  dge_mode=nisa.dge_mode.hwdge)
    nisa.tensor_copy(dst=rounded, src=part)
    bad = _tile(128, 1)
    _nonfinite_count(rounded, latent, bad)
    nisa.tensor_scalar(dst=keep, data=bad, op0=nl.equal, operand0=0.0)
    drop = _tile(128, 1, nl.uint8)
    nisa.tensor_scalar(dst=drop, data=bad, op0=nl.not_equal, operand0=0.0)
    zero = _tile(128, 1, nl.bfloat16)
    nisa.memset(dst=zero, value=0.0)
    nisa.tensor_copy_predicated(dst=rounded, src=zero.ap(pattern=[[1, 128], [0, latent]]),
                                predicate=drop.ap(pattern=[[1, 128], [0, latent]]))
    _flip(rounded, 128, latent // _KT, natural)
    nisa.quantize_mx(dst=quant, src=natural, dst_scale=scale)
    nisa.memset(dst=acc, value=0.0)


def _latent_mx(groups, rows, latent):
    """MXFP8 stationaries ``[128, D/512, P]`` and scales of the latent groups.

    Partition ``p`` of K tile ``k`` holds latent values ``k*512 + 4p .. + 3``; each
    group of 8 partitions by 4 values is one token's 32 consecutive values.
    """
    tiles = latent // _KT
    natural = nl.ndarray((128, tiles, rows, _Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first, count, clean in groups:
        parts = tiles * count
        flipped = nl.ndarray((128, _Q, parts), dtype=nl.bfloat16, buffer=nl.psum)
        for q in range(_Q):
            nisa.nc_transpose(dst=flipped[:, q, :], data=clean.ap(
                pattern=[[_KT, parts], [_Q, 128]], offset=q))
        nisa.tensor_copy(
            dst=natural.ap(pattern=[[tiles * rows * _Q, 128], [rows * _Q, tiles], [_Q, count],
                                    [1, _Q]], offset=first * _Q),
            src=flipped.ap(pattern=[[_Q * parts, 128], [count, tiles], [1, count], [parts, _Q]]))
    quant = nl.ndarray((128, tiles, rows), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scale = nl.ndarray((128, tiles, rows), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=quant, src=natural, dst_scale=scale)
    return quant, scale


def _operand_tiles(nk, ni, nd, defined=False):
    """SBUF tiles for one expert's operands on one program, allocated once.

    With ``defined``, the scales start at 0, so a pass over the tiles before any load
    reads finite values.
    """
    gate_up = nl.ndarray((128, 2 * ni, nk * _KT), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    down = nl.ndarray((128, nd, ni * _KT), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    scales = []
    for blocks in (2 * ni * nk, nd * ni):
        scale = nl.ndarray((128, blocks // _SLOTS * _KT), dtype=nl.uint8, buffer=nl.sbuf)
        if defined:
            nisa.memset(dst=scale, value=0, engine=nisa.engine.gpsimd)
        else:
            # The products read only the 12 rows per quadrant that every load writes.
            nisa.memset(dst=scale[0:1, 0:4], value=0, engine=nisa.engine.gpsimd)
        scales.append(scale)
    return gate_up, scales[0], down, scales[1]


def _scale_block(scale, block, blocks):
    """The scale view of block ``block`` of ``blocks`` in a 3-slot scale tile."""
    slot, local = divmod(block, blocks // _SLOTS)
    return scale[4 * slot:, local * _KT:(local + 1) * _KT]


def _load_operands(bank, expert, half, tiles, dge):
    """DMA expert ``expert`` of ``bank`` for ``half``: ten DMAs.

    ``expert`` is an int, a static address, or a [1, 1] uint32 tile read at run time.
    All go on the queue of ``dge``. Run-time loads issue in program order, each weight
    ahead of its scales, so the weight packets start first and the scale packets fill in
    behind them; static loads are placed by the scheduler. A run-time id past the bank's
    last expert moves nothing.
    """
    gate_up_weight, gate_up_scale, down_weight, down_scale = bank
    gate_up, gate_up_s, down, down_s = tiles
    static = isinstance(expert, int)
    with contextlib.nullcontext() if static else nl.no_reorder():
        if static:
            index, base = None, expert
        else:
            index, base = nisa.register_alloc(), 0
            nisa.register_load(dst=index, src=expert)
        for scale_src, scale_dst, weight_src, weight_dst in (
                (gate_up_scale, gate_up_s, gate_up_weight, gate_up),
                (down_scale, down_s, down_weight, down)):
            _, halves, slot_rows, run = scale_src.shape
            _, _, count, _, cols = weight_src.shape
            nisa.dma_copy(
                dst=weight_dst,
                src=weight_src.ap(pattern=[[cols, 128], [128 * cols, count], [1, cols]],
                                  offset=(base * halves + half) * count * 128 * cols,
                                  scalar_offset=index, indirect_dim=0),
                oob_mode=nisa.oob_mode.skip, **dge)
            quad_rows = slot_rows // 4
            for quad in range(4):
                nisa.dma_copy(
                    dst=scale_dst[32 * quad:32 * quad + quad_rows, :],
                    src=scale_src.ap(pattern=[[run, quad_rows], [1, run]],
                                     offset=((base * halves + half) * slot_rows
                                             + quad * quad_rows) * run,
                                     scalar_offset=index, indirect_dim=0),
                    oob_mode=nisa.oob_mode.skip, **dge)


def _expert_rows(stationary, stationary_scale, tiles, affinity, acc, beta, linear_beta, sink=None):
    """``acc [P, D] += affinity [P, 1] * expert(rows)`` for one loaded expert's program half.

    ``sink``, in the test mode, returns the (codes, scales) HBM views that receive the
    quantized intermediate; it is called where the stores issue.
    """
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
                                  moving_scale=_scale_block(gate_up_s, n * nk + k, 2 * ni * nk),
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
    natural = _flip(act.reshape((rows, ni * _KT)), rows, ni)
    act_q = nl.ndarray((128, ni, rows), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    act_s = nl.ndarray((128, ni, rows), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=act_q, src=natural, dst_scale=act_s)
    if sink is not None:
        codes_dst, scales_dst = sink()
        nisa.dma_copy(dst=codes_dst, src=act_q.view(nl.int32).reshape((128, ni * rows)),
                      oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=scales_dst, src=act_s.reshape((128, ni * rows)),
                      oob_mode=nisa.oob_mode.skip)
    for d in range(nd):
        out = nl.ndarray((rows, _KT), dtype=nl.float32, buffer=nl.psum)
        for k in range(ni):
            nisa.nc_matmul_mx(dst=out, stationary=act_q[:, k, :], moving=down[:, d, k * _KT:(k + 1) * _KT],
                              stationary_scale=act_s[:, k, :],
                              moving_scale=_scale_block(down_s, d * ni + k, nd * ni),
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


def _code_sink(exposed, table, p, program, dge):
    """Stage the local expert in column ``p`` of ``table`` for the test mode's outputs.

    Returns a function giving the outputs' HBM views at that expert. It loads the expert into
    a register, so call it where the stores issue: the register and the stores that use it
    must sit in one ordered block.
    """
    expert_tile = _tile(1, 1, nl.uint32)
    _stage(table, p, expert_tile, dge)

    def views():
        expert = nisa.register_alloc()
        nisa.register_load(dst=expert, src=expert_tile)
        found = []
        for out in exposed:
            _, programs, parts, cols = out.shape
            found.append(out.ap(pattern=[[cols, parts], [1, cols]], offset=program * parts * cols,
                                scalar_offset=expert, indirect_dim=0))
        return found

    return views


def _visit_experts(chosen, affinity, stationary, stationary_scale, bank, program, acc,
                   beta, linear_beta, exposed=None):
    """Run each local expert with a chosen token once, ascending, next expert's loads in flight.

    One hardware loop runs the items in pairs (2p, 2p + 1) on two operand buffers swapped
    statically: item 2p + 1 loads while 2p computes, and 2p + 2 while 2p + 1 computes. Each
    pass stages the next pass's first expert id before it ends, so that load starts at the
    top of the pass. A load past the last item names no expert and moves nothing, so each of
    the ``count`` items loads once. With an odd count the last pass also computes a padding item at
    affinity 0 on the second buffer's resident data, and stores none of its codes.
    ``exposed``, in the test mode, is the (codes, scales) HBM outputs indexed by local expert.
    """
    rows, local = chosen.shape
    nk = stationary.shape[1]
    two_ni = bank[0].shape[2]
    nd = bank[2].shape[2]
    pairs = (local + 1) // 2
    padded = 2 * pairs + 2

    # Experts with at least one chosen token, compacted; padding entries hold ``local``.
    ones = _tile(rows, 1)
    nisa.memset(dst=ones, value=1.0)
    counts = nl.ndarray((1, local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=counts, stationary=ones, moving=chosen)
    hits = _tile(1, local)
    nisa.tensor_copy(dst=hits, src=counts)
    visit = _tile(1, local + 1, nl.int32)
    nisa.nonzero_with_count(dst=visit, src=hits, index_offset=0, padding_val=local)
    count = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=count, src=visit[:, local:local + 1])
    rounded = _tile(1, 1, nl.uint32)
    nisa.tensor_scalar(dst=rounded, data=count, op0=nl.add, operand0=1)
    passes_n = _tile(1, 1, nl.uint32)
    nisa.tensor_scalar(dst=passes_n, data=rounded, op0=nl.right_shift, operand0=1)
    passes = nisa.register_alloc()
    nisa.register_load(dst=passes, src=passes_n)
    flat = _tile(1, padded, nl.int32)
    nisa.memset(dst=flat, value=local)
    nisa.tensor_copy(dst=flat[:, 0:local], src=visit[:, 0:local])

    # Item i's affinity column, aff[:, flat[i]], by a select and an exact sum; 0 for padding.
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

    # Per pair position q: item 2p + q's expert (q < 4) and affinity column (q < 2) at
    # loop register p.
    expert_at, aff_at = [], []
    for q in range(4):
        e_q = _tile(1, pairs, nl.int32)
        nisa.tensor_copy(dst=e_q, src=flat.ap(pattern=[[padded, 1], [2, pairs]], offset=q))
        expert_at.append(e_q)
    for q in range(2):
        a_q = nl.ndarray((rows, pairs), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=a_q, src=item_aff.ap(pattern=[[padded, rows], [2, pairs]], offset=q))
        aff_at.append(a_q)

    buffers = [_operand_tiles(nk, two_ni // 2, nd, defined=b == 1) for b in range(2)]
    loads = {}  # the compiler's descriptor queue, on the GpSimd engine
    # Table entries go on the Sync engine's hardware queue, so they wait behind no expert load.
    entries = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    first = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=first, src=flat[:, 0:1])
    _load_operands(bank, first, program, buffers[0], loads)
    upcoming = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=upcoming, src=flat[:, 1:2])
    held = _tile(1, 1, nl.uint32)

    def compute(q, p, column, tiles):
        sink = None if exposed is None else _code_sink(exposed, expert_at[q], p, program, entries)
        # In program order, so no engine starts the next expert ahead of this one's products.
        with nl.no_reorder():
            _expert_rows(stationary, stationary_scale, tiles, column, acc, beta, linear_beta, sink)

    def run_pair(p):
        # The pass's table entries stage ahead of its expert loads: an entry staged after a
        # run-time load waits for that load to land, and the second item's products with it.
        # A full-height tile starts at partition 0, as the accumulator does.
        columns = [nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)[0:rows, :] for _ in range(2)]
        for q in range(2):
            _stage(aff_at[q], p, columns[q], entries)
        _stage(expert_at[2], p, held, entries)
        _load_operands(bank, upcoming, program, buffers[1], loads)
        compute(0, p, columns[0], buffers[0])
        _load_operands(bank, held, program, buffers[0], loads)
        compute(1, p, columns[1], buffers[1])
        _stage(expert_at[3], p, upcoming, entries)

    nl.fori_loop(0, passes, run_pair)


@nki.jit
def _routed_experts_kernel(routed, score_bias, expert_base, gate_up_weight, gate_up_scale,
                           down_weight, down_scale, beta, linear_beta, scaling, replicas, chunks,
                           expose_codes=False):
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
    bias = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias, src=score_bias.ap(pattern=[[0, rows], [1, experts]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    base_i = _tile(rows, 1, nl.int32)
    nisa.dma_copy(dst=base_i, src=expert_base.ap(pattern=[[0, rows], [1, 1]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    base = _tile(rows, 1)
    nisa.tensor_copy(dst=base, src=base_i)

    groups, bad_hidden = _latent_groups(routed, tokens, rows, latent)
    bad_logit = _tile(rows, 1)
    _nonfinite_count(logits, experts, bad_logit)
    bad = _tile(rows, 1)
    nisa.tensor_tensor(dst=bad, data1=bad_logit, data2=bad_hidden, op=nl.add)
    keep = _keep_rows(bad, rows)

    index, weights = _select(logits, bias, keep, scaling)
    affinity, chosen = _local_affinity(index, weights, keep, base, local)
    stationary, stationary_scale = _latent_mx(groups, rows, latent)

    acc = nl.ndarray((rows, latent), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    exposed = None
    if expose_codes:
        code_cols = two_ni // 2 * rows
        exposed = tuple(nl.ndarray((local, programs, 128, code_cols), dtype=dtype, buffer=nl.shared_hbm)
                        for dtype in (nl.int32, nl.uint8))
    _visit_experts(chosen, affinity, stationary, stationary_scale, bank, program, acc,
                   beta, linear_beta, exposed)

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
    if expose_codes:
        return out, exposed[0], exposed[1]
    return out


@nki.jit
def _routed_expert_tiles_kernel(rows, weights, gate_up_weight, gate_up_scale, down_weight,
                                down_scale, beta, linear_beta, expose_codes=False):
    """Alternate items per program: each item's expert, every bank half, over its 128-row tiles."""
    items, per_item, latent = rows.shape
    _, halves, two_ni, _, _ = gate_up_weight.shape
    nk = latent // _KT
    ni = two_ni // 2
    nd = down_weight.shape[2]
    programs = nl.num_programs(0) if nl.program_ndim() != 0 else 1
    program = nl.program_id(0) if programs > 1 else 0
    bank = (gate_up_weight.view(nl.float4_e2m1fn_x4), gate_up_scale,
            down_weight.view(nl.float4_e2m1fn_x4), down_scale)
    out = nl.ndarray((items, per_item, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    code_cols = ni * per_item
    exposed = None
    if expose_codes:
        exposed = tuple(nl.ndarray((items, halves, 128, code_cols), dtype=dtype, buffer=nl.shared_hbm)
                        for dtype in (nl.int32, nl.uint8))
    buffers = [_operand_tiles(nk, ni, nd) for _ in range(2)]
    work = _tile_work(latent)
    # Two items' state, so one item quantizes while the other computes.
    states = [[_tile_state(latent) for _ in range(0, per_item, 128)] for _ in range(2)]
    nan = _tile(128, 1)
    nisa.memset(dst=nan, value=_NAN)
    visit = 0
    for slot, item in enumerate(range(program, items, programs)):
        state = states[slot % 2]
        for r, first in enumerate(range(0, per_item, 128)):
            _tile_mx(rows, item, first, work, state[r])
            nisa.dma_copy(dst=state[r][3], src=weights.ap(pattern=[[1, 128], [1, 1]],
                                                          offset=item * per_item + first))
        for half in range(halves):
            tiles = buffers[visit % 2]
            visit += 1
            _load_operands(bank, item, half, tiles, {})
            for r, first in enumerate(range(0, per_item, 128)):
                stationary, stationary_scale, _, weight, acc = state[r]
                sink = None
                if expose_codes:
                    views = tuple(t.ap(pattern=[[code_cols, 128], [per_item, ni], [1, 128]],
                                       offset=(item * halves + half) * 128 * code_cols + first)
                                  for t in exposed)
                    sink = lambda views=views: views
                _expert_rows(stationary, stationary_scale, tiles, weight, acc, beta, linear_beta,
                             sink)
        for r, first in enumerate(range(0, per_item, 128)):
            _, _, keep, _, acc = state[r]
            drop = _tile(128, 1, nl.uint8)
            nisa.tensor_scalar(dst=drop, data=keep, op0=nl.equal, operand0=0)
            nisa.tensor_copy_predicated(dst=acc, src=nan.ap(pattern=[[1, 128], [0, latent]]),
                                        predicate=drop.ap(pattern=[[1, 128], [0, latent]]))
            nisa.dma_copy(dst=out.ap(pattern=[[latent, 128], [1, latent]],
                                     offset=(item * per_item + first) * latent), src=acc)
    if expose_codes:
        return out, exposed[0], exposed[1]
    return out
