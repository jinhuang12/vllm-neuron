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

Past 128 tokens the rows run in token tiles of at most 128 rows. Selection, the latent
quantization and the products run per tile; the counts behind the visited experts and
the replica choice sum over the tiles. Each expert visit loads the weights once and runs
every tile on them. A token's row never meets another token's in a sum, so the tiles
change no value: at 128 tokens or fewer the call is one tile.

A token whose logits or latent row hold a non-finite value chooses no expert and
gets a NaN output row; its non-finite latent values stream as zeros. All three use
selects.

With ``r`` replica slots the bank holds the rank's own experts, then ``r`` replicas:
slot ``j`` of rank ``k``'s experts has a second home on rank ``k + 1 + j*(R // (r +
1))`` mod ``R``, ``R`` the expert ranks. Every rank holds every token's logits, so
every rank makes the same choice. A chosen expert with no replica runs on its home.
The chosen experts with a replica are taken slot by slot, then by bit ``v`` of the
home rank (``2^v`` the largest power of two dividing the slot's rank offset), then
by rank, and each runs on whichever of its two homes has run fewer chosen experts
so far, its own home on a tie. Within one slot and bit no two experts share a home,
so the kernel takes each such group at once. The loop then visits the bank entries
this rank runs. Where the kernel cannot run, a torch branch computes the same partial.

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
from vllm_neuron.utils.neuron_utils import can_run_kernel

TOP_K = 16
#: Added to the sum of the chosen scores before the renormalizing division.
RENORM_GUARD = 1e-20
_KT = 512  # contraction values per MX K tile
_Q = 4  # values per x4 element
_SLOTS = 3  # scale blocks side by side in each quadrant's rows
_GROUP = 16  # tokens per latent group: 16 tokens by 7 K tiles fill 112 partitions
#: Rows per token tile of the decode kernel.
TOKEN_TILE = 128
#: Tokens per decode call: two full token tiles.
MAX_TOKENS = 256
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


def replica_shifts(ranks: int, slots: int) -> tuple[int, ...]:
    """Rank offset of each replica slot: slot ``j`` of rank ``k`` is also on rank ``k + shift[j]`` mod ``ranks``."""
    return tuple((1 + j * (ranks // (slots + 1))) % ranks for j in range(slots))


def replica_experts(rank: int, ranks: int, own: int, slots: int) -> list[int]:
    """Global ids of the experts in ``rank``'s replica slots, in bank order after its own experts."""
    return [own * ((rank - shift) % ranks) + j for j, shift in enumerate(replica_shifts(ranks, slots))]


def _replica_groups(ranks: int, slots: int):
    """The choice order as groups ``(slot, shift, bit, side)``: the ranks whose bit ``bit`` is ``side``, ascending."""
    for j, shift in enumerate(replica_shifts(ranks, slots)):
        bit = (shift & -shift).bit_length() - 1
        for side in (0, 1):
            yield j, shift, bit, side


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
    replica_slots: int = 0,
    token_tile: int = TOKEN_TILE,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Return this rank's routed-expert partial of ``T`` decode tokens in combine order.

    Args:
        routed: ``[T, D + E]`` fp32: the latent rows, then the router logits.
        score_bias: ``[E]`` fp32 score correction bias.
        expert_base: ``[1, 1]`` int32 global id of this rank's first own expert.
        gate_up_weight, gate_up_scale, down_weight, down_scale: the local bank: the own
            experts, then the ``replica_slots`` replicas of :func:`replica_experts`.
        beta: SiTU gate beta.
        linear_beta: SiTU up beta.
        scaling: routed scaling factor.
        replicas: token groups ``R`` in ``routed``, ``T/R`` rows each.
        chunks: column chunks ``C`` per replica in the combine.
        expose_codes: a test mode that also returns each visited expert's MXFP8
            intermediate, the input of its down product.
        replica_slots: replicas ``r`` at the end of the bank; the expert ranks are
            ``E / (bank entries - r)``, a power of two above ``r``.
        token_tile: rows per token tile, even and at most 128; ``T`` up to
            ``MAX_TOKENS`` runs in tiles of this many rows, the last one shorter.

    Returns:
        ``[R, C, T/R, D/C]`` fp32: entry ``[r, c, t, j]`` is token ``r*T/R + t``,
        latent column ``c*D/C + j``. With ``expose_codes``, also the codes ``[E, G,
        128, NI*P]`` int32 and the block scales ``[E, G, 128, NI*P]`` uint8 of each
        local expert the loop visits (other experts' entries are undefined), ``P`` the
        rows rounded up to even: byte ``j`` of code ``[e, g, p, k*P + t]`` is the e4m3
        code of row ``t`` at intermediate index ``g*I/G + k*512 + 4p + j``, and scale
        ``[e, g, 32q + r, k*P + t]`` is the E8M0 scale of that row's 32 values from
        ``g*I/G + k*512 + 32(4q + r)``. With replica slots the test mode also returns
        ``[1, E]`` fp32, nonzero at each bank entry the loop visits, and the whole choice,
        ``[1, 2rR]`` fp32: entry ``j*R + k`` is 1 where expert ``j`` of rank ``k`` is chosen
        and runs on rank ``k``, entry ``(r + j)*R + k`` 1 where it runs on its other home.
    """
    tokens = routed.shape[0]
    experts, programs, _, _, packed = gate_up_weight.shape
    latent = packed
    own = experts - replica_slots
    ranks = score_bias.shape[0] // max(own, 1)
    assert 0 <= replica_slots < own and own * ranks == score_bias.shape[0], (
        f"{experts} bank entries less {replica_slots} replicas must divide {score_bias.shape[0]} experts")
    assert not replica_slots or replica_slots < ranks, f"{replica_slots} replicas need more ranks than {ranks}"
    if not can_run_kernel(routed):
        assert not expose_codes, "the test mode reads the kernel's codes"
        return _torch_routed_experts(routed, score_bias, expert_base, gate_up_weight, gate_up_scale,
                                     down_weight, down_scale, beta, linear_beta, scaling, replicas,
                                     chunks, replica_slots)
    assert not replica_slots or ranks & (ranks - 1) == 0, f"the kernel needs a power of two ranks, got {ranks}"
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
    assert 1 <= tokens <= MAX_TOKENS and tokens % replicas == 0, (
        f"T={tokens} must be in [1, {MAX_TOKENS}] and divide into {replicas} replicas"
    )
    assert 2 <= token_tile <= 128 and token_tile % 2 == 0, f"token_tile={token_tile} must be even, 2 to 128"
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
        replica_slots=replica_slots,
        token_tile=token_tile,
    )


def _mx8_torch(values: torch.Tensor) -> torch.Tensor:
    """bf16 rounding, then MXFP8 e4m3 over 32 consecutive values of the last dimension, back to fp32."""
    groups = values.bfloat16().float().reshape(*values.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    exponent = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 7, 0.0)
    codes = (groups / torch.exp2(exponent)).to(torch.float8_e4m3fn).float()
    return (codes * torch.exp2(exponent)).reshape(values.shape)


def _fp4_values(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """uint16 ``[out, in/4]`` MXFP4 (low nibble first) and E8M0 scales ``[out, in/32]`` as fp32 ``[out, in]``."""
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    raw = packed.contiguous().view(torch.uint8)
    nibbles = torch.stack([raw & 15, raw >> 4], dim=-1).reshape(raw.shape[0], -1, 32).long()
    return (table[nibbles] * torch.exp2(scale.float() - 127)[..., None]).reshape(raw.shape[0], -1)


def _unslot(scale: torch.Tensor) -> torch.Tensor:
    """A 3-slot scale tile ``[G, 48, B/3*512]`` as ``[G, 16, B, 512]``."""
    programs, _, width = scale.shape
    return scale.view(programs, 4, 3, 4, width // 512, 512).permute(0, 1, 3, 2, 4, 5).reshape(
        programs, 16, -1, 512)


def _expert_weights(bank, e):
    """Bank entry ``e`` as fp32 gate and up ``[I, D]`` and down ``[D, I]``, the inverse of the loader's repack."""
    gate_up_weight, gate_up_scale, down_weight, down_scale = bank
    programs, two_ni, _, cols = gate_up_weight.shape[1:]
    ni, nk, nd = two_ni // 2, cols // _KT, down_weight.shape[2]
    width, latent = programs * ni * _KT, nk * _KT
    gu = gate_up_weight[e].view(programs, 2, ni, 128, nk, _KT).permute(1, 0, 2, 5, 4, 3)
    gu_s = _unslot(gate_up_scale[e]).view(programs, 16, 2, ni, nk, _KT).permute(2, 0, 3, 5, 4, 1)
    gate_up = _fp4_values(gu.reshape(2 * width, latent // 4), gu_s.reshape(2 * width, latent // 32))
    dw = down_weight[e].view(programs, nd, 128, ni, _KT).permute(1, 4, 0, 3, 2)
    dw_s = _unslot(down_scale[e]).view(programs, 16, nd, ni, _KT).permute(2, 4, 0, 3, 1)
    down = _fp4_values(dw.reshape(latent, width // 4), dw_s.reshape(latent, width // 32))
    return gate_up[:width], gate_up[width:], down


def _replica_runs(hit: list[bool], rank: int, own: int, slots: int) -> list[bool]:
    """Which bank entries ``rank`` runs, by the replica choice over the chosen experts ``hit``."""
    ranks = len(hit) // own
    load = [sum(hit[k * own + s] for s in range(slots, own)) for k in range(ranks)]
    stays = {}
    for j, shift, bit, side in _replica_groups(ranks, slots):
        for k in range(ranks):
            e = k * own + j
            if (k >> bit) & 1 == side and hit[e]:
                other = (k + shift) % ranks
                stays[e] = load[k] <= load[other]
                load[k if stays[e] else other] += 1
    runs = [stays.get(rank * own + s, True) for s in range(slots)] + [True] * (own - slots)
    return runs + [not stays.get(e, True) for e in replica_experts(rank, ranks, own, slots)]


def _torch_routed_experts(routed, score_bias, expert_base, gate_up_weight, gate_up_scale, down_weight,
                          down_scale, beta, linear_beta, scaling, replicas, chunks, replica_slots):
    """The routed-expert partial in torch, for the CPU venue where the kernel does not run."""
    tokens, latent = routed.shape[0], gate_up_weight.shape[-1]
    local, own = gate_up_weight.shape[0], gate_up_weight.shape[0] - replica_slots
    keep = torch.isfinite(routed.float()).all(-1)
    x = _mx8_torch(torch.where(keep[:, None], routed[:, :latent].float(), 0.0))
    scores = torch.sigmoid(routed[:, latent:].float())
    choice = torch.where(keep[:, None], scores + score_bias.float(), 0.0)
    index = choice.topk(TOP_K, dim=-1).indices
    chosen = scores.gather(1, index)
    weights = torch.where(keep[:, None], chosen / (chosen.sum(-1, keepdim=True) + RENORM_GUARD) * scaling, 0.0)
    hit = torch.zeros(score_bias.shape[0], dtype=torch.bool)
    hit[index[keep].flatten()] = True
    rank = int(expert_base.reshape(-1)[0]) // own
    ids = [rank * own + s for s in range(own)] + replica_experts(rank, len(hit) // own, own, replica_slots)
    runs = _replica_runs(hit.tolist(), rank, own, replica_slots)
    bank = (gate_up_weight, gate_up_scale, down_weight, down_scale)
    partial = torch.zeros(tokens, latent)
    for e in range(local):
        if not (runs[e] and hit[ids[e]]):
            continue
        affinity = torch.where(index == ids[e], weights, 0.0).sum(-1)
        gate_w, up_w, down_w = _expert_weights(bank, e)
        gate, up = x @ gate_w.T, x @ up_w.T
        situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (
            linear_beta * torch.tanh(up / linear_beta))
        partial += affinity[:, None] * (_mx8_torch(situ) @ down_w.T)
    partial = torch.where(keep[:, None], partial, float("nan"))
    per = tokens // replicas
    return partial.view(replicas, per, chunks, latent // chunks).permute(0, 2, 1, 3).contiguous()


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


def _head(t, rows):
    """The first ``rows`` partitions of ``t``."""
    if t.shape[0] == rows:
        return t
    return t[(slice(0, rows),) + (slice(None),) * (len(t.shape) - 1)]


def _token_tiles(tokens, tile):
    """(first row, rows, real rows) of each token tile of ``tile`` rows; the last tile's rows round up to even."""
    spans = []
    for first in range(0, tokens, tile):
        real = min(tile, tokens - first)
        spans.append((first, real + real % 2, real))
    return spans


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


def _select(logits, bias, keep, scaling, matched=False):
    """Top-k of ``logits [P, E]`` by sigmoid score plus ``bias``; rows with ``keep`` 0 choose zeros.

    Returns (indices ``[P, 16]`` uint32, weights ``[P, 16]`` fp32); with ``matched`` also
    ``[P, E]`` fp32 holding ``_MIN_FLOAT`` exactly at each row's chosen experts.
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
    if matched:
        return index, weights, second
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


def _local_affinity(index, weights, keep, base, local, ids=None):
    """Dense ``[P, local]`` affinities and 0/1 choices of the experts ``base .. base + local - 1``.

    With ``ids [P, local]`` fp32, of the experts whose global ids it holds instead.
    """
    rows = index.shape[0]
    index_f = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=index_f, src=index)
    if ids is None:
        offset = nl.ndarray((rows, TOP_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=offset, data=index_f, op0=nl.subtract, operand0=base)
        slot = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=slot, pattern=[[1, local]], offset=0)
    else:
        offset, slot = index_f, ids
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


def _bank_ids(base, rows, own, slots, experts):
    """``[rows, own + slots]`` fp32 global id of each bank entry: the own experts, then the replicas."""
    step = experts // own // (slots + 1)
    ids = _tile(rows, own + slots)
    nisa.iota(dst=ids[:, 0:own], pattern=[[1, own]], offset=0)
    # Replica j holds expert own*(rank - 1 - j*step) + j, before the wrap past expert 0.
    nisa.iota(dst=ids[:, own:], pattern=[[1 - own * step, slots]], offset=-own)
    nisa.tensor_scalar(dst=ids, data=ids, op0=nl.add, operand0=base)
    wrap = _tile(rows, slots)
    nisa.tensor_scalar(dst=wrap, data=ids[:, own:], op0=nl.less, operand0=0.0,
                       op1=nl.multiply, operand1=float(experts))
    nisa.tensor_tensor(dst=ids[:, own:], data1=ids[:, own:], data2=wrap, op=nl.add)
    return ids


def _choose_replicas(matched, keep, base, own, slots, experts):
    """``[1, own + slots]`` fp32, 1 at each bank entry this rank runs under the replica choice,
    and the whole choice as ``stays`` and ``moved`` (see :func:`_choice_record`).

    ``matched`` and ``keep`` hold one entry per token tile. The loads are counts of chosen
    experts, so every step is exact in fp32.
    """
    ranks, step, local = experts // own, experts // own // (slots + 1), own + slots
    tiles = []
    for tile_matched, tile_keep in zip(matched, keep):
        rows = tile_matched.shape[0]
        picked = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=picked, data=tile_matched, op0=nl.equal, operand0=_MIN_FLOAT)
        keep_f = _tile(rows, 1)
        nisa.tensor_copy(dst=keep_f, src=tile_keep)
        tiles.append((picked, keep_f))
    # hits[e] = 1 where a kept token of any tile chose expert e.
    hits = _tile(1, experts)
    for first in range(0, experts, 512):
        width = min(512, experts - first)
        counts = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.psum)
        # The compiler sets the accumulate flags: the first tile writes, the others add.
        for picked, keep_f in tiles:
            nisa.nc_matmul(dst=counts, stationary=keep_f, moving=picked[:, first:first + width])
        nisa.tensor_scalar(dst=hits[:, first:first + width], data=counts, op0=nl.greater, operand0=0.0)
    # Each rank's load, held twice so a rank offset reads one span: first its chosen experts
    # with no replica.
    loads = _tile(1, 2 * ranks)
    nisa.tensor_reduce(dst=loads[:, 0:ranks], data=hits.ap(
        pattern=[[experts, 1], [own, ranks], [1, own - slots]], offset=slots), op=nl.add, axis=(2,))
    nisa.tensor_copy(dst=loads[:, ranks:], src=loads[:, 0:ranks])
    # stays[j*R + k]: expert own*k + j is chosen and runs on rank k. moved[g*2R + k], at both
    # copies of k: the group's expert of rank k is chosen and runs on rank k + shift.
    stays = _tile(1, slots * ranks)
    moved = _tile(1, 4 * slots * ranks)
    nisa.memset(dst=moved, value=0.0)
    for g, (j, shift, bit, side) in enumerate(_replica_groups(ranks, slots)):
        block = 1 << bit
        span = [[2 * block, ranks // (2 * block)], [1, block]]
        first = side * block
        homes = loads.ap(pattern=[[2 * ranks, 1], *span], offset=first)
        others = loads.ap(pattern=[[2 * ranks, 1], *span], offset=first + shift)
        stay = stays.ap(pattern=[[slots * ranks, 1], *span], offset=j * ranks + first)
        stay2 = stays.ap(pattern=[[slots * ranks, 1], [0, 2], *span], offset=j * ranks + first)
        chosen = [[2 * block * own, ranks // (2 * block)], [own, block]]
        nisa.tensor_tensor(dst=stay, data1=homes, data2=others, op=nl.less_equal)
        nisa.tensor_tensor(dst=stay, data1=stay, data2=hits.ap(
            pattern=[[experts, 1], *chosen], offset=own * first + j), op=nl.multiply)
        nisa.tensor_tensor(dst=moved.ap(pattern=[[4 * slots * ranks, 1], [ranks, 2], *span],
                                        offset=2 * g * ranks + first),
                           data1=hits.ap(pattern=[[experts, 1], [0, 2], *chosen], offset=own * first + j),
                           data2=stay2, op=nl.subtract)
        both = loads.ap(pattern=[[2 * ranks, 1], [ranks, 2], *span], offset=first)
        nisa.tensor_tensor(dst=both, data1=both, data2=stay2, op=nl.add)
        nisa.tensor_tensor(dst=loads, data1=loads, data2=moved.ap(
            pattern=[[4 * slots * ranks, 1], [0, 2], [1, ranks]], offset=2 * g * ranks + ranks - shift),
            op=nl.add)
    # This rank's entries: its own experts j < r where they stay, its replicas where they moved.
    rank_ids = _tile(1, ranks)
    nisa.iota(dst=rank_ids, pattern=[[own, ranks]], offset=0)
    mine = _tile(1, ranks)
    nisa.tensor_scalar(dst=mine, data=rank_ids, op0=nl.equal, operand0=base[0:1, 0:1])
    runs = _tile(1, local)
    nisa.memset(dst=runs, value=1.0)
    kept = nl.ndarray((1, slots, ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=kept, data1=stays.ap(pattern=[[slots * ranks, 1], [ranks, slots], [1, ranks]]),
                       data2=mine.ap(pattern=[[ranks, 1], [0, slots], [1, ranks]]), op=nl.multiply)
    nisa.tensor_reduce(dst=runs[:, 0:slots], data=kept, op=nl.add, axis=(2,))
    # Replica j is expert own*(rank - shift_j) + j; shift_j = 1 + j*step.
    taken = nl.ndarray((1, slots, 2, ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=taken, data1=moved.ap(
        pattern=[[4 * slots * ranks, 1], [4 * ranks - step, slots], [2 * ranks, 2], [1, ranks]],
        offset=ranks - 1), data2=mine.ap(pattern=[[ranks, 1], [0, slots], [0, 2], [1, ranks]]),
        op=nl.multiply)
    nisa.tensor_reduce(dst=runs[:, own:local], data=taken, op=nl.add, axis=(2, 3))
    return runs, stays, moved


def _choice_record(stays, moved, slots, ranks):
    """``[1, 2*slots*ranks]`` fp32 for the test mode: ``[j*R + k]`` is 1 where expert ``own*k + j``
    is chosen and runs on rank ``k``, ``[(slots + j)*R + k]`` 1 where it runs on its other home."""
    record = _tile(1, 2 * slots * ranks)
    nisa.tensor_copy(dst=record[:, 0:slots * ranks], src=stays)
    # Group (j, side) holds its ranks at 4jR + 2R*side + k and zeros elsewhere.
    sides = [moved.ap(pattern=[[4 * slots * ranks, 1], [4 * ranks, slots], [1, ranks]], offset=2 * ranks * side)
             for side in (0, 1)]
    nisa.tensor_tensor(dst=record.ap(pattern=[[2 * slots * ranks, 1], [ranks, slots], [1, ranks]],
                                     offset=slots * ranks), data1=sides[0], data2=sides[1], op=nl.add)
    return record


def _latent_groups(routed, start, tokens, rows, latent):
    """The latent of ``routed`` from row ``start`` in bf16 groups of up to 16 rows.

    Partition ``k*n + t`` of a group holds K tile ``k`` of its row ``t``. Each K tile of a
    group lands in one DMA, ``n`` partitions of one 512-value run each, so the loads and the
    checks spread over up to 112 partitions. A non-finite value has
    its partition's tile zeroed, so no transpose carries it to another token. Rows past
    ``tokens`` read as zeros. Returns the groups as (first row, row count ``n``, tile), rows
    counted from ``start``, and ``bad [rows, 1]``, each row's count of non-finite values.
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
                pattern=[[width, real], [1, _KT]], offset=(start + first) * width + k * _KT),
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


def _operand_tiles(nk, ni, nd):
    """SBUF tiles for one expert's operands on one program, allocated once."""
    gate_up = nl.ndarray((128, 2 * ni, nk * _KT), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    down = nl.ndarray((128, nd, ni * _KT), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    scales = []
    for blocks in (2 * ni * nk, nd * ni):
        scale = nl.ndarray((128, blocks // _SLOTS * _KT), dtype=nl.uint8, buffer=nl.sbuf)
        # The products read only the 12 rows per quadrant that every load writes.
        nisa.memset(dst=scale[0:1, 0:4], value=0, engine=nisa.engine.gpsimd)
        scales.append(scale)
    return gate_up, scales[0], down, scales[1]


def _scale_block(scale, block, blocks):
    """The scale view of block ``block`` of ``blocks`` in a 3-slot scale tile."""
    slot, local = divmod(block, blocks // _SLOTS)
    return scale[4 * slot:, local * _KT:(local + 1) * _KT]


def _load_operands(bank, expert, half, tiles, dge, spread=False):
    """DMA expert ``expert`` of ``bank`` for ``half``: ten DMAs.

    ``expert`` is an int (a static address), a [1, 1] uint32 tile read at run time, or a
    register that holds the id. All go on the queue of ``dge``. Run-time loads issue in
    program order, each weight ahead of its scales, so the weight packets start first and
    the scale packets fill in behind them; static loads are placed by the scheduler. A
    copy's rows go to the copy engines by row index, so a quadrant's 12 scale rows reach 12
    of the 16; with ``spread`` each row goes in 4 parts, and the quadrant's 48 pieces reach
    all 16, 3 each. A run-time id past the bank's last expert moves nothing.
    """
    gate_up_weight, gate_up_scale, down_weight, down_scale = bank
    gate_up, gate_up_s, down, down_s = tiles
    static = isinstance(expert, int)
    with contextlib.nullcontext() if static else nl.no_reorder():
        if static:
            index, base = None, expert
        elif isinstance(expert, nisa.VirtualRegister):
            index, base = expert, 0
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
            pattern = [[run, quad_rows], [1, run]]
            if spread:
                # Four parts per row, 32-byte aligned, each 128 bytes into the next, so the
                # parts do not merge back into rows; the overlaps carry the same bytes.
                part = (run // 128 - 1) * 32
                pattern = [[run, quad_rows], [part, 4], [1, run - 3 * part]]
            for quad in range(4):
                rows = scale_dst[32 * quad:32 * quad + quad_rows, :]
                nisa.dma_copy(
                    dst=rows.ap(pattern=pattern) if spread else rows,
                    src=scale_src.ap(pattern=pattern,
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


def _stage_pair(pairs, p, rows, dge):
    """DMA pair ``p`` (a register) of the SBUF ``pairs [rows, n, 2]`` into a new [rows, 2] tile.

    A full-height tile starts at partition 0, as the accumulator does.
    """
    _, n, two = pairs.shape
    dst = nl.ndarray((128, two), dtype=pairs.dtype, buffer=nl.sbuf)[0:rows, :]
    nisa.dma_copy(dst=dst, src=pairs.ap(pattern=[[n * two, rows], [1, two]], scalar_offset=p,
                                        indirect_dim=1), **dge)
    return dst


def _code_sink(exposed, table, p, program, dge, rows):
    """Stage the local expert in column ``p`` of ``table`` for the test mode's outputs.

    Returns a function of a token tile's first row and row count, of ``rows`` in all, giving
    the outputs' HBM views at that expert and tile. It loads the expert into a register, so
    call it where the stores issue: the register and the stores that use it must sit in one
    ordered block.
    """
    expert_tile = _tile(1, 1, nl.uint32)
    _stage(table, p, expert_tile, dge)

    def views(first, count):
        expert = nisa.register_alloc()
        nisa.register_load(dst=expert, src=expert_tile)
        found = []
        for out in exposed:
            _, programs, parts, cols = out.shape
            pattern = [[cols, parts], [1, cols]]
            if count < rows:
                pattern = [[cols, parts], [rows, cols // rows], [1, count]]
            found.append(out.ap(pattern=pattern, offset=program * parts * cols + first,
                                scalar_offset=expert, indirect_dim=0))
        return found

    return views


class _TokenTile:
    """One token tile's operands in the expert loop: its first row, chosen experts, affinities,
    MXFP8 stationaries and scales, accumulator and keep flags."""

    def __init__(self, first, chosen, affinity, stationary, stationary_scale, acc, keep):
        self.first, self.rows = first, chosen.shape[0]
        self.chosen, self.affinity, self.acc, self.keep = chosen, affinity, acc, keep
        self.stationary, self.stationary_scale = stationary, stationary_scale


def _visit_experts(tiles, bank, program, beta, linear_beta, exposed=None, runs=None, visits=None):
    """Run each local expert with a chosen token once, ascending, next expert's loads in flight.

    One hardware loop runs the full pairs (2p, 2p + 1) on two operand buffers swapped
    statically: item 2p + 1 loads while 2p computes, and 2p + 2 while 2p + 1 computes. The
    loop's condition register holds item 2p + 1's expert id, so each pass reads the id of its
    first load before the pass starts; past the last full pair it holds 0. The pass number
    lives in SBUF. A load past the last item names no expert and moves nothing, so each of the
    ``count`` items loads once. With an odd count a second loop of one pass computes the last
    item from the first buffer, where the last pair pass loaded it; no item is computed that
    was not chosen. Each item runs every token tile of ``tiles`` (:class:`_TokenTile`) on its
    loaded weights, the tiles in order.
    ``exposed``, in the test mode, is the (codes, scales) HBM outputs indexed by local expert.
    ``runs [1, local]``, with replica slots, is 0 at the entries this rank leaves to their other
    home; ``visits``, in the test mode, receives the entries' visit counts.
    """
    rows, local = tiles[0].chosen.shape
    total = sum(t.rows for t in tiles)
    nk = tiles[0].stationary.shape[1]
    two_ni = bank[0].shape[2]
    nd = bank[2].shape[2]
    pairs = (local + 1) // 2
    padded = 2 * pairs + 2

    # Experts with at least one chosen token in any tile, compacted; padding entries hold ``local``.
    ones = _tile(rows, 1)
    nisa.memset(dst=ones, value=1.0)
    counts = nl.ndarray((1, local), dtype=nl.float32, buffer=nl.psum)
    for tile in tiles:
        nisa.nc_matmul(dst=counts, stationary=_head(ones, tile.rows), moving=tile.chosen)
    hits = _tile(1, local)
    if runs is None:
        nisa.tensor_copy(dst=hits, src=counts)
    else:
        nisa.tensor_tensor(dst=hits, data1=counts, data2=runs, op=nl.multiply)
    if visits is not None:
        nisa.dma_copy(dst=visits, src=hits)
    visit = _tile(1, local + 1, nl.int32)
    nisa.nonzero_with_count(dst=visit, src=hits, index_offset=0, padding_val=local)

    buffers = [_operand_tiles(nk, two_ni // 2, nd) for _ in range(2)]
    loads = {}  # the compiler's descriptor queue, on the GpSimd engine
    # The first expert's loads go out ahead of the tables below.
    _load_operands(bank, visit[:, 0:1], program, buffers[0], loads, spread=True)

    count = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=count, src=visit[:, local:local + 1])
    pairs_n = _tile(1, 1, nl.uint32)
    nisa.tensor_scalar(dst=pairs_n, data=count, op0=nl.right_shift, operand0=1)
    rounded = _tile(1, 1, nl.uint32)
    nisa.tensor_scalar(dst=rounded, data=count, op0=nl.add, operand0=1)
    passes_n = _tile(1, 1, nl.uint32)
    nisa.tensor_scalar(dst=passes_n, data=rounded, op0=nl.right_shift, operand0=1)
    full_pairs = nisa.register_alloc()
    nisa.register_load(dst=full_pairs, src=pairs_n)
    passes = nisa.register_alloc()
    nisa.register_load(dst=passes, src=passes_n)
    flat = _tile(1, padded, nl.int32)
    nisa.memset(dst=flat, value=local)
    nisa.tensor_copy(dst=flat[:, 0:local], src=visit[:, 0:local])

    # Item i's affinity column, aff[:, flat[i]], by a select and an exact sum; 0 for padding.
    # The select pattern is the same on every row, so the tiles share the first tile's.
    flat_f = _tile(1, padded)
    nisa.tensor_copy(dst=flat_f, src=flat)
    item_expert = _row_broadcast(flat_f, rows, padded)
    slot = nl.ndarray((rows, local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=slot, pattern=[[1, local]], offset=0)
    match = nl.ndarray((rows, padded, local), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=match, data1=item_expert.ap(pattern=[[padded, rows], [1, padded], [0, local]]),
                       data2=slot.ap(pattern=[[local, rows], [0, padded], [1, local]]), op=nl.equal)
    pair_affs = []
    for tile in tiles:
        n = tile.rows
        spread = nl.ndarray((n, padded, local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=spread, value=0.0)
        nisa.tensor_copy_predicated(dst=spread, src=tile.affinity.ap(
            pattern=[[local, n], [0, padded], [1, local]]), predicate=_head(match, n))
        item_aff = nl.ndarray((n, padded), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=item_aff, data=spread, op=nl.add, axis=(2,))
        # Items 2p and 2p + 1 have adjacent affinity columns, so one stage brings both.
        pair_affs.append(item_aff.reshape((n, padded // 2, 2)))

    # Per pair position q < 3: item 2p + q's expert at pass p.
    expert_at = []
    for q in range(3):
        e_q = _tile(1, pairs, nl.int32)
        nisa.tensor_copy(dst=e_q, src=flat.ap(pattern=[[padded, 1], [2, pairs]], offset=q))
        expert_at.append(e_q)
    # The loop condition at pass p: item 2p + 1's expert id where that item is real, else 0.
    # A real item 2p + 1 follows item 2p in ascending order, so its id is at least 1.
    odd = flat_f.ap(pattern=[[padded, 1], [2, pairs + 1]], offset=1)
    condition = _tile(1, pairs + 1, nl.int32)
    nisa.scalar_tensor_tensor(dst=condition, data=odd, op0=nl.less, operand0=float(local),
                              op1=nl.multiply, operand1=odd)
    next_condition = _tile(1, pairs, nl.int32)
    nisa.tensor_copy(dst=next_condition, src=condition[:, 1:pairs + 1])
    first_pair = nisa.register_alloc()
    nisa.register_load(dst=first_pair, src=condition[:, 0:1])
    step = _tile(1, 1, nl.uint32)
    nisa.memset(dst=step, value=0)

    # Table entries go on the Sync engine's hardware queue, so they wait behind no expert load.
    entries = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    held = _tile(1, 1, nl.uint32)
    upcoming = _tile(1, 1, nl.uint32)

    def compute(q, p, columns, operands):
        sink = None if exposed is None else _code_sink(exposed, expert_at[q], p, program, entries, total)
        # In program order, so no engine starts the next expert or tile ahead of this one's products.
        with nl.no_reorder():
            for tile, column in zip(tiles, columns):
                tile_sink = None if sink is None else (lambda tile=tile: sink(tile.first, tile.rows))
                _expert_rows(tile.stationary, tile.stationary_scale, operands, column, tile.acc, beta,
                             linear_beta, tile_sink)

    def run_pair(expert):
        # The pass's table entries stage ahead of its expert loads: an entry staged after a
        # run-time load waits for that load to land, and the second item's products with it.
        columns = [_stage_pair(pair_aff, step, tile.rows, entries) for pair_aff, tile in zip(pair_affs, tiles)]
        _stage(expert_at[2], step, held, entries)
        _stage(next_condition, step, upcoming, entries)
        _load_operands(bank, expert, program, buffers[1], loads, spread=True)
        compute(0, step, [c[:, 0:1] for c in columns], buffers[0])
        _load_operands(bank, held, program, buffers[0], loads, spread=True)
        compute(1, step, [c[:, 1:2] for c in columns], buffers[1])
        nisa.tensor_scalar(dst=step, data=step, op0=nl.add, operand0=1)
        # Read before the pass ends, so the next pass's first load needs no read of its own.
        following = nisa.register_alloc()
        nisa.register_load(dst=following, src=upcoming)
        return following

    def run_last(p):
        columns = [_stage_pair(pair_aff, p, tile.rows, entries) for pair_aff, tile in zip(pair_affs, tiles)]
        compute(0, p, [c[:, 0:1] for c in columns], buffers[0])

    nl.while_loop(first_pair, run_pair)
    # Item 2p at p = count // 2, which exists only for an odd count.
    nl.fori_loop(full_pairs, passes, run_last)


@nki.jit
def _routed_experts_kernel(routed, score_bias, expert_base, gate_up_weight, gate_up_scale,
                           down_weight, down_scale, beta, linear_beta, scaling, replicas, chunks,
                           expose_codes=False, replica_slots=0, token_tile=TOKEN_TILE):
    """Selection per token tile, the replica choice, the distinct-expert loop, the LNC 2 merge and the store."""
    tokens, width = routed.shape
    local, programs, two_ni, _, gu_cols = gate_up_weight.shape
    latent = gu_cols
    experts = width - latent
    per_replica = tokens // replicas
    kernel_assert(programs == (nl.num_programs(0) if nl.program_ndim() != 0 else 1),
                  "one program per bank half")
    program = nl.program_id(0) if programs > 1 else 0
    spans = _token_tiles(tokens, token_tile)
    rows = spans[0][1]  # the first tile's rows: the tables every tile reads have this many
    total = sum(n for _, n, _ in spans)  # the stationary free size of each tile must be even
    bank = (gate_up_weight.view(nl.float4_e2m1fn_x4), gate_up_scale,
            down_weight.view(nl.float4_e2m1fn_x4), down_scale)
    own = local - replica_slots
    if replica_slots:
        kernel_assert(experts % own == 0 and (experts // own) & (experts // own - 1) == 0,
                      "replica slots need a power of two expert ranks")

    picks = []
    for t, (first, n, real) in enumerate(spans):
        # Rows past T read as NaN logits, so they choose nothing and stream as zeros.
        logits = nl.ndarray((n, experts), dtype=nl.float32, buffer=nl.sbuf)
        if n > real:
            nisa.memset(dst=logits, value=_NAN)
        nisa.dma_copy(dst=logits[0:real, :], src=routed[first:first + real, latent:width],
                      dge_mode=nisa.dge_mode.hwdge)
        if t == 0:
            bias = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bias, src=score_bias.ap(pattern=[[0, rows], [1, experts]], offset=0),
                          dge_mode=nisa.dge_mode.hwdge)
            base_i = _tile(rows, 1, nl.int32)
            nisa.dma_copy(dst=base_i, src=expert_base.ap(pattern=[[0, rows], [1, 1]], offset=0),
                          dge_mode=nisa.dge_mode.hwdge)
            base = _tile(rows, 1)
            nisa.tensor_copy(dst=base, src=base_i)

        groups, bad_hidden = _latent_groups(routed, first, real, n, latent)
        bad_logit = _tile(n, 1)
        _nonfinite_count(logits, experts, bad_logit)
        bad = _tile(n, 1)
        nisa.tensor_tensor(dst=bad, data1=bad_logit, data2=bad_hidden, op=nl.add)
        keep = _keep_rows(bad, n)

        matched = None
        if replica_slots:
            index, weights, matched = _select(logits, _head(bias, n), keep, scaling, matched=True)
            if t == 0:
                ids = _bank_ids(base, rows, own, replica_slots, experts)
            affinity, chosen = _local_affinity(index, weights, keep, _head(base, n), local, _head(ids, n))
        else:
            index, weights = _select(logits, _head(bias, n), keep, scaling)
            affinity, chosen = _local_affinity(index, weights, keep, _head(base, n), local)
        picks.append((first, groups, keep, matched, affinity, chosen))

    runs = visits = None
    if replica_slots:
        runs, stays, moved = _choose_replicas([p[3] for p in picks], [p[2] for p in picks], base, own,
                                              replica_slots, experts)
    tiles = []
    for first, groups, keep, _, affinity, chosen in picks:
        n = chosen.shape[0]
        stationary, stationary_scale = _latent_mx(groups, n, latent)
        acc = nl.ndarray((n, latent), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        tiles.append(_TokenTile(first, chosen, affinity, stationary, stationary_scale, acc, keep))

    exposed = None
    if expose_codes:
        code_cols = two_ni // 2 * total
        exposed = tuple(nl.ndarray((local, programs, 128, code_cols), dtype=dtype, buffer=nl.shared_hbm)
                        for dtype in (nl.int32, nl.uint8))
        if replica_slots:
            visits = nl.ndarray((1, local), dtype=nl.float32, buffer=nl.shared_hbm)
            ranks = experts // own
            choice = nl.ndarray((1, 2 * replica_slots * ranks), dtype=nl.float32, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=choice, src=_choice_record(stays, moved, replica_slots, ranks))
    _visit_experts(tiles, bank, program, beta, linear_beta, exposed, runs, visits)

    half = latent // programs
    width_c = latent // chunks
    own_chunks = chunks // programs
    out = nl.ndarray((replicas, chunks, per_replica, width_c), dtype=nl.float32,
                     buffer=nl.shared_hbm)
    for tile, (first, n, real) in zip(tiles, spans):
        if programs == 2:
            other = 1 - program
            theirs = nl.ndarray((n, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=tile.acc[:, other * half:(other + 1) * half], dst=theirs,
                          send_to_rank=other, recv_from_rank=other, pipe_id=0)
            summed = nl.ndarray((n, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=summed, data1=tile.acc[:, program * half:(program + 1) * half],
                               data2=theirs, op=nl.add)
        else:
            summed = tile.acc
        marked = nl.ndarray((n, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=marked, value=_NAN)
        nisa.tensor_copy_predicated(dst=marked, src=summed, predicate=tile.keep.ap(
            pattern=[[tile.keep.shape[1], n], [0, half]]))
        # Each replica's rows in this tile, at their place in the combine order.
        for r in range(replicas):
            lo, hi = max(first, r * per_replica), min(first + real, (r + 1) * per_replica)
            if lo >= hi:
                continue
            nisa.dma_copy(
                dst=out.ap(pattern=[[width_c, hi - lo], [per_replica * width_c, own_chunks],
                                    [1, width_c]],
                           offset=(r * chunks + program * own_chunks) * per_replica * width_c
                           + (lo - r * per_replica) * width_c),
                src=marked[lo - first:hi - first, :])
    if visits is not None:
        return out, exposed[0], exposed[1], visits, choice
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
