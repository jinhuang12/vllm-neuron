# SPDX-License-Identifier: Apache-2.0
"""The MoE combine's slot build as an NKI kernel: each token's rows of the fused expert emission.

``combine_slots(expert_affinities, block, rows, top_k)`` returns ``(index, valid)``, the
``[T, K]`` operands of ``token_gather_combine``, ``K = min(top_k, E)``::

    mask[t, e]  = affinities[t, e] != 0
    place[t, e] = #{t' <= t : mask[t', e]} - 1        token t's place among expert e's tokens
    blocks[e]   = ceil(#{t : mask[t, e]} / block)     expert e's blocks in the mapping
    first[e]    = sum of blocks[e'] over e' < e       expert e's first block
    row[t, e]   = first[e] * rows + place[t, e]       token t's emission row for expert e
    slot j of token t: its j-th selected expert e in expert order
    valid[t, j] = 1 and index[t, j] = row[t, e] for a filled slot, both 0 for an empty one

The emission is ``build_blockwise_mapping``'s layout with each block cut to its first
``rows`` rows, so expert ``e``'s tokens start at row ``first[e] * rows`` in token order. The
slots follow expert order, the order a scatter-add over the emission meets them in; an empty
slot reads row 0 with weight 0.

Why a kernel. In torch the slots are a ``[T, E, K]`` one-hot summed over ``E``, which
neuronx-cc lowers as reduces over a middle axis; at ``T`` = 1024, ``E`` = 18, ``K`` = 8 the slot
build is 0.232 ms of compiler ops per MoE layer, 65x its ENTITLEMENT, most of it one
``reduce f32[1024, 8]`` (MEASURED, ``reports/prefill_fixed_cost.md`` section 5).

Method. 128 tokens on the partitions; a last tile of fewer tokens is zero-filled, and its
empty partitions select nothing. The token-order count is two 0/1 products on the tensor
engine: a lower-triangular ``[128, 128]`` product with the tile's mask, plus an all-ones
product with the sum of the earlier tiles' masks. ``blocks`` counts thresholds instead of
dividing, ``ceil(c / block) = #{m >= 0 : m * block < c}`` with ``m < ceil(T / block)``, and
one product with a strictly upper-triangular ``[E, E]`` takes ``first`` to every token
partition. The slots need no one-hot: each selected expert gets the key
``(E - e) * scale + row[t, e]`` (unselected: 0), with ``scale`` (:func:`key_scale`) a power
of two above every row, and the vector engine's ``max8`` returns each token's 8 largest keys
in descending order: its selected experts in expert order, then zeros. So
``valid = key != 0`` and ``index = key mod scale``, an integer ``and``.

Numerics. Every value is an integer below ``2**24``, and every product operand is 0, 1, a
sum of at most ``ceil(T / 128)`` masks or a block count, so each product, sum, ``max8`` and
convert is exact in fp32: the outputs are bit-equal to the torch construction
(:func:`combine_slots_torch`) in any summation order.

LNC2: one program, and every loop runs over trace-time ints (fully unrolled), so there is no
register-bounded loop and no data-dependent branch.

Where the kernel does not run (:func:`can_run_combine_slots`): a device without NKI, a graph
captured in CPU mode (the simulator's dispatch drops the kernel's int arguments, as for
``moe_blockwise.build_blockwise_mapping``'s subkernels), and shapes outside the kernel's
tile limits: ``E`` above one partition tile or below ``max8``'s 8 keys, more than 8 slots,
or keys past fp32's exact integers.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_assert import kernel_assert
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Tokens per tile: the partition count; also the most experts one tile holds.
TOKEN_TILE = nl.tile_size.pmax

#: Values per partition that ``max8`` reads at least and returns: the most slots a token has.
MAX8_WIDTH = 8

#: fp32 values per partition of one PSUM bank: one tensor-engine product writes within it.
PSUM_BANK_FP32 = nl.tile_size.psum_fmax

#: fp32 holds every integer up to this bound exactly.
FP32_EXACT_INTEGERS = 2**24


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _triangle(size, strict):
    """``[size, size]`` fp32 0/1: ``tri[p, c] = p < c`` (``strict``) or ``p <= c``."""
    row = _tile(size, size)
    nisa.iota(dst=row, pattern=[[0, size]], offset=0, channel_multiplier=1)
    col = _tile(size, size)
    nisa.iota(dst=col, pattern=[[1, size]], offset=0, channel_multiplier=0)
    tri = _tile(size, size)
    nisa.tensor_tensor(dst=tri, data1=row, data2=col, op=nl.less if strict else nl.less_equal)
    return tri


def _store_by_tile(out_hbm, sbuf, slots):
    """``out_hbm[i * 128 + q, j] = sbuf[q, i, j]`` for ``j < slots``: one DMA for the whole
    tiles, one for the last tile's tokens."""
    tokens = out_hbm.shape[0]
    p = TOKEN_TILE
    full = tokens // p
    tail = tokens % p
    if full:
        nisa.dma_copy(dst=out_hbm.ap(pattern=[[slots, p], [p * slots, full], [1, slots]],
                                     offset=0), src=sbuf[:, 0:full, 0:slots])
    if tail:
        nisa.dma_copy(dst=out_hbm.ap(pattern=[[slots, tail], [1, slots]],
                                     offset=full * p * slots), src=sbuf[0:tail, full, 0:slots])


def key_scale(tokens: int, experts: int, block: int, rows: int) -> int:
    """The least power of two above every emission row: each expert holds at most
    ``ceil(T / block)`` blocks of ``rows`` rows."""
    return 1 << (experts * -(-tokens // block) * rows - 1).bit_length()


@nki.jit
def combine_slots_kernel(affinities_hbm, block, rows, slots, scale):
    """``[T, E]`` affinities -> ``([T, K]`` int32 ``index``, ``[T, K]`` fp32 ``valid)``.

    Args:
        affinities_hbm: ``[T, E]`` local router scores, zero where unselected, any float
            dtype; ``8 <= E <= 128``.
        block: tokens per block of the mapping (trace-time int).
        rows: emission rows kept per block (trace-time int).
        slots: ``K``, slots per token (trace-time int, ``1 <= K <= 8``).
        scale: :func:`key_scale` of the shapes (trace-time int; the kernel language has no
            ``int.bit_length``).
    """
    tokens, experts = affinities_hbm.shape
    p = TOKEN_TILE
    kernel_assert(MAX8_WIDTH <= experts <= p, "8 <= E <= 128: max8 reads at least 8 keys")
    kernel_assert(1 <= slots <= MAX8_WIDTH, "1 <= K <= 8")
    thresholds = -(-tokens // block)
    kernel_assert(scale & (scale - 1) == 0 and scale >= experts * thresholds * rows,
                  "scale is a power of two above every emission row")
    kernel_assert((experts + 1) * scale <= FP32_EXACT_INTEGERS, "every key is exact in fp32")
    full = tokens // p
    tail = tokens % p
    tiles = -(-tokens // p)
    # Tiles whose [128, E] fp32 counts share one PSUM bank.
    group = PSUM_BANK_FP32 // experts
    index_out = nl.ndarray((tokens, slots), dtype=nl.int32, buffer=nl.shared_hbm)
    valid_out = nl.ndarray((tokens, slots), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Constants. -------------------------------------------------------- #
    ones = _tile(p, p)
    nisa.memset(dst=ones, value=1.0)
    lower = _triangle(p, strict=False)          # lower[t', t] = t' <= t
    before = _triangle(experts, strict=True)    # before[e', e] = e' < e
    # threshold[e, m] = m * block.
    threshold = _tile(experts, thresholds)
    nisa.iota(dst=threshold, pattern=[[block, thresholds]], offset=0, channel_multiplier=0)
    # order[t, e] = (E - e) * scale: the high part of the key, larger for a lower expert.
    order = nl.ndarray((p, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=order, pattern=[[-scale, experts]], offset=experts * scale,
              channel_multiplier=0)

    # ---- The mask of every tile; earlier[:, i] = the sum of tiles 0 .. i-1. #
    # Token i * 128 + q on partition q of tile i: one DMA for the whole tiles, one for the
    # last tile's tokens (the rest of that tile stays zero).
    scores = nl.ndarray((p, tiles, experts), dtype=affinities_hbm.dtype, buffer=nl.sbuf)
    if full:
        nisa.dma_copy(dst=scores[:, 0:full, :], src=affinities_hbm.ap(
            pattern=[[experts, p], [p * experts, full], [1, experts]], offset=0))
    if tail:
        nisa.memset(dst=scores[:, full, :], value=0.0)
        nisa.dma_copy(dst=scores[0:tail, full, :], src=affinities_hbm.ap(
            pattern=[[experts, tail], [1, experts]], offset=full * p * experts))
    mask = nl.ndarray((p, tiles, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mask, data=scores, op0=nl.not_equal, operand0=0.0)
    earlier = nl.ndarray((p, tiles + 1, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=earlier[:, 0, :], value=0.0)
    for i in range(tiles):
        nisa.tensor_tensor(dst=earlier[:, i + 1, :], data1=earlier[:, i, :],
                           data2=mask[:, i, :], op=nl.add)

    # ---- Per expert: its token count, blocks and first emission row. -------- #
    # count[e] = sum over partitions of earlier[:, tiles, e], with e on the partitions.
    count_ps = nl.ndarray((experts, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=count_ps, stationary=earlier[:, tiles, :], moving=ones[:, 0:1],
                   is_moving_onezero=True, accumulate=False)
    count = _tile(experts, 1)
    nisa.tensor_copy(dst=count, src=count_ps)
    above = _tile(experts, thresholds)
    nisa.tensor_scalar(dst=above, data=threshold, op0=nl.less, operand0=count)
    blocks = _tile(experts, 1)
    nisa.tensor_reduce(dst=blocks, op=nl.add, data=above, axis=1)
    # blocks[e'] on every column, then first[t, e] = sum over e' < e, on every token.
    spread = _tile(experts, p)
    nisa.tensor_scalar(dst=spread, data=ones[0:experts, :], op0=nl.multiply, operand0=blocks)
    first_ps = nl.ndarray((p, experts), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=first_ps, stationary=spread, moving=before, is_moving_onezero=True,
                   accumulate=False)
    # base = order + first * rows - 1, so a key is (base + the inclusive token count) * mask.
    base = nl.ndarray((p, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base, data=first_ps, op0=nl.multiply, operand0=float(rows),
                       op1=nl.add, operand1=-1.0)
    nisa.tensor_tensor(dst=base, data1=base, data2=order, op=nl.add)

    # ---- Keys and each token's 8 largest, a PSUM bank of tiles at a time. ---- #
    top = nl.ndarray((p, tiles, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    for g0 in range(0, tiles, group):
        n = min(group, tiles - g0)
        counted_ps = nl.ndarray((p, n, experts), dtype=nl.float32, buffer=nl.psum)
        for i in range(g0, g0 + n):
            if i > 0:
                nisa.nc_matmul(dst=counted_ps[:, i - g0, :], stationary=ones,
                               moving=earlier[:, i, :], is_stationary_onezero=True,
                               accumulate=False)
            nisa.nc_matmul(dst=counted_ps[:, i - g0, :], stationary=lower, moving=mask[:, i, :],
                           is_stationary_onezero=True, is_moving_onezero=True,
                           accumulate=i > 0)
        key = nl.ndarray((p, n, experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=key, data1=counted_ps, data2=base.ap(
            pattern=[[experts, p], [0, n], [1, experts]]), op=nl.add)
        nisa.tensor_tensor(dst=key, data1=key, data2=mask[:, g0:g0 + n, :], op=nl.multiply)
        for i in range(g0, g0 + n):
            nisa.max8(dst=top[:, i, :], src=key[:, i - g0, :])

    # ---- valid = key != 0, index = key mod scale; out by the same tiling. ---- #
    valid = nl.ndarray((p, tiles, MAX8_WIDTH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=valid, data=top, op0=nl.not_equal, operand0=0.0)
    whole = nl.ndarray((p, tiles, MAX8_WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=whole, src=top)
    index = nl.ndarray((p, tiles, MAX8_WIDTH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=index, data=whole, op0=nl.bitwise_and, operand0=scale - 1)
    _store_by_tile(index_out, index, slots)
    _store_by_tile(valid_out, valid, slots)
    return index_out, valid_out


def combine_slots_torch(expert_affinities: Tensor, block: int, rows: int,
                        top_k: int) -> tuple[Tensor, Tensor]:
    """The slot build in torch: the reference, and the route where the kernel does not run."""
    from vllm_neuron.functional.moe.moe_blockwise import _cumsum_matmul

    experts = expert_affinities.shape[1]
    device = expert_affinities.device
    mask = (expert_affinities != 0).to(torch.float32)  # [T, E]
    # Token t's place among expert e's tokens, and the first block expert e owns.
    place = _cumsum_matmul(mask) - 1.0
    blocks = torch.ceil(mask.sum(dim=0) / block)
    first_block = _cumsum_matmul(blocks.unsqueeze(1)).squeeze(1) - blocks
    row = first_block.unsqueeze(0) * rows + place  # [T, E], exact in fp32
    # Token t's j-th selected expert, one-hot over E for each slot j.
    upper = torch.triu(
        torch.ones(experts, experts, dtype=torch.float32, device=device)
    )
    rank = torch.matmul(mask, upper) - 1.0  # [T, E]
    slots = min(top_k, experts)
    slot_ids = torch.arange(slots, dtype=torch.float32, device=device)
    pick = mask.unsqueeze(2) * (rank.unsqueeze(2) == slot_ids).to(torch.float32)
    valid = pick.sum(dim=1)  # [T, k], 0/1
    index = (pick * row.unsqueeze(2)).sum(dim=1).to(torch.int32)  # [T, k]
    return index, valid


def can_run_combine_slots(expert_affinities: Tensor, block: int, rows: int, top_k: int) -> bool:
    """A device that runs NKI, not a graph captured in CPU mode (whose simulator dispatch drops
    the int arguments), and the kernel's tile limits: ``8 <= E <= 128`` (``max8``'s least read,
    one partition tile), at most 8 slots (``max8``'s width), every key exact in fp32."""
    tokens, experts = (int(n) for n in expert_affinities.shape)
    return (
        can_run_kernel(expert_affinities)
        and not (envs.VLLM_NEURON_CPU_MODE and torch.compiler.is_compiling())
        and MAX8_WIDTH <= experts <= TOKEN_TILE
        and min(int(top_k), experts) <= MAX8_WIDTH
        and (experts + 1) * key_scale(tokens, experts, int(block), int(rows))
        <= FP32_EXACT_INTEGERS
    )


def combine_slots(expert_affinities: Tensor, block: int, rows: int,
                  top_k: int) -> tuple[Tensor, Tensor]:
    """``(index [T, K] int32, valid [T, K] fp32)``: each token's emission rows, ``K = min(top_k, E)``.

    Args:
        expert_affinities: ``[T, E]`` local router scores, zero where unselected.
        block: tokens per block in the mapping.
        rows: emission rows kept per block, ``min(T, block)``.
        top_k: experts each token selects.
    """
    if not can_run_combine_slots(expert_affinities, block, rows, top_k):
        return combine_slots_torch(expert_affinities, block, rows, top_k)
    tokens, experts = (int(n) for n in expert_affinities.shape)
    return wrap_nki(combine_slots_kernel)(
        expert_affinities.contiguous(), int(block), int(rows), min(int(top_k), experts),
        key_scale(tokens, experts, int(block), int(rows)))
