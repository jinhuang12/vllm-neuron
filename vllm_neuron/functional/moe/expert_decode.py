# SPDX-License-Identifier: Apache-2.0
"""Decode-time routed experts: ``T <= 64`` tokens, one launch, no mapping.

The 5938748 decode path routed through ``build_blockwise_mapping`` (XLA), padded
the hidden and affinity tensors, ran ``compact_decode_kernel`` (``q == 1``) or
the general kernel per 128-row block, and combined the per-block rows with an
fp32 token gather. That kernel applied the block-fp8 scale after every 128x128
product: 384 vector instructions per expert at I=512, H=4096, ~80 us of the
~125 us one hit cost. This kernel reads the router's global ``[T, E]`` output
and this rank's group id, and does the rest itself:

* **Distinct experts only.** The local affinity slice is summed over tokens and
  compacted with ``nonzero_with_count``; a dynamic loop visits each expert with
  at least one routed token once, whatever the number of tokens that chose it.
  At ``T <= TOKEN_AXIS_MAX`` every product streams all ``T`` tokens, and a token
  that did not choose the expert has affinity 0 there, so it adds exactly 0.
* **Token groups above that** (``_grouped_experts``). A work item is one expert
  and up to ``GROUP_TOKENS`` of its own tokens: a one-hot PE product gathers
  their ``x^T`` columns, the products stream ``GROUP_TOKENS`` columns instead of
  ``T``, and each real column adds into its token's accumulator column. The
  vector work per visited expert no longer grows with ``T``.
* **Scales in the epilogue of the products.** Each 128x128 product writes its own
  PSUM column range (``[128, slot, T]``, up to one 2 KiB bank per group); one
  ``tensor_tensor`` multiplies a whole bank by the broadcast block scales and one
  ``tensor_reduce`` sums the hidden blocks. The down projection folds the gate
  weight into its scale (``scale * affinity``). Per expert: 2 + 5 (SwiGLU) + 4
  vector instructions at ``T <= 4`` instead of 384 + 7.
* **Both LNC2 cores per expert.** Program ``c`` computes intermediate blocks
  ``[c * I/2, (c+1) * I/2)`` of every visited expert -- half the weight bytes and
  half the 128x128 weight loads each -- and the two fp32 partial sums meet once
  per layer in one ``sendrecv``; each program then stores its half of ``H``.
* **The combine is the accumulator.** Expert outputs add into one fp32
  ``[128, H/128, T]`` tile, in ascending local expert order; the result is
  transposed once and stored in the output dtype.

Numerics against 5938748: the same fp32 128x128 products, the same SwiGLU
instruction sequence and bf16 activation. Partial sums are associated
differently (hidden blocks in one reduce, two I halves, experts before the
cast), so results agree to fp32 round-off and, rarely, one bf16 activation step.
Against 0a08ff4 the grouped path takes the same products in the same order
(the gather and the gate broadcast are exact one-hot products).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.utils.neuron_utils import can_run_kernel

from .fused_fp8_pack import PackedExperts

#: Largest token count the decode kernel serves in one launch.
EXPERT_DECODE_MAX_TOKENS = 64

#: fp32 columns in one 2 KiB PSUM bank.
_BANK = 512

#: Largest token count that streams every token through each visited expert
#: (the 0a08ff4 schedule). Above it, each expert sees only its own tokens.
TOKEN_AXIS_MAX = 16

#: Tokens of one expert per grouped work item: the moving width of every
#: product, so the vector work per expert scales with this, not with T.
GROUP_TOKENS = 16

#: This file's content digest, a trace-time int default of the kernel. The
#: compile caches key on a kernel's own source and its arguments; the kernel's
#: body lives in the helpers below, so without it an edit to a helper would be
#: served the previous build.
_SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)


def decode_plan(tokens: int):
    """The kernel's schedule for ``tokens`` rows: ``(name, moving width)``.

    ``("token_axis", T)`` for ``T <= TOKEN_AXIS_MAX``: every visited expert
    streams all ``T`` tokens (0a08ff4). ``("grouped", GROUP_TOKENS)`` above:
    each visited expert streams only its own tokens, ``GROUP_TOKENS`` at a time.
    """
    if tokens <= TOKEN_AXIS_MAX:
        return "token_axis", tokens
    return "grouped", GROUP_TOKENS


def _tile(rows, cols, dtype=nl.float32):
    """An SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _weight_panel(weights, expert, panel, fp8):
    """One packed panel ``[128, H/128, 128]`` of expert ``expert`` (a register)."""
    nh = weights.shape[3]
    dtype = weights.dtype if fp8 else nl.bfloat16
    tile = nl.ndarray((128, nh, 128), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=tile.reshape((128, nh * 128)),
        src=weights.ap(
            pattern=[[nh * 128, 128], [1, nh * 128]],
            offset=panel * 128 * nh * 128,
            scalar_offset=expert, indirect_dim=0,
        ),
    )
    return tile


def _expert_operands(weights, scales, expert, i0, nic, fp8):
    """One expert's panels and broadcast scale rows for intermediate blocks
    ``[i0, i0 + nic)``: (gate/up stationaries, down stationaries, flat scales).

    ``flat_scale`` is ``[128, 3 * nic * H/128]``: gate, up, down rows of this
    program's blocks, the same row on every partition.
    """
    _, panels, _, nh, _ = weights.shape
    ni = panels // 3
    scale = nl.ndarray((128, 3, nic, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale.reshape((128, 3 * nic * nh)), src=scales.ap(
        pattern=[[0, 128], [ni * nh, 3], [1, nic * nh]], offset=i0 * nh,
        scalar_offset=expert, indirect_dim=0))
    stationaries = []
    for kind in range(2):
        for k in range(nic):
            stationaries.append(_weight_panel(weights, expert, kind * ni + i0 + k, fp8))
    down = []
    for k in range(nic):
        down.append(_weight_panel(weights, expert, 2 * ni + i0 + k, fp8))
    return stationaries, down, scale.reshape((128, 3 * nic * nh))


def _expert_contribution(moving, gate, stationaries, down, flat_scale, clamp,
                         nh, nic, columns):
    """``gate * expert(x)`` of ``columns`` token columns: ``[128, H/128, columns]``.

    ``moving`` is ``x^T`` as ``[128 (h mod 128), H/128, columns]`` bf16 and
    ``gate`` the ``[128, columns]`` fp32 gate weight of each column (0 for a
    column that did not choose the expert). The 0a08ff4 arithmetic: fp32
    128x128 products, block scales applied per PSUM bank, hidden blocks summed
    in one reduce, the clamped SwiGLU in bf16, the down projection's scale
    multiplied by the gate weight, intermediate blocks summed in one reduce.
    """
    gate_up_slots = 2 * nic * nh
    down_slots = nh * nic
    per_bank = max(1, _BANK // columns)  # product slots of `columns` fp32 per bank

    # Gate/up: slot s = panel * nh + hb, product [128 (I), columns] per slot.
    scaled = nl.ndarray((128, gate_up_slots, columns), dtype=nl.float32,
                        buffer=nl.sbuf)
    for s0 in range(0, gate_up_slots, per_bank):
        sn = min(per_bank, gate_up_slots - s0)
        bank = nl.ndarray((128, sn, columns), dtype=nl.float32, buffer=nl.psum)
        for j in range(sn):
            panel = (s0 + j) // nh
            hb = (s0 + j) % nh
            nisa.nc_matmul(dst=bank[:, j, :], stationary=stationaries[panel][:, hb, :],
                           moving=moving[:, hb, :], accumulate=False)
        nisa.tensor_tensor(
            dst=scaled[:, s0:s0 + sn, :], data1=bank,
            data2=flat_scale.ap(pattern=[[3 * nic * nh, 128], [1, sn], [0, columns]],
                                offset=s0),
            op=nl.multiply)
    pre = nl.ndarray((128, 2 * nic, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pre, data=scaled.ap(pattern=[
        [gate_up_slots * columns, 128], [nh * columns, 2 * nic], [1, columns],
        [columns, nh]]), op=nl.add, axis=(3,))

    # SwiGLU, the 5938748 instruction sequence: clamp, sigmoid, two products.
    g = nl.ndarray((128, nic, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=pre[:, 0:nic, :], op0=nl.minimum,
                       operand0=clamp[:, 0:1])
    u = nl.ndarray((128, nic, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=pre[:, nic:2 * nic, :], op0=nl.maximum,
                       operand0=clamp[:, 1:2], op1=nl.minimum,
                       operand1=clamp[:, 2:3])
    sig = nl.ndarray((128, nic, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sig, op=nl.sigmoid, data=g)
    silu = nl.ndarray((128, nic, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=silu, data1=g, data2=sig, op=nl.multiply)
    act = nl.ndarray((128, nic, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=silu, data2=u, op=nl.multiply)

    # Down: slot s = hb * nic + k, product [128 (H), columns]; scale * gate weight.
    weight = nl.ndarray((128, nh, nic, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=weight,
        data1=flat_scale.ap(pattern=[[3 * nic * nh, 128], [1, nh], [nh, nic],
                                     [0, columns]], offset=2 * nic * nh),
        data2=gate.ap(pattern=[[columns, 128], [0, nh], [0, nic], [1, columns]]),
        op=nl.multiply)
    flat_weight = weight.reshape((128, down_slots, columns))
    terms = nl.ndarray((128, down_slots, columns), dtype=nl.float32, buffer=nl.sbuf)
    for s0 in range(0, down_slots, per_bank):
        sn = min(per_bank, down_slots - s0)
        bank = nl.ndarray((128, sn, columns), dtype=nl.float32, buffer=nl.psum)
        for j in range(sn):
            hb = (s0 + j) // nic
            k = (s0 + j) % nic
            nisa.nc_matmul(dst=bank[:, j, :], stationary=down[k][:, hb, :],
                           moving=act[:, k, :], accumulate=False)
        nisa.tensor_tensor(dst=terms[:, s0:s0 + sn, :], data1=bank,
                           data2=flat_weight[:, s0:s0 + sn, :], op=nl.multiply)
    contribution = nl.ndarray((128, nh, columns), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=contribution, data=terms.ap(pattern=[
        [down_slots * columns, 128], [nic * columns, nh], [1, columns],
        [columns, nic]]), op=nl.add, axis=(3,))
    return contribution


def _token_axis_experts(hidden, local, weights, scales, clamp, acc, i0, nic, fp8):
    """``T <= TOKEN_AXIS_MAX``: every token through each visited expert (0a08ff4).

    Adds, in ascending local expert order, each visited expert's contribution
    for all ``T`` tokens into ``acc`` ``[128, H/128, T]``; a token that did not
    choose the expert has gate weight 0 there and adds exactly 0.
    """
    tokens, hidden_size = hidden.shape
    _, _, experts = local.shape
    nh = weights.shape[3]
    load = _tile(1, experts)
    nisa.tensor_reduce(dst=load, data=local.ap(
        pattern=[[tokens * experts, 1], [1, experts], [experts, tokens]]),
        op=nl.add, axis=(2,))
    visit = _tile(1, experts + 1, nl.int32)
    nisa.nonzero_with_count(dst=visit, src=load, index_offset=0, padding_val=0)
    count = nisa.register_alloc()
    nisa.register_load(dst=count, src=visit[:, experts:experts + 1])
    # Every partition needs the gate weights: broadcast by a K=1 ones product
    # (1.0 * a with no accumulation is exact in the fp32 PE path).
    ones = _tile(1, 128)
    nisa.memset(dst=ones, value=1.0)
    gates = nl.ndarray((128, tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    flat_local = local.reshape((1, tokens * experts))
    flat_gates = gates.reshape((128, tokens * experts))
    for c0 in range(0, tokens * experts, _BANK):
        cn = min(_BANK, tokens * experts - c0)
        spread = nl.ndarray((128, cn), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=spread, stationary=ones, moving=flat_local[:, c0:c0 + cn])
        nisa.tensor_copy(dst=flat_gates[:, c0:c0 + cn], src=spread)

    # x^T as [128 (h mod 128), H/128, T]: rows of 128 hidden values, transposed.
    xt = nl.ndarray((128, nh, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
    per_chunk = max(1, 128 // nh)
    for t0 in range(0, tokens, per_chunk):
        nt = min(per_chunk, tokens - t0)
        rows = nl.ndarray((nt * nh, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=hidden.ap(
            pattern=[[128, nt * nh], [1, 128]], offset=t0 * hidden_size))
        flipped = nl.ndarray((128, nt * nh), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=flipped, data=rows)
        nisa.tensor_copy(
            dst=xt.ap(pattern=[[nh * tokens, 128], [1, nt], [tokens, nh]], offset=t0),
            src=flipped.reshape((128, nt, nh)))

    def visit_expert(slot):
        # uint32: the hardware's dynamic SBUF reads take an unsigned offset.
        expert_sb = _tile(1, 1, nl.uint32)
        nisa.tensor_copy(dst=expert_sb, src=visit.ap(
            pattern=[[max(experts + 1, 8), 1], [1, 1]],
            scalar_offset=slot, indirect_dim=1))
        expert = nisa.register_alloc()
        nisa.register_load(dst=expert, src=expert_sb)
        stationaries, down, flat_scale = _expert_operands(
            weights, scales, expert, i0, nic, fp8)
        gate = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gate, src=gates.ap(
            pattern=[[tokens * experts, 128], [experts, tokens]],
            scalar_offset=expert, indirect_dim=2))
        contribution = _expert_contribution(xt, gate, stationaries, down, flat_scale,
                                            clamp, nh, nic, tokens)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=contribution, op=nl.add)

    nl.fori_loop(0, count, visit_expert)


def _grouped_experts(hidden, affinity, rank_reg, local, weights, scales, clamp, acc,
                     i0, nic, fp8):
    """``T > TOKEN_AXIS_MAX``: each visited expert sees only the tokens that chose it.

    Work items are ``(expert, group)`` pairs, ``GROUP_TOKENS`` tokens of one
    expert each, visited in ascending expert order (an expert with more tokens
    than one group gets several items). Per item the token columns are gathered
    exactly on the PE (a one-hot ``[T, G]`` selection times the token-major
    hidden rows), the expert runs on ``G`` columns with the token-axis path's
    arithmetic, and each real column is added into its token's ``acc`` column.
    The vector work per expert scales with ``G``, not with ``T``.
    """
    tokens, hidden_size = hidden.shape
    _, groups, experts = affinity.shape
    nh = weights.shape[3]
    group = GROUP_TOKENS
    need = (tokens + group - 1) // group
    slots = 1
    shift = 0
    while slots < need:
        slots = slots * 2
        shift = shift + 1
    width = slots * group  # token columns of one expert's padded mask row
    items = experts * slots
    wide = max(experts, 8)

    # Token-major hidden rows (the gather's stationaries) and this group's
    # affinities with the token on the partition axis.
    x_rows = nl.ndarray((tokens, hidden_size), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_rows, src=hidden)
    aff_t = nl.ndarray((tokens, wide), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aff_t[:, 0:experts], src=affinity.ap(
        pattern=[[groups * experts, tokens], [1, experts]],
        scalar_offset=rank_reg, indirect_dim=1))

    # Expert-major token mask [1, E, width], zero past T; tokens per expert.
    mask = nl.ndarray((1, experts, width), dtype=nl.float32, buffer=nl.sbuf)
    if width > tokens:
        nisa.memset(dst=mask, value=0.0)
    nisa.tensor_scalar(dst=mask[:, :, 0:tokens], data=local.ap(
        pattern=[[tokens * experts, 1], [1, experts], [experts, tokens]]),
        op0=nl.not_equal, operand0=0.0)
    counts = nl.ndarray((1, wide), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=counts[:, 0:experts], data=mask, op=nl.add, axis=(2,))
    # Item (e, g) exists when expert e has more than g * G tokens; index e * slots + g.
    wanted = nl.ndarray((1, experts, slots), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(slots):
        nisa.tensor_scalar(
            dst=wanted.ap(pattern=[[items, 1], [slots, experts]], offset=g),
            data=counts[:, 0:experts], op0=nl.greater, operand0=float(g * group))
    work = _tile(1, items + 1, nl.int32)
    nisa.nonzero_with_count(dst=work, src=wanted.reshape((1, items)),
                            index_offset=0, padding_val=0)
    count = nisa.register_alloc()
    nisa.register_load(dst=count, src=work[:, items:items + 1])
    item_expert = nl.ndarray((1, max(items, 8)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=item_expert[:, 0:items], data=work[:, 0:items],
                       op0=nl.right_shift, operand0=shift)
    item_group = nl.ndarray((1, max(items, 8)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=item_group[:, 0:items], data=work[:, 0:items],
                       op0=nl.bitwise_and, operand0=slots - 1)

    # Constants: K=1 ones (row broadcast), K=T ones (gate broadcast), positions.
    ones_row = _tile(1, tokens)
    nisa.memset(dst=ones_row, value=1.0)
    ones_col = nl.ndarray((tokens, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)
    position = _tile(tokens, 1)
    nisa.iota(dst=position, pattern=[[0, 1]], offset=0, channel_multiplier=1)

    def visit_item(slot):
        expert_sb = _tile(1, 1, nl.uint32)
        nisa.tensor_copy(dst=expert_sb, src=item_expert.ap(
            pattern=[[max(items, 8), 1], [1, 1]], scalar_offset=slot, indirect_dim=1))
        expert = nisa.register_alloc()
        nisa.register_load(dst=expert, src=expert_sb)
        group_sb = _tile(1, 1, nl.uint32)
        nisa.tensor_copy(dst=group_sb, src=item_group.ap(
            pattern=[[max(items, 8), 1], [1, 1]], scalar_offset=slot, indirect_dim=1))
        which = nisa.register_alloc()
        nisa.register_load(dst=which, src=group_sb)
        stationaries, down, flat_scale = _expert_operands(
            weights, scales, expert, i0, nic, fp8)

        # This expert's token positions, ascending, padded with T (no token).
        row = _tile(1, width)
        nisa.tensor_copy(dst=row, src=mask.ap(
            pattern=[[experts * width, 1], [1, width]], scalar_offset=expert,
            indirect_dim=1))
        positions = nl.ndarray((1, slots + 1, group), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nonzero_with_count(
            dst=positions.reshape((1, (slots + 1) * group))[:, 0:width + 1], src=row,
            index_offset=0, padding_val=tokens)
        picked = _tile(1, group, nl.int32)
        nisa.tensor_copy(dst=picked, src=positions.ap(
            pattern=[[(slots + 1) * group, 1], [1, group]], scalar_offset=which,
            indirect_dim=1))
        # Real columns of this item: min(count_e - g * G, G).
        held = _tile(1, 1)
        nisa.tensor_copy(dst=held, src=counts.ap(
            pattern=[[wide, 1], [1, 1]], scalar_offset=expert, indirect_dim=1))
        first = _tile(1, 1)
        nisa.tensor_copy(dst=first, src=group_sb)
        left = _tile(1, 1)
        nisa.tensor_scalar(dst=left, data=first, op0=nl.multiply,
                           operand0=-float(group), op1=nl.add, operand1=held)
        live_sb = _tile(1, 1, nl.uint32)
        nisa.tensor_scalar(dst=live_sb, data=left, op0=nl.minimum,
                           operand0=float(group))
        live = nisa.register_alloc()
        nisa.register_load(dst=live, src=live_sb)

        # One-hot selection [T, G]: column j is 1 at token picked[j] (no row for T).
        picked_f = _tile(1, group)
        nisa.tensor_copy(dst=picked_f, src=picked)
        spread = nl.ndarray((tokens, group), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=spread, stationary=ones_row, moving=picked_f)
        select = nl.ndarray((tokens, group), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=select, data=spread, op0=nl.equal, operand0=position)
        # The selected tokens' gate weights on every partition, [128, G] (one
        # nonzero term per column: exact in the fp32 PE path).
        column_aff = _tile(tokens, 1)
        nisa.tensor_copy(dst=column_aff, src=aff_t.ap(
            pattern=[[wide, tokens], [1, 1]], scalar_offset=expert, indirect_dim=1))
        weighted = nl.ndarray((tokens, group), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=weighted, data=select, op0=nl.multiply,
                           operand0=column_aff)
        spread_gate = nl.ndarray((128, group), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=spread_gate, stationary=ones_col, moving=weighted)
        gate = nl.ndarray((128, group), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gate, src=spread_gate)
        # x^T of the selected tokens, [128 (h mod 128), H/128, G]: exact copies.
        gathered = nl.ndarray((128, nh, group), dtype=nl.float32, buffer=nl.psum)
        for hb in range(nh):
            nisa.nc_matmul(dst=gathered[:, hb, :],
                           stationary=x_rows[:, hb * 128:(hb + 1) * 128],
                           moving=select, accumulate=False)
        moving = nl.ndarray((128, nh, group), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=moving, src=gathered)

        contribution = _expert_contribution(moving, gate, stationaries, down,
                                            flat_scale, clamp, nh, nic, group)

        def scatter(j):
            at_sb = _tile(1, 1, nl.uint32)
            nisa.tensor_copy(dst=at_sb, src=picked.ap(
                pattern=[[max(group, 8), 1], [1, 1]], scalar_offset=j, indirect_dim=1))
            at = nisa.register_alloc()
            nisa.register_load(dst=at, src=at_sb)
            column = acc.ap(pattern=[[nh * tokens, 128], [tokens, nh], [1, 1]],
                            scalar_offset=at, indirect_dim=2)
            nisa.tensor_tensor(dst=column, data1=column, data2=contribution.ap(
                pattern=[[nh * group, 128], [group, nh], [1, 1]], scalar_offset=j,
                indirect_dim=2), op=nl.add)

        nl.fori_loop(0, live, scatter)

    nl.fori_loop(0, count, visit_item)


@nki.jit
def expert_decode_kernel(hidden, affinity, rank, weights, scales, bounds,
                         OUT_FP32: bool = False, WEIGHT_FP8: bool = False,
                         SOURCE_DIGEST: int = _SOURCE_DIGEST):
    """Routed-expert output of ``T`` decode tokens on one rank's expert shard.

    Args:
        hidden: ``[T, H]`` bf16 expert inputs, ``1 <= T <= 64``.
        affinity: ``[T, G, E]`` fp32, the router's scattered global output viewed
            as ``G`` expert-parallel groups of this bank's ``E`` experts.
        rank: ``[1, 1]`` int32, the group this bank holds.
        weights: ``[E, 3 * I/128, 128, H/128, 128]`` packed fp8 bank.
        scales: ``[E, 3 * I/128, H/128]`` fp32 block scales.
        bounds: ``[128, 3]`` fp32 SwiGLU bounds (gate upper, up lower, up upper).
        OUT_FP32: store fp32 instead of bf16.
        WEIGHT_FP8: feed the fp8 tiles to the PE as stationaries instead of
            casting them to bf16 in the DMA. The products are identical (an e4m3
            by bf16 product is exact in fp32); only the SBUF traffic differs.
        SOURCE_DIGEST: this file's digest; it only keys the compile caches.

    Returns:
        ``[T, H]`` -- the sum over this bank's experts of ``affinity * expert(x)``.
    """
    tokens, hidden_size = hidden.shape
    _, groups, experts = affinity.shape
    _, panels, _, nh, _ = weights.shape
    ni = panels // 3
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(1 <= tokens <= EXPERT_DECODE_MAX_TOKENS, "1 <= T <= 64")
    kernel_assert(nh * 128 == hidden_size and nh <= 128, "packed H mismatch")
    kernel_assert(weights.shape[0] == experts, "affinity groups mismatch the bank")
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(ni % programs == 0 and nh % programs == 0,
                  "I/128 and H/128 split evenly over the programs")
    nic = ni // programs  # intermediate blocks of this program
    i0 = program * nic
    own = nh // programs  # hidden blocks this program stores
    h0 = program * own
    out_dtype = nl.float32 if OUT_FP32 else nl.bfloat16
    out = nl.ndarray((tokens, hidden_size), dtype=out_dtype, buffer=nl.shared_hbm)

    # ---- Prologue: rank, local affinities. ----------------------------------- #
    rank_sb = _tile(1, 1, nl.int32)
    nisa.dma_copy(dst=rank_sb, src=rank)
    # Clamp to [0, G): a rank operand outside the router's groups must not move
    # the affinity read out of bounds. A valid rank is unchanged; the call site
    # refuses an out-of-range python-int rank before it gets here.
    group = _tile(1, 1, nl.int32)
    nisa.tensor_scalar(dst=group, data=rank_sb, op0=nl.maximum, operand0=0,
                       op1=nl.minimum, operand1=groups - 1)
    rank_reg = nisa.register_alloc()
    nisa.register_load(dst=rank_reg, src=group)
    # Partition 0: this group's [T, E] slice, token-major.
    local = nl.ndarray((1, tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=local, src=affinity.ap(
        pattern=[[0, 1], [groups * experts, tokens], [1, experts]],
        scalar_offset=rank_reg, indirect_dim=1))
    clamp = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=clamp, src=bounds)
    acc = nl.ndarray((128, nh, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)

    if tokens <= TOKEN_AXIS_MAX:
        _token_axis_experts(hidden, local, weights, scales, clamp, acc, i0, nic,
                            WEIGHT_FP8)
    else:
        _grouped_experts(hidden, affinity, rank_reg, local, weights, scales, clamp,
                         acc, i0, nic, WEIGHT_FP8)

    # ---- Epilogue: meet the other I half, store this program's H half. ----- #
    if programs == 2:
        other = 1 - program
        theirs = nl.ndarray((128, own, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=acc[:, other * own:(other + 1) * own, :], dst=theirs,
                      send_to_rank=other, recv_from_rank=other, pipe_id=0)
        total = nl.ndarray((128, own, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=total, data1=acc[:, h0:h0 + own, :], data2=theirs,
                           op=nl.add)
    else:
        total = acc
    rows_out = nl.ndarray((tokens, own * 128), dtype=out_dtype, buffer=nl.sbuf)
    per_bank = _BANK // 128
    for b0 in range(0, own, per_bank):
        bn = min(per_bank, own - b0)
        flipped = nl.ndarray((tokens, bn * 128), dtype=nl.float32, buffer=nl.psum)
        for j in range(bn):
            nisa.nc_transpose(dst=flipped[:, j * 128:(j + 1) * 128],
                              data=total[:, b0 + j, :])
        nisa.tensor_copy(dst=rows_out[:, b0 * 128:(b0 + bn) * 128], src=flipped)
    nisa.dma_copy(dst=out.ap(pattern=[[hidden_size, tokens], [1, own * 128]],
                             offset=h0 * 128), src=rows_out)
    return out


# ---------------------------------------------------------------------------- #
# Admission, operands and the torch reference. The public entry point is
# ``fused_fp8.fused_fp8_decode_experts``, which counts into the fused seam.
# ---------------------------------------------------------------------------- #


def geometry(packed: PackedExperts):
    """``(experts, I/128, H/128)`` of a packed bank."""
    experts, panels, contraction, nh, channels = packed.weights.shape
    return experts, panels // 3, nh


def can_run_expert_decode(hidden_states: Tensor, expert_affinities: Tensor,
                          packed: PackedExperts) -> bool:
    """True when the decode kernel serves this call on an NKI device or simulator."""
    if packed.weights.ndim != 5 or hidden_states.ndim != 2:
        return False
    experts, ni, nh = geometry(packed)
    tokens, hidden = hidden_states.shape
    routed = expert_affinities.shape[-1]
    return (
        1 <= tokens <= EXPERT_DECODE_MAX_TOKENS
        and hidden == nh * 128
        and hidden_states.dtype == torch.bfloat16
        and tuple(packed.weights.shape[1:]) == (3 * ni, 128, nh, 128)
        and packed.weights.dtype == torch.float8_e4m3fn
        and tuple(packed.scales.shape) == (experts, 3 * ni, nh)
        and packed.scales.dtype == torch.float32
        and tuple(expert_affinities.shape) == (tokens, routed)
        and routed % experts == 0
        and nh <= 128
        and can_run_kernel(hidden_states)
    )


def default_programs(ni: int, nh: int) -> int:
    """2 programs (both LNC2 cores) when the runtime is LNC2 and I, H split evenly."""
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and ni % 2 == 0 and nh % 2 == 0:
        return 2
    return 1


def rank_operand(rank, device) -> Tensor:
    """The EP group as the kernel's ``[1, 1]`` int32 operand."""
    if isinstance(rank, Tensor):
        return rank.reshape(1, 1).to(device=device, dtype=torch.int32)
    return torch.full((1, 1), int(rank), dtype=torch.int32, device=device)


def expert_decode_torch_oracle(hidden_states, expert_affinities, packed, bounds,
                               expert_parallel_rank, out_dtype=None):
    """Torch reference of the kernel's arithmetic, for tests and benchmark checks."""
    from .fused_fp8_pack import unpack_experts

    out_dtype = hidden_states.dtype if out_dtype is None else out_dtype
    experts, ni, nh = geometry(packed)
    tokens, hidden = hidden_states.shape
    rank = int(expert_parallel_rank.reshape(-1)[0]) if isinstance(
        expert_parallel_rank, Tensor) else int(expert_parallel_rank)
    local = expert_affinities.reshape(tokens, -1, experts)[:, rank, :].to(torch.float32)
    gate_up, down, gate_up_scale, down_scale = unpack_experts(packed)
    x = hidden_states.to(torch.float32).reshape(tokens, nh, 128)
    clamp = bounds[0].to(torch.float32)
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=hidden_states.device)
    for e in range(experts):
        if not bool((local[:, e] != 0).any()):
            continue
        w = gate_up[e].to(torch.float32).reshape(nh, 128, 2 * ni, 128)
        partial = torch.einsum("thc,hcpo->tpho", x, w)  # [T, panel, hb, 128]
        s = gate_up_scale[e].reshape(nh, 2 * ni).t()  # [panel, hb]
        pre = (partial * s[None, :, :, None]).sum(dim=2).reshape(tokens, 2 * ni * 128)
        g = pre[:, : ni * 128].clamp(max=float(clamp[0]))
        u = pre[:, ni * 128:].clamp(min=float(clamp[1]), max=float(clamp[2]))
        act = (g * torch.sigmoid(g) * u).to(torch.bfloat16).to(torch.float32)
        wd = down[e].to(torch.float32).reshape(ni, 128, nh, 128)
        dpart = torch.einsum("tkc,kcho->tkho", act.reshape(tokens, ni, 128), wd)
        weight = down_scale[e][None, :, :, None] * local[:, e][:, None, None, None]
        out += (dpart * weight).sum(dim=1).reshape(tokens, hidden)
    return out.to(out_dtype)
