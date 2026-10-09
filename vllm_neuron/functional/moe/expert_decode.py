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
  ``T``, and a second one-hot PE product adds each column into its token's
  accumulator column. The item tables are built once, before the item loop,
  so an item reads them at the loop register and needs no register of its
  own except the next item's expert (for its weight DMA).
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
(the gather, the gate broadcast and the scatter are exact one-hot products).
One difference: a NaN or Inf in one token's hidden row or expert output
reaches the other tokens of its items through those products (``0 * NaN``);
0a08ff4 keeps it in that token's row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
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
GROUP_SHIFT = GROUP_TOKENS.bit_length() - 1
assert 1 << GROUP_SHIFT == GROUP_TOKENS, "GROUP_TOKENS is a power of two"

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


def _operand_tiles(weights, nic, fp8):
    """Tiles for one expert's operands (see ``_load_operands``), allocated once."""
    nh = weights.shape[3]
    dtype = weights.dtype if fp8 else nl.bfloat16
    stationaries = []
    for _ in range(2 * nic):
        stationaries.append(nl.ndarray((128, nh, 128), dtype=dtype, buffer=nl.sbuf))
    down = []
    for _ in range(nic):
        down.append(nl.ndarray((128, nh, 128), dtype=dtype, buffer=nl.sbuf))
    scale = nl.ndarray((128, 3, nic, nh), dtype=nl.float32, buffer=nl.sbuf)
    return stationaries, down, scale


def _load_operands(weights, scales, expert, i0, nic, tiles):
    """``_expert_operands`` into preallocated ``tiles`` (the grouped path's
    double buffer): the same DMAs, so the same operands."""
    _, panels, _, nh, _ = weights.shape
    ni = panels // 3
    stationaries, down, scale = tiles
    nisa.dma_copy(dst=scale.reshape((128, 3 * nic * nh)), src=scales.ap(
        pattern=[[0, 128], [ni * nh, 3], [1, nic * nh]], offset=i0 * nh,
        scalar_offset=expert, indirect_dim=0))
    for slot in range(3 * nic):
        tile = stationaries[slot] if slot < 2 * nic else down[slot - 2 * nic]
        kind = slot // nic
        panel = kind * ni + i0 + slot % nic
        nisa.dma_copy(
            dst=tile.reshape((128, nh * 128)),
            src=weights.ap(pattern=[[nh * 128, 128], [1, nh * 128]],
                           offset=panel * 128 * nh * 128,
                           scalar_offset=expert, indirect_dim=0))
    return stationaries, down, scale.reshape((128, 3 * nic * nh))


def _expert_contribution(moving, gate, stationaries, down, flat_scale, clamp,
                         nh, nic, columns, out=None):
    """``gate * expert(x)`` of ``columns`` token columns: ``[128, H/128, columns]``.

    ``moving`` is ``x^T`` as ``[128 (h mod 128), H/128, columns]`` bf16 and
    ``gate`` the ``[128, columns]`` fp32 gate weight of each column (0 for a
    column that did not choose the expert). The 0a08ff4 arithmetic: fp32
    128x128 products, block scales applied per PSUM bank, hidden blocks summed
    in one reduce, the clamped SwiGLU in bf16, the down projection's scale
    multiplied by the gate weight, intermediate blocks summed in one reduce.
    ``out``, if given, is the ``[128, H/128, columns]`` fp32 tile to write.
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
    contribution = out
    if contribution is None:
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


def _pair_entry(table, pairs, p, held_sb):
    """``table[p]`` (``p`` a register) in a register, through the ``[1, 1]``
    uint32 tile ``held_sb``."""
    nisa.tensor_copy(dst=held_sb, src=table.ap(
        pattern=[[pairs, 1], [1, 1]], scalar_offset=p, indirect_dim=1))
    value = nisa.register_alloc()
    nisa.register_load(dst=value, src=held_sb)
    return value, held_sb


def _grouped_load(weights, scales, table, pairs, p, i0, nic, tiles, held_sb):
    """DMA the operands of the item at ``table[p]`` into ``tiles``.

    ``held_sb`` is this load's own ``[1, 1]`` tile, allocated before the loop:
    the DMA trigger engine rereads it while the DMAs issue, so a tile shared
    with the other load of the pair would stall that load's write until
    these DMAs have issued.
    """
    expert, _ = _pair_entry(table, pairs, p, held_sb)
    _load_operands(weights, scales, expert, i0, nic, tiles)


def _grouped_compute(state, tables, p, operands):
    """One grouped work item from loaded ``operands``: gather its tokens, run the
    expert on ``GROUP_TOKENS`` columns, add each column into its token's ``acc``
    column with one-hot PE products.

    ``tables`` are the item's selection and gate tables of this pair position
    (see ``_grouped_tables``); ``p`` (the loop register) is the only dynamic
    offset, and only ``tensor_copy`` reads at it.
    """
    x_rows, ident, blockmask, acc, clamp, geometry = state
    tokens, nh, nic, pairs, hb_blk = geometry
    select_table, gate_table = tables
    group = GROUP_TOKENS
    rows = hb_blk * group
    blocks = nh // hb_blk
    stationaries, down, flat_scale = operands

    # The item's one-hot selection [T, G]: column j is its token of rank
    # first + j among the expert's tokens (zero past the last one), and the
    # selected tokens' gate weights on every partition [128, G].
    select = nl.ndarray((tokens, group), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=select, src=select_table.ap(
        pattern=[[pairs * group, tokens], [1, group]], scalar_offset=p, indirect_dim=1))
    gate = nl.ndarray((128, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gate, src=gate_table.ap(
        pattern=[[pairs * group, 128], [1, group]], scalar_offset=p, indirect_dim=1))
    # x^T of the selected tokens, [128 (h mod 128), H/128, G]: exact copies.
    gathered = nl.ndarray((128, nh, group), dtype=nl.float32, buffer=nl.psum)
    for hb in range(nh):
        nisa.nc_matmul(dst=gathered[:, hb, :],
                       stationary=x_rows[:, hb * 128:(hb + 1) * 128],
                       moving=select, accumulate=False)
    moving = nl.ndarray((128, nh, group), dtype=nl.bfloat16, buffer=nl.sbuf)
    # The plain copies of the item run on the Scalar engine, which is otherwise
    # idle; the values are bf16-exact, so the cast is exact on either engine.
    nisa.tensor_copy(dst=moving, src=gathered, engine=nisa.engine.scalar)

    contribution = _expert_contribution(moving, gate, stationaries, down, flat_scale,
                                        clamp, nh, nic, group)

    # The scatter matrix [(b, j), (b', t)] = (b == b') * select[t, j] for one
    # block of hb_blk hidden blocks: select^T on each partition block (an exact
    # one-hot product), masked to the block diagonal.
    tiled = nl.ndarray((tokens, hb_blk, group), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tiled, src=select.ap(
        pattern=[[group, tokens], [0, hb_blk], [1, group]]), engine=nisa.engine.scalar)
    spread = nl.ndarray((rows, tokens), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=spread, stationary=tiled.reshape((tokens, rows)), moving=ident)
    scatter = nl.ndarray((rows, hb_blk, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scatter, data1=spread.ap(
        pattern=[[tokens, rows], [0, hb_blk], [1, tokens]]), data2=blockmask,
        op=nl.multiply)
    # Per block: the contribution transposed to [(b, j), h] (a bit-accurate PE
    # transpose), then acc[h, b', t] += sum_(b, j) C^T[(b, j), h] * scatter[(b, j),
    # (b', t)] -- one nonzero term per token column of the item, 0 elsewhere,
    # so each token's acc column gets exactly its expert's column added, in
    # ascending expert order, as in 0a08ff4.
    per_bank = max(1, _BANK // 128)
    for c0 in range(0, blocks, per_bank):
        cn = min(per_bank, blocks - c0)
        flipped = nl.ndarray((rows, cn, 128), dtype=nl.float32, buffer=nl.psum)
        for c in range(cn):
            nisa.nc_transpose(dst=flipped[:, c, :], data=contribution.ap(
                pattern=[[nh * group, 128], [1, rows]], offset=(c0 + c) * rows))
        held = nl.ndarray((rows, cn, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=held, src=flipped, engine=nisa.engine.scalar)
        for c in range(cn):
            h0 = (c0 + c) * hb_blk
            added = nl.ndarray((128, hb_blk * tokens), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=added, stationary=held[:, c, :],
                           moving=scatter.reshape((rows, hb_blk * tokens)),
                           is_moving_onezero=True)
            nisa.tensor_tensor(dst=acc[:, h0:h0 + hb_blk, :],
                               data1=acc[:, h0:h0 + hb_blk, :],
                               data2=added.reshape((128, hb_blk, tokens)), op=nl.add)


def _grouped_tables(aff_t, ranks, by_pair, constants, geometry):
    """Each item's selection and gate weights, in the loop's pair layout.

    Item ``i`` (expert ``e_i``, first column ``f_i``): ``select[t, j] = 1`` when
    token ``t`` has rank ``f_i + j`` among ``e_i``'s tokens, and ``gate[:, j]`` is
    that token's affinity (0 for a column with no token). Returns, for pair
    position ``q`` in (0, 1), ``(select [T, pairs, G] bf16, gate [128, pairs, G]
    fp32)`` of items ``2p + q``. All products are one-hot (exact).
    """
    tokens, experts, pairs = geometry
    ones_e, expert_id, column_id, ones_col = constants
    group = GROUP_TOKENS
    both = 2 * pairs
    # Item expert ids and first columns as fp32 rows [1, (q, p)].
    item_expert = nl.ndarray((1, both), dtype=nl.float32, buffer=nl.sbuf)
    item_first = nl.ndarray((1, both), dtype=nl.float32, buffer=nl.sbuf)
    for q in range(2):
        nisa.tensor_copy(dst=item_expert[:, q * pairs:(q + 1) * pairs], src=by_pair[q][0])
        nisa.tensor_copy(dst=item_first[:, q * pairs:(q + 1) * pairs], src=by_pair[q][1])
    # One-hot [E, items]: onehot[e, i] = (e_i == e). The ids reach every
    # partition through a K=1 ones product.
    spread_e = nl.ndarray((experts, both), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=spread_e, stationary=ones_e[:, 0:experts], moving=item_expert,
                   is_stationary_onezero=True)
    onehot = nl.ndarray((experts, both), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=onehot, data=spread_e, op0=nl.equal, operand0=expert_id)
    # Affinities and ranks with the expert on the partition axis (bit-accurate
    # PE transposes), then each item's column of them [T, items].
    flipped = nl.ndarray((experts, 2, tokens), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=flipped[:, 0, :], data=aff_t[:, 0:experts])
    nisa.nc_transpose(dst=flipped[:, 1, :], data=ranks[:, 0:experts])
    by_expert = nl.ndarray((experts, 2, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=by_expert, src=flipped)
    item_aff = nl.ndarray((tokens, both), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=item_aff, stationary=by_expert[:, 0, :], moving=onehot,
                   is_moving_onezero=True)
    item_rank = nl.ndarray((tokens, both), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=item_rank, stationary=by_expert[:, 1, :], moving=onehot,
                   is_moving_onezero=True)
    first_t = nl.ndarray((tokens, both), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=first_t, stationary=ones_e[:, 0:tokens], moving=item_first,
                   is_stationary_onezero=True)
    # wanted[t, i, j] = f_i + j; select = (rank of t in e_i == wanted).
    wanted = nl.ndarray((tokens, both, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wanted, data1=column_id, data2=first_t.ap(
        pattern=[[both, tokens], [1, both], [0, group]]), op=nl.add)
    select = nl.ndarray((tokens, both, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=select, data1=wanted, data2=item_rank.ap(
        pattern=[[both, tokens], [1, both], [0, group]]), op=nl.equal)
    weighted = nl.ndarray((tokens, both, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=weighted, data1=select, data2=item_aff.ap(
        pattern=[[both, tokens], [1, both], [0, group]]), op=nl.multiply)
    out = []
    for q in range(2):
        select_q = nl.ndarray((tokens, pairs, group), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=select_q, src=select[:, q * pairs:(q + 1) * pairs, :])
        gate_q = nl.ndarray((128, pairs, group), dtype=nl.float32, buffer=nl.sbuf)
        flat_gate = gate_q.reshape((128, pairs * group))
        for c0 in range(0, pairs * group, _BANK):
            cn = min(_BANK, pairs * group - c0)
            spread = nl.ndarray((128, cn), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=spread, stationary=ones_col, moving=weighted.ap(
                pattern=[[both * group, tokens], [1, cn]], offset=q * pairs * group + c0),
                is_stationary_onezero=True)
            nisa.tensor_copy(dst=flat_gate[:, c0:c0 + cn], src=spread)
        out.append((select_q, gate_q))
    return out


def _grouped_experts(hidden, affinity, rank_reg, local, weights, scales, clamp, acc,
                     i0, nic, fp8):
    """``T > TOKEN_AXIS_MAX``: each visited expert sees only the tokens that chose it.

    Work items are ``(expert, group)`` pairs, ``GROUP_TOKENS`` tokens of one
    expert each, visited in ascending expert order (an expert with more tokens
    than one group gets several items). Per item the token columns are gathered
    exactly on the PE (a one-hot ``[T, G]`` selection times the token-major
    hidden rows), the expert runs on ``G`` columns with the token-axis path's
    arithmetic, and a one-hot PE product adds each column into its token's
    ``acc`` column. The vector work per item scales with ``G``, except the
    ``acc`` add (``H/128 * T`` per partition).
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
    # Hidden blocks per scatter product: hb_blk * G partitions, hb_blk * T columns.
    hb_blk = min(nh, 128 // group)
    while nh % hb_blk:
        hb_blk = hb_blk - 1
    kernel_assert(hb_blk * tokens <= _BANK, "scatter product fits one PSUM bank")
    rows = hb_blk * group
    # Items run in pairs (2p, 2p + 1), see below; pairs covers the work list
    # plus the prefetch past its end.
    pairs = items // 2 + 2
    kernel_assert(2 * pairs <= _BANK, "item tables fit one PSUM bank")

    # Token-major hidden rows (the gather's stationaries) and this group's
    # affinities with the token on the partition axis.
    x_rows = nl.ndarray((tokens, hidden_size), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_rows, src=hidden)
    aff_t = nl.ndarray((tokens, wide), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aff_t[:, 0:experts], src=affinity.ap(
        pattern=[[groups * experts, tokens], [1, experts]],
        scalar_offset=rank_reg, indirect_dim=1))

    # Constants, built first: their iotas run on the engine that triggers the
    # weight DMAs, and built after the work list they delay item 0's DMA.
    upper = nl.ndarray((tokens, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=upper, pattern=[[1, tokens]], offset=0, channel_multiplier=-1)
    # The scatter's T x T identity (bf16) and block mask [(b, j), b', t] =
    # (b == b'), i.e. 0 <= (b, j) - b' * G < G.
    ident = nl.ndarray((tokens, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident, data=upper, op0=nl.equal, operand0=0.0)
    nisa.tensor_scalar(dst=upper, data=upper, op0=nl.greater_equal, operand0=0.0)
    offset = nl.ndarray((rows, hb_blk, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=offset, pattern=[[-group, hb_blk], [0, tokens]], offset=0,
              channel_multiplier=1)
    low = nl.ndarray((rows, hb_blk, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=low, data=offset, op0=nl.greater_equal, operand0=0.0)
    blockmask = nl.ndarray((rows, hb_blk, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=blockmask, data=offset, op0=nl.less, operand0=float(group))
    nisa.tensor_tensor(dst=blockmask, data1=blockmask, data2=low, op=nl.multiply)
    # The item tables' constants: ones rows and columns, expert and column ids.
    ones_e = _tile(1, max(experts, tokens))
    nisa.memset(dst=ones_e, value=1.0)
    expert_id = _tile(experts, 1)
    nisa.iota(dst=expert_id, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    column_id = nl.ndarray((tokens, 2 * pairs, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=column_id, pattern=[[0, 2 * pairs], [1, group]], offset=0,
              channel_multiplier=0)
    # K=T ones: broadcasts the selected gate weights to every partition.
    ones_col = nl.ndarray((tokens, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)

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
    item_expert = nl.ndarray((1, max(items, 8)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=item_expert[:, 0:items], data=work[:, 0:items],
                       op0=nl.right_shift, operand0=shift)
    item_group = nl.ndarray((1, max(items, 8)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=item_group[:, 0:items], data=work[:, 0:items],
                       op0=nl.bitwise_and, operand0=slots - 1)
    # The item's first column among its expert's tokens: group * G.
    item_first = nl.ndarray((1, max(items, 8)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=item_first[:, 0:items], data=item_group[:, 0:items],
                       op0=nl.left_shift, operand0=GROUP_SHIFT)
    # Items run in pairs (2p, 2p + 1) so that item 2p + 1's weights stream in
    # while item 2p computes, and item 2p + 2's while 2p + 1 computes: two
    # operand buffers, swapped statically. by_pair[q][p] is item 2p + q's
    # expert or first column (q = 0, 1, 2); entries past the work list are 0
    # (expert 0, loaded but never computed).
    padded = 2 * pairs + 2
    flat_expert = nl.ndarray((1, padded), dtype=nl.int32, buffer=nl.sbuf)
    flat_first = nl.ndarray((1, padded), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=flat_expert, value=0)
    nisa.memset(dst=flat_first, value=0)
    nisa.tensor_copy(dst=flat_expert[:, 0:items], src=item_expert[:, 0:items])
    nisa.tensor_copy(dst=flat_first[:, 0:items], src=item_first[:, 0:items])
    by_pair = []
    for q in range(3):
        pair_expert = nl.ndarray((1, pairs), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pair_expert, src=flat_expert.ap(
            pattern=[[padded, 1], [2, pairs]], offset=q))
        pair_first = nl.ndarray((1, pairs), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pair_first, src=flat_first.ap(
            pattern=[[padded, 1], [2, pairs]], offset=q))
        by_pair.append((pair_expert, pair_first))
    halves = _tile(1, 2, nl.int32)
    nisa.tensor_scalar(dst=halves[:, 0:1], data=work[:, items:items + 1],
                       op0=nl.right_shift, operand0=1)
    nisa.tensor_scalar(dst=halves[:, 1:2], data=work[:, items:items + 1],
                       op0=nl.bitwise_and, operand0=1)
    full_pairs = nisa.register_alloc()
    nisa.register_load(dst=full_pairs, src=halves[:, 0:1])
    odd = nisa.register_alloc()
    nisa.register_load(dst=odd, src=halves[:, 1:2])

    buffers = []
    operand_views = []
    for _ in range(2):
        tiles = _operand_tiles(weights, nic, fp8)
        buffers.append(tiles)
        operand_views.append((tiles[0], tiles[1], tiles[2].reshape((128, 3 * nic * nh))))

    # Item 0's operands stream in while the tables below are built.
    first_load = _tile(1, 1, nl.uint32)
    nisa.tensor_copy(dst=first_load, src=by_pair[0][0][:, 0:1])
    first_expert = nisa.register_alloc()
    nisa.register_load(dst=first_expert, src=first_load)
    _load_operands(weights, scales, first_expert, i0, nic, buffers[0])

    # Each token's rank among the tokens of each expert, token on the partition
    # axis: ranks[t, e] = #{t' <= t routed to e} - 1, and -1 where t did not
    # choose e. The prefix count is one PE product with an upper-triangular ones
    # matrix (integer sums of at most 64 ones: exact). No compaction runs inside
    # the item loop (the backend does not accept one there).
    mask_t = _tile(tokens, experts)
    nisa.tensor_scalar(dst=mask_t, data=aff_t[:, 0:experts], op0=nl.not_equal,
                       operand0=0.0)
    prefix = nl.ndarray((tokens, wide), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=prefix[:, 0:experts], stationary=upper, moving=mask_t)
    ranks = nl.ndarray((tokens, wide), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ranks[:, 0:experts], data1=prefix[:, 0:experts],
                       data2=mask_t, op=nl.multiply)
    nisa.tensor_scalar(dst=ranks[:, 0:experts], data=ranks[:, 0:experts],
                       op0=nl.subtract, operand0=1.0)
    tables = _grouped_tables(aff_t, ranks, by_pair,
                             (ones_e, expert_id, column_id, ones_col),
                             (tokens, experts, pairs))

    state = (x_rows, ident, blockmask, acc, clamp, (tokens, nh, nic, pairs, hb_blk))

    held = [_tile(1, 1, nl.uint32), _tile(1, 1, nl.uint32)]

    def run_pair(p):
        _grouped_load(weights, scales, by_pair[1][0], pairs, p, i0, nic, buffers[1],
                      held[1])
        _grouped_compute(state, tables[0], p, operand_views[0])
        _grouped_load(weights, scales, by_pair[2][0], pairs, p, i0, nic, buffers[0],
                      held[0])
        _grouped_compute(state, tables[1], p, operand_views[1])

    nl.fori_loop(0, full_pairs, run_pair)

    def run_last(_):
        _grouped_compute(state, tables[0], full_pairs, operand_views[0])

    nl.fori_loop(0, odd, run_last)


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
    """2 programs (both LNC2 cores) when the runtime is LNC2 and I, H split evenly.

    :func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair` refuses a setting other
    than unset, 1 or 2 (``LaunchGridError``).
    """
    if lnc_pair() and ni % 2 == 0 and nh % 2 == 0:
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
