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
  The token axis is real: every product streams all ``T`` tokens, and a token
  that did not choose the expert has affinity 0 there, so it adds exactly 0.
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
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

from .fused_fp8_pack import PackedExperts

#: Largest token count the decode kernel serves in one launch.
EXPERT_DECODE_MAX_TOKENS = 64

#: fp32 columns in one 2 KiB PSUM bank.
_BANK = 512


@dataclass
class _Counters:
    nki_dispatch: int = 0
    torch_fallback: int = 0


_COUNTERS = _Counters()


def reset_expert_decode_counters() -> None:
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def expert_decode_dispatch_counters() -> Tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    _COUNTERS.torch_fallback += 1


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


def _slot_groups(slots, tokens):
    """Split ``slots`` product slots of ``tokens`` fp32 columns into PSUM banks."""
    per_bank = max(1, _BANK // tokens)
    return [(s0, min(per_bank, slots - s0)) for s0 in range(0, slots, per_bank)]


@nki.jit
def expert_decode_kernel(hidden, affinity, rank, weights, scales, bounds,
                         OUT_FP32: bool = False, WEIGHT_FP8: bool = False):
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

    # ---- Prologue: rank, local affinities, the visited-expert list. --------- #
    rank_sb = _tile(1, 1, nl.int32)
    nisa.dma_copy(dst=rank_sb, src=rank)
    rank_reg = nisa.register_alloc()
    nisa.register_load(dst=rank_reg, src=rank_sb)
    # Partition 0: this group's [T, E] slice, token-major.
    local = nl.ndarray((1, tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=local, src=affinity.ap(
        pattern=[[0, 1], [groups * experts, tokens], [1, experts]],
        scalar_offset=rank_reg, indirect_dim=1))
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

    clamp = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=clamp, src=bounds)
    acc = nl.ndarray((128, nh, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)

    gate_up_slots = 2 * nic * nh
    down_slots = nh * nic

    def visit_expert(slot):
        expert_sb = _tile(1, 1, nl.int32)
        nisa.tensor_copy(dst=expert_sb, src=visit.ap(
            pattern=[[max(experts + 1, 8), 1], [1, 1]],
            scalar_offset=slot, indirect_dim=1))
        expert = nisa.register_alloc()
        nisa.register_load(dst=expert, src=expert_sb)

        # This program's scale rows, broadcast: [128, 3 (gate, up, down), nic, nh].
        scale = nl.ndarray((128, 3, nic, nh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=scale.reshape((128, 3 * nic * nh)), src=scales.ap(
            pattern=[[0, 128], [ni * nh, 3], [1, nic * nh]], offset=i0 * nh,
            scalar_offset=expert, indirect_dim=0))
        flat_scale = scale.reshape((128, 3 * nic * nh))
        stationaries = []
        for kind in range(2):
            for k in range(nic):
                stationaries.append(
                    _weight_panel(weights, expert, kind * ni + i0 + k, WEIGHT_FP8))
        down = [_weight_panel(weights, expert, 2 * ni + i0 + k, WEIGHT_FP8)
                for k in range(nic)]
        gate = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gate, src=gates.ap(
            pattern=[[tokens * experts, 128], [experts, tokens]],
            scalar_offset=expert, indirect_dim=2))

        # Gate/up: slot s = panel * nh + hb, product [128 (I), T] per slot.
        scaled = nl.ndarray((128, gate_up_slots, tokens), dtype=nl.float32,
                            buffer=nl.sbuf)
        for s0, sn in _slot_groups(gate_up_slots, tokens):
            bank = nl.ndarray((128, sn, tokens), dtype=nl.float32, buffer=nl.psum)
            for j in range(sn):
                panel, hb = divmod(s0 + j, nh)
                nisa.nc_matmul(dst=bank[:, j, :], stationary=stationaries[panel][:, hb, :],
                               moving=xt[:, hb, :], accumulate=False)
            nisa.tensor_tensor(
                dst=scaled[:, s0:s0 + sn, :], data1=bank,
                data2=flat_scale.ap(pattern=[[3 * nic * nh, 128], [1, sn], [0, tokens]],
                                    offset=s0),
                op=nl.multiply)
        pre = nl.ndarray((128, 2 * nic, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=pre, data=scaled.ap(pattern=[
            [gate_up_slots * tokens, 128], [nh * tokens, 2 * nic], [1, tokens],
            [tokens, nh]]), op=nl.add, axis=(3,))

        # SwiGLU, the 5938748 instruction sequence: clamp, sigmoid, two products.
        g = nl.ndarray((128, nic, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g, data=pre[:, 0:nic, :], op0=nl.minimum,
                           operand0=clamp[:, 0:1])
        u = nl.ndarray((128, nic, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=u, data=pre[:, nic:2 * nic, :], op0=nl.maximum,
                           operand0=clamp[:, 1:2], op1=nl.minimum,
                           operand1=clamp[:, 2:3])
        sig = nl.ndarray((128, nic, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sig, op=nl.sigmoid, data=g)
        silu = nl.ndarray((128, nic, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=silu, data1=g, data2=sig, op=nl.multiply)
        act = nl.ndarray((128, nic, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act, data1=silu, data2=u, op=nl.multiply)

        # Down: slot s = hb * nic + k, product [128 (H), T]; scale * gate weight.
        weight = nl.ndarray((128, nh, nic, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=weight,
            data1=flat_scale.ap(pattern=[[3 * nic * nh, 128], [1, nh], [nh, nic],
                                         [0, tokens]], offset=2 * nic * nh),
            data2=gate.ap(pattern=[[tokens, 128], [0, nh], [0, nic], [1, tokens]]),
            op=nl.multiply)
        flat_weight = weight.reshape((128, down_slots, tokens))
        terms = nl.ndarray((128, down_slots, tokens), dtype=nl.float32, buffer=nl.sbuf)
        for s0, sn in _slot_groups(down_slots, tokens):
            bank = nl.ndarray((128, sn, tokens), dtype=nl.float32, buffer=nl.psum)
            for j in range(sn):
                hb, k = divmod(s0 + j, nic)
                nisa.nc_matmul(dst=bank[:, j, :], stationary=down[k][:, hb, :],
                               moving=act[:, k, :], accumulate=False)
            nisa.tensor_tensor(dst=terms[:, s0:s0 + sn, :], data1=bank,
                               data2=flat_weight[:, s0:s0 + sn, :], op=nl.multiply)
        contribution = nl.ndarray((128, nh, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=contribution, data=terms.ap(pattern=[
            [down_slots * tokens, 128], [nic * tokens, nh], [1, tokens],
            [tokens, nic]]), op=nl.add, axis=(3,))
        nisa.tensor_tensor(dst=acc, data1=acc, data2=contribution, op=nl.add)

    nl.fori_loop(0, count, visit_expert)

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
# Host wrapper
# ---------------------------------------------------------------------------- #


def _geometry(packed: PackedExperts):
    experts, panels, contraction, nh, channels = packed.weights.shape
    return experts, panels // 3, nh


def can_run_expert_decode(hidden_states: Tensor, expert_affinities: Tensor,
                          packed: PackedExperts) -> bool:
    """True when the decode kernel serves this call on an NKI device or simulator."""
    if packed.weights.ndim != 5 or hidden_states.ndim != 2:
        return False
    experts, ni, nh = _geometry(packed)
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


def _default_programs(ni: int, nh: int) -> int:
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and ni % 2 == 0 and nh % 2 == 0:
        return 2
    return 1


def _rank_operand(rank, device) -> Tensor:
    if isinstance(rank, Tensor):
        return rank.reshape(1, 1).to(device=device, dtype=torch.int32)
    return torch.full((1, 1), int(rank), dtype=torch.int32, device=device)


def expert_decode_torch_oracle(hidden_states, expert_affinities, packed, bounds,
                               expert_parallel_rank, out_dtype=None):
    """Torch reference with the kernel's arithmetic; used when no device is present."""
    from .fused_fp8_pack import unpack_experts

    out_dtype = hidden_states.dtype if out_dtype is None else out_dtype
    experts, ni, nh = _geometry(packed)
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


def fused_fp8_decode_experts(
    hidden_states: Tensor,
    expert_affinities: Tensor,
    packed: PackedExperts,
    bounds: Tensor,
    expert_parallel_rank: int | Tensor = 0,
    *,
    programs: int | None = None,
    out_dtype: torch.dtype | None = None,
    weight_fp8: bool = False,
) -> Tensor:
    """This rank's routed-expert output for ``T <= 64`` decode tokens, one launch.

    Args:
        hidden_states: ``[T, H]`` bf16 expert inputs (real tokens only, no pad row).
        expert_affinities: ``[T, E_global]`` fp32 scattered router output, the
            form ``route_tokens`` returns; ``E_global`` is a multiple of the
            bank's expert count.
        packed: the packed fp8 bank of this rank's experts.
        bounds: ``[128, 3]`` fp32 SwiGLU bounds (``_swiglu_bound_operand``).
        expert_parallel_rank: this bank's group: an int, or an int tensor of one
            element (the runner's device operand, so every rank shares a graph).
        programs: 1 or 2 programs; default 2 under ``NEURON_LOGICAL_NC_CONFIG=2``.
        out_dtype: ``torch.bfloat16`` (default: ``hidden_states.dtype``) or
            ``torch.float32``.
        weight_fp8: fp8 stationaries instead of a bf16 DMA cast (same products).

    Returns:
        ``[T, H]`` in ``out_dtype``: ``sum_e affinity[t, e] * expert_e(x_t)`` over
        this bank's experts, accumulated in fp32.
    """
    out_dtype = hidden_states.dtype if out_dtype is None else out_dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"out_dtype must be bf16 or fp32, got {out_dtype}")
    if tuple(bounds.shape) != (128, 3) or bounds.dtype != torch.float32:
        raise ValueError("bounds must be fp32 [128, 3]")
    experts, ni, nh = _geometry(packed)
    tokens = hidden_states.shape[0]
    if not can_run_expert_decode(hidden_states, expert_affinities, packed):
        _count_torch_fallback()
        return expert_decode_torch_oracle(hidden_states, expert_affinities, packed,
                                          bounds, expert_parallel_rank, out_dtype)
    programs = _default_programs(ni, nh) if programs is None else int(programs)
    _count_nki_dispatch()
    kernel = wrap_nki(expert_decode_kernel)
    if programs != 1:
        kernel = kernel[programs]
    return kernel(
        hidden=hidden_states.contiguous(),
        affinity=expert_affinities.to(torch.float32).reshape(
            tokens, -1, experts).contiguous(),
        rank=_rank_operand(expert_parallel_rank, hidden_states.device),
        weights=packed.weights,
        scales=packed.scales,
        bounds=bounds.contiguous(),
        OUT_FP32=out_dtype == torch.float32,
        WEIGHT_FP8=bool(weight_fp8),
    )
