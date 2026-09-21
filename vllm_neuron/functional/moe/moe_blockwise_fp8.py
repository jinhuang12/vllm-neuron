# SPDX-License-Identifier: Apache-2.0
"""Block-quantised fp8 MoE expert matmuls for Neuron.

Two routes share this module. :func:`blockwise_fp8_moe` adapts ``nkilib``'s
``blockwise_mm_baseline_shard_intermediate``, which consumes ``256 x 256`` block
scales, so the host retiles the checkpoint's ``128 x 128`` grid first
(:mod:`vllm_neuron.functional.moe.blockwise_fp8_retile`). The three ``moe_*``
kernels below index the checkpoint's ``128 x 128`` scales directly: the gate/up
projection, SwiGLU with a transposed result, and the down projection with the
expert affinity folded in.

The vendor kernel reads scales from a tensor whose last axis is ``TILE_SIZE``:
``[E, H//256, 2, I_TP//256, TILE_SIZE]`` for gate/up and
``[E, I_TP//256, H//256, TILE_SIZE]`` for down. Its DMA walks the partition axis
with stride 1 and the block axis with stride ``TILE_SIZE``, so the producer's flat
``(E, n_blocks * TILE_SIZE)`` tensor reshapes onto it in C order.

Every dispatch wrapper counts its NKI and torch entries so a caller can tell which
route ran.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate,
)
from nkilib.core.moe.moe_cte.bwmm_shard_on_I_torch import (
    blockwise_mm_baseline_shard_intermediate_torch_ref,
)
from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    SkipMode,
)
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    DOWN,
    GATE_UP,
    TILE_SIZE,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: SPMD launch grid (``nl.num_programs(axes=0)``); the vendor kernel accepts only 2.
NUM_SHARDS = 2


def _program_block_range(kernel: str, count: int, n_prgs: int, prg_id: int) -> tuple[int, int]:
    """Half-open range of loop indices this program runs, out of ``count``.

    Every program runs ``ceil(count / NUM_SHARDS)`` trips so both cores carry one loop
    structure. The last program is pulled back to end at ``count``, so when the trips
    do not divide ``count`` two programs share one index and write the same values to
    its rows. ``n_prgs`` is the traced launch degree; any other degree is refused
    because ``get_verified_program_sharding_info`` verifies nothing and a one-program
    launch would run the whole range on one core and still compile. It is an
    ``assert`` because the NKI front end refuses ``raise`` inside a kernel.
    """
    assert n_prgs == NUM_SHARDS, (
        f"{kernel}: traced with {n_prgs} programs, the kernel wants {NUM_SHARDS} "
        f"(launch it as wrap_nki(kernel)[NUM_SHARDS])"
    )
    trips = -(-count // n_prgs)
    start = min(prg_id * trips, count - trips)
    return start, start + trips

#: ``H`` bounds from the vendor kernel's own compatibility asserts.
MIN_HIDDEN = 512
MAX_HIDDEN = 8192

#: The vendor kernel also needs ``H % PSUM_SIZE == 0``.
PSUM_SIZE = 512

#: The checkpoint's scale block: one fp32 scale per ``128 x 128`` weight block.
#: Not ``BLOCK_QUANT_SIZE``; that 256 is the vendor kernel's granularity.
GATE_UP_SCALE_BLOCK = TILE_SIZE

#: Contraction tiles per scale block; 1 here, so no partial sum spans two scales.
GATE_UP_H_TILES_PER_BLOCK = GATE_UP_SCALE_BLOCK // TILE_SIZE

#: Gate and up, in that order, fused on one weight axis.
GATE_UP_FUSION = 2

#: Source rows per DMA transpose with a 2-byte operand and a 128-wide row.
DGE_TRANSPOSE_ROWS = 16
#: Elements per padded SBUF row: 16 of a 2- or 4-byte dtype are whole 32-byte lines.
ROW_ALIGN = 16

__all__ = [
    "BLOCK_QUANT_SIZE",
    "DOWN",
    "GATE_UP",
    "GATE_UP_FUSION",
    "GATE_UP_H_TILES_PER_BLOCK",
    "GATE_UP_SCALE_BLOCK",
    "NUM_SHARDS",
    "TILE_SIZE",
    "MoeBlockwiseFp8Error",
    "blockwise_fp8_moe",
    "blockwise_fp8_moe_torch_oracle",
    "can_run_blockwise_fp8_moe",
    "can_run_moe_down_blockwise_fp8",
    "can_run_moe_gate_up_blockwise_fp8",
    "dispatch_counters",
    "down_dispatch_counters",
    "down_flat_scale_index",
    "down_kernel_identity",
    "down_kernel_scale_shape",
    "gate_up_dispatch_counters",
    "gate_up_flat_scale_index",
    "gate_up_kernel_identity",
    "gate_up_kernel_scale_shape",
    "kernel_identity",
    "kernel_scale_shape",
    "moe_down_blockwise_fp8",
    "moe_down_blockwise_fp8_kernel",
    "moe_gate_up_blockwise_fp8",
    "moe_gate_up_blockwise_fp8_kernel",
    "moe_swiglu_transposed",
    "moe_swiglu_transposed_kernel",
    "reset_dispatch_counters",
    "reset_down_dispatch_counters",
    "reset_gate_up_dispatch_counters",
    "reset_swiglu_dispatch_counters",
    "seam_identity",
    "swiglu_dispatch_counters",
    "swiglu_kernel_identity",
    "to_down_kernel_scale_operand",
    "to_gate_up_kernel_scale_operand",
    "to_kernel_scale_layout",
]


class MoeBlockwiseFp8Error(ValueError):
    """A weight or routing geometry this module refuses before a kernel traps on it."""


def kernel_scale_shape(
    num_experts: int, rows: int, cols: int, projection: str = DOWN
) -> tuple[int, ...]:
    """Logical scale shape the vendor kernel and its torch reference consume.

    ``rows`` is ``H`` and ``cols`` is ``I_TP``, both global (unsharded): the host
    tensor is sized on the full ``I_TP`` even though the device buffer is sharded.
    The last axis is ``TILE_SIZE``; see the module docstring.
    """
    _require_blocked(rows, cols)
    if num_experts < 1:
        raise MoeBlockwiseFp8Error(f"num_experts must be >= 1, got {num_experts}")
    h_256 = rows // BLOCK_QUANT_SIZE
    i_256 = cols // BLOCK_QUANT_SIZE
    if projection == DOWN:
        return (num_experts, i_256, h_256, TILE_SIZE)
    if projection == GATE_UP:
        # The 2 is the gate/up fusion.
        return (num_experts, h_256, 2, i_256, TILE_SIZE)
    raise MoeBlockwiseFp8Error(
        f"projection must be {DOWN!r} or {GATE_UP!r}, got {projection!r}"
    )


def to_kernel_scale_layout(
    consumer_scales: Tensor,
    num_experts: int,
    rows: int,
    cols: int,
    projection: str = DOWN,
) -> Tensor:
    """Reshape the flat ``(E, n_blocks * TILE_SIZE)`` scales into the kernel's view.

    Each ``256``-block scale is broadcast across ``TILE_SIZE`` partitions. The
    reshape is C order: the flat offset of block ``b``, replica ``t`` is
    ``b * TILE_SIZE + t``, which is the offset the kernel's DMA reads.

    Raises:
        MoeBlockwiseFp8Error: if ``consumer_scales`` does not have the flat shape
            the extents imply. A mis-sized tensor can reshape without error onto a
            different block-to-scale assignment.
    """
    target = kernel_scale_shape(num_experts, rows, cols, projection)
    n_blocks = 1
    for extent in target[1:-1]:
        n_blocks *= extent
    expected_flat = (num_experts, n_blocks * TILE_SIZE)
    if tuple(consumer_scales.shape) != expected_flat:
        raise MoeBlockwiseFp8Error(
            f"consumer_scales has shape {tuple(consumer_scales.shape)}, expected "
            f"{expected_flat} for projection={projection!r} at "
            f"[H={rows}, I={cols}], E={num_experts}. Refusing to reshape: a "
            f"mis-sized scale tensor can reshape without error onto a different "
            f"block-to-scale assignment."
        )
    return consumer_scales.reshape(target).contiguous()


def _require_blocked(rows: int, cols: int) -> None:
    """Check every extent condition the vendor kernel's own asserts impose."""
    problems: list[str] = []
    if rows <= 0 or rows % BLOCK_QUANT_SIZE:
        problems.append(
            f"H={rows} is not a positive multiple of {BLOCK_QUANT_SIZE} (:680)"
        )
    if not MIN_HIDDEN <= rows <= MAX_HIDDEN:
        problems.append(f"H={rows} outside [{MIN_HIDDEN}, {MAX_HIDDEN}] (:668)")
    if rows % PSUM_SIZE:
        problems.append(f"H={rows} is not a multiple of PSUM_SIZE={PSUM_SIZE} (:204)")
    if cols <= 0 or cols % BLOCK_QUANT_SIZE:
        problems.append(
            f"I_TP={cols} is not a positive multiple of {BLOCK_QUANT_SIZE} (:681)"
        )
    # The kernel's scale index divides I_TP_sharded while its asserts only constrain
    # I_TP, so an odd multiple of 256 truncates the sharded block extent at
    # NUM_SHARDS=2 and the index walks off its block row. The kernel does not refuse
    # this, so it is refused here.
    if cols % (BLOCK_QUANT_SIZE * NUM_SHARDS):
        problems.append(
            f"I_TP={cols} is not a multiple of "
            f"{BLOCK_QUANT_SIZE * NUM_SHARDS} (= BLOCK_QUANT_SIZE * NUM_SHARDS); "
            f"I_TP_sharded={cols // NUM_SHARDS} would truncate to "
            f"{cols // NUM_SHARDS // BLOCK_QUANT_SIZE} scale blocks while the "
            f"kernel's i_block index reaches "
            f"{max(0, cols // NUM_SHARDS // TILE_SIZE // 2 - 1)}"
        )
    if problems:
        raise MoeBlockwiseFp8Error(
            "block-quant MoE refuses this weight geometry: " + "; ".join(problems)
        )


@dataclass
class _DispatchCounters:
    """Entries into the NKI route and into the torch route, counted separately."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch off the traced graph; a traced store is a guard."""
    _COUNTERS.nki_dispatch += 1


def can_run_blockwise_fp8_moe(
    hidden_states: Tensor, rows: int, cols: int
) -> bool:
    """Whether the NKI route is available for an admissible geometry.

    ``can_run_kernel`` answers whether a device or simulator is present;
    :func:`_require_blocked` answers whether the kernel accepts these extents. An
    inadmissible geometry raises rather than falling back to torch.

    Raises:
        MoeBlockwiseFp8Error: if the geometry is inadmissible.
    """
    _require_blocked(rows, cols)
    return can_run_kernel(hidden_states)


@nki.jit(mode="trace")
def _torch_compatible_blockwise_mm_baseline_shard_intermediate(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    block_size: int,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    gate_up_hidden_scale: Optional[nl.NkiTensor] = None,
    down_hidden_scale: Optional[nl.NkiTensor] = None,
    is_block_quant: bool = False,
    is_per_tensor: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    # The vendor's ``skip_dma: SkipMode = SkipMode()`` becomes two flat booleans;
    # the object is rebuilt inside the kernel body.
    skip_token: bool = False,
    skip_weight: bool = False,
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = (
        ExpertAffinityScaleMode.PRE_SCALE
    ),
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    accumulation_dtype: Optional[Any] = None,
    skip_gate_proj: bool = False,
):
    """The vendor kernel with a torch-traceable signature. Numerics unchanged.

    The vendor kernel defaults ``skip_dma`` to a live ``SkipMode()`` object.
    ``wrap_nki`` folds a kernel's defaults at trace time, and Dynamo can turn that
    object neither into a constant nor into a graph proxy, so the raw kernel cannot
    be traced. Every default here is ``None``, a primitive or an enum member, and
    ``SkipMode`` is built inside the body from two flat booleans, because the NKI
    parser cannot read attributes off a nested NKIObject. ``skip_token=False,
    skip_weight=False`` is the state ``SkipMode()`` produced.
    """
    skip_dma = SkipMode(skip_token=skip_token, skip_weight=skip_weight)

    return blockwise_mm_baseline_shard_intermediate(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        gate_up_hidden_scale=gate_up_hidden_scale,
        down_hidden_scale=down_hidden_scale,
        is_block_quant=is_block_quant,
        is_per_tensor=is_per_tensor,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        accumulation_dtype=accumulation_dtype,
        skip_gate_proj=skip_gate_proj,
    )


def blockwise_fp8_moe(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    block_size: int,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    gate_up_proj_scale: Tensor,
    down_proj_scale: Tensor,
    **kernel_kwargs: Any,
) -> Tensor:
    """Block-quantised fp8 MoE matmul through the vendor kernel, with a torch fallback.

    Args:
        hidden_states: ``[T+1, H]``. The trailing row is the padding-token slot.
        expert_affinities_masked: ``[(T+1) * E, 1]``.
        gate_up_proj_weight: ``[E, H, 2, I_TP]``, fp8.
        down_proj_weight: ``[E, I_TP, H]``, fp8.
        block_size: tokens per block, a multiple of ``256``.
        token_position_to_id: ``[N * B]``, int32.
        block_to_expert: ``[N, 1]``, int32.
        gate_up_proj_scale: logical ``[E, H//256, 2, I_TP//256, TILE_SIZE]``.
        down_proj_scale: logical ``[E, I_TP//256, H//256, TILE_SIZE]``.
        **kernel_kwargs: forwarded verbatim to the vendor kernel and, on the
            fallback path, to its torch reference.

    Returns:
        ``[T+1, H]`` output hidden states.
    """
    rows = hidden_states.shape[-1]
    cols = down_proj_weight.shape[-2]

    if not can_run_blockwise_fp8_moe(hidden_states, rows, cols):
        _count_torch_fallback()
        logger.debug(
            "blockwise_fp8_moe: NKI route unavailable, using the torch path "
            "(oracle / constraint-violation fallback, not the shipped path)"
        )
        return blockwise_fp8_moe_torch_oracle(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            **kernel_kwargs,
        )

    _count_nki_dispatch()
    # ``wrap_nki(...)[NUM_SHARDS]`` is the SPMD launch grid, not an output arity.
    wrapped = wrap_nki(_torch_compatible_blockwise_mm_baseline_shard_intermediate)
    return wrapped[NUM_SHARDS](
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        is_block_quant=True,
        **kernel_kwargs,
    )


def blockwise_fp8_moe_torch_oracle(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    block_size: int,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    gate_up_proj_scale: Tensor,
    down_proj_scale: Tensor,
    **kernel_kwargs: Any,
) -> Tensor:
    """The vendor kernel's own torch reference, block-quant path.

    The wrapper is called rather than ``_moe_cte_torch_ref_impl`` because the two
    default ``expert_affinities_scaling_mode`` differently: the wrapper matches the
    kernel (``PRE_SCALE``), the impl uses ``POST_SCALE``.

    Returns:
        ``[T+1, H]``, fp32.
    """
    result = blockwise_mm_baseline_shard_intermediate_torch_ref(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        is_block_quant=True,
        **kernel_kwargs,
    )
    return result["output"]


def _gate_up_sbuf(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """One fp32 accumulator tile in SBUF."""
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _gate_up_psum(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """One fp32 matmul destination in PSUM.

    Allocated per contraction block so PSUM is reclaimed per block and that block's
    scale is applied before the next block accumulates.
    """
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _padded(width: int) -> int:
    """``width`` rounded up to a whole :data:`ROW_ALIGN` block."""
    return ((width + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN


def _column(rows: int):
    """An int32 ``(rows, 1)`` view of an SBUF tile whose row is whole 32-byte lines.

    Index tiles carry one address per partition; a ``(1, 1)`` tile does not compile
    as ``nisa.tensor_scalar``'s ``operand0``.
    """
    return nl.ndarray((rows, ROW_ALIGN), dtype=nl.int32, buffer=nl.sbuf)[:, 0:1]


def _row_iota(iota_hbm, rows: int):
    """The partition ramp ``0..rows-1`` as an int32 ``(rows, 1)`` SBUF tile.

    This NKI build has no ``nl.arange``, ``mgrid`` or ``nl.iota``, so the ramp
    arrives as an operand.
    """
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=iota_hbm.ap(pattern=[[1, rows], [1, 1]], offset=0))
    return tile


def _loaded(source, rows: int, width: int, dtype):
    """``rows`` by ``width`` of an HBM access pattern, in SBUF.

    The DMA converts the element type on the way in.
    """
    tile = nl.ndarray((rows, _padded(width)), dtype=dtype, buffer=nl.sbuf)[:, 0:width]
    nisa.dma_copy(dst=tile, src=source)
    return tile


def _stored(destination, tile):
    """Write one SBUF tile through an HBM access pattern on device descriptors.

    ``nl.store`` takes a slice, and a slice cannot carry a device offset.
    """
    nisa.dma_copy(destination, tile, dge_mode=nisa.dge_mode.hwdge)


def _dense_column(hbm, rows: int, offset: int, at_block):
    """``rows`` consecutive int32 of an ``[n, 1]`` HBM tensor, one per partition.

    The slice is contiguous and its start is a trace-time integer, so no index tile
    is needed. ``at_block`` moves the start by one device block on a tensor reshaped
    ``(blocks, block, 1)``.
    """
    tile = _column(rows)
    nisa.dma_copy(
        dst=tile,
        src=hbm.ap(
            pattern=[[1, rows], [1, 1]],
            offset=offset,
            scalar_offset=at_block,
            indirect_dim=0,
        ),
    )
    return tile


def _broadcast_row(hbm, row: int, rows: int, at_block):
    """One int32 of an ``[n, 1]`` HBM tensor, replicated into every partition.

    A zero partition stride does the replication: all ``rows`` partitions read one
    address. One row of an ``[n, 1]`` tensor is one element, so ``at_block``
    addresses the block itself.
    """
    tile = _column(rows)
    nisa.dma_copy(
        dst=tile,
        src=hbm.ap(
            pattern=[[0, rows], [1, 1]],
            offset=row,
            scalar_offset=at_block,
            indirect_dim=0,
        ),
    )
    return tile


def _padding_resolved(rows: int, positions, pad_row: int):
    """Token positions with the routing map's ``-1`` sent to the padding row, on device.

    ``index = pos + (pos < 0) * (pad_row + 1)``: ``-1`` lands on ``pad_row`` and every
    real position is unchanged. Three calls, each with a Python scalar in the scalar
    slot, the only form of that slot this NKI build compiles. A real appended row is
    used rather than an out-of-bounds read because the torch reference can compute
    it too, so padded rows are compared rather than excused.
    """
    negative = _column(rows)
    nisa.tensor_scalar(dst=negative, data=positions, op0=nl.less, operand0=0)
    bumped = _column(rows)
    nisa.tensor_scalar(
        dst=bumped, data=negative, op0=nl.multiply, operand0=pad_row + 1
    )
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=positions, data2=bumped, op=nl.add)
    return index


def _bank_rows(rows: int, expert, iota, stride: int, offset: int, step: int):
    """Row addresses of one expert's slab: ``expert * stride + offset + step * i``.

    ``nisa.tensor_scalar`` forms this int32 in float32, so every address must stay
    below 2**24 or its low bits round away; a weight bank is therefore addressed in
    contraction rows (``expert * H + h``), never in 128-wide column blocks. The
    expert stays a device scalar: no host read and no slab copy stands between the
    routing map and the weight. ``step`` is the ramp's stride, 1 for a bank viewed
    ``[E * rows, cols]``.
    """
    base = _column(rows)
    nisa.tensor_scalar(dst=base, data=expert, op0=nl.multiply, operand0=stride)
    nisa.tensor_scalar(dst=base, data=base, op0=nl.add, operand0=offset)
    ramp = _column(rows)
    nisa.tensor_scalar(dst=ramp, data=iota, op0=nl.multiply, operand0=step)
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=base, data2=ramp, op=nl.add)
    return index


def _affinity_rows(rows: int, resolved, expert, n_experts: int):
    """``row * E_local + expert`` per partition: the flat affinity address.

    The routing map flattens affinities token-major, ``[T * E_local, 1]``, so one
    token's expert entry is one row of that tensor.
    """
    scaled = _column(rows)
    nisa.tensor_scalar(dst=scaled, data=resolved, op0=nl.multiply, operand0=n_experts)
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=scaled, data2=expert, op=nl.add)
    return index


def _gathered(
    hbm, index, rows: int, width: int, dtype, row_width: int = 0, column: int = 0
):
    """``rows`` by ``width`` of an ``[n, row_width]`` HBM tensor, gathered by ``index``.

    Indirect DMA: the pattern's partition dimension is replaced by one row address
    per partition. ``column`` is the static element offset of the tile inside its
    row, so one call reads one column block of a wider row; ``row_width`` defaults
    to ``width``.
    """
    tile = nl.ndarray((rows, _padded(width)), dtype=dtype, buffer=nl.sbuf)[:, 0:width]
    nisa.dma_copy(
        dst=tile,
        src=hbm.ap(
            pattern=[[row_width or width, rows], [1, width]],
            offset=column,
            vector_offset=index,
            indirect_dim=0,
        ),
    )
    return tile


def gate_up_bank_address(
    h_extent: int, h0: int, column_block: int
) -> tuple[int, int, int, int]:
    """Address of one gate/up weight tile in the ``[E * H, 2 * I]`` bank.

    Returns ``(stride, offset, step, column)``: the row is ``expert * h_extent + h0
    + i`` and the column block is a static element offset. The largest row,
    ``E * H - 1``, stays below 2**24 for this model.
    """
    return h_extent, h0, 1, column_block * GATE_UP_SCALE_BLOCK


def down_bank_address(
    i_extent: int, i0: int, h_block: int
) -> tuple[int, int, int, int]:
    """Address of one down weight tile in the ``[E * I, H]`` bank; see gate/up."""
    return i_extent, i0, 1, h_block * GATE_UP_SCALE_BLOCK


def _transpose_rows(
    dst, src_hbm, row_stride: int, rows: int, width: int, offset: int, at_block
):
    """Transpose ``rows`` source rows of ``width`` elements onto partitions.

    16 rows per DMA is the transpose engine's descriptor shape for a 2-byte operand.
    """
    for r0 in range(0, rows, DGE_TRANSPOSE_ROWS):
        n = min(DGE_TRANSPOSE_ROWS, rows - r0)
        nisa.dma_transpose(
            dst=dst[:, r0 : r0 + n],
            src=src_hbm.ap(
                pattern=[[row_stride, n], [1, width]],
                offset=offset + r0 * row_stride,
                scalar_offset=at_block,
                indirect_dim=0,
            ),
        )


def gate_up_flat_scale_index(
    h_block: int, gate_or_up: int, i_block: int, n_i_blocks: int
) -> int:
    """Column of the gate/up kernel's scale operand that holds one block's scale.

    The operand is ``[TILE_SIZE, n_blocks]``: one column per ``128 x 128`` weight
    block, replicated down the partition axis because ``nisa.tensor_scalar``
    broadcasts only along the free axis. The column order is the C-order flattening
    of the checkpoint's ``[H//128, 2, I//128]`` grid.
    """
    if n_i_blocks < 1:
        raise MoeBlockwiseFp8Error(f"n_i_blocks must be >= 1, got {n_i_blocks}")
    if not 0 <= gate_or_up < GATE_UP_FUSION:
        raise MoeBlockwiseFp8Error(
            f"gate_or_up must be 0 (gate) or 1 (up), got {gate_or_up}"
        )
    return (h_block * GATE_UP_FUSION + gate_or_up) * n_i_blocks + i_block


def gate_up_kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``[TILE_SIZE, n_blocks]``, the gate/up scale operand shape for one expert.

    ``rows`` is ``H`` and ``cols`` is ``I`` (one half's width, not the fused width).
    """
    _require_gate_up_weight_blocked(rows, cols)
    n_blocks = (rows // GATE_UP_SCALE_BLOCK) * GATE_UP_FUSION * (
        cols // GATE_UP_SCALE_BLOCK
    )
    return (TILE_SIZE, n_blocks)


def to_gate_up_kernel_scale_operand(
    checkpoint_scales: Tensor, rows: int, cols: int
) -> Tensor:
    """One expert's ``[H//128, 2, I//128]`` checkpoint scales as the kernel operand.

    A C-order reshape and a partition-axis broadcast; the column order is the
    grid's own order.

    Raises:
        MoeBlockwiseFp8Error: if the grid is not the shape the extents imply. A
            mis-sized grid reshapes without error onto the wrong scales.
    """
    _require_gate_up_weight_blocked(rows, cols)
    expected = (
        rows // GATE_UP_SCALE_BLOCK,
        GATE_UP_FUSION,
        cols // GATE_UP_SCALE_BLOCK,
    )
    if tuple(checkpoint_scales.shape) != expected:
        raise MoeBlockwiseFp8Error(
            f"mis-sized gate/up scale grid: got {tuple(checkpoint_scales.shape)}, "
            f"expected {expected} at [H={rows}, I={cols}] and block "
            f"{GATE_UP_SCALE_BLOCK}. Refusing to reshape: a mis-sized grid maps "
            f"blocks to the wrong scales without raising."
        )
    flat = checkpoint_scales.to(torch.float32).reshape(1, -1)
    return flat.expand(TILE_SIZE, flat.shape[1]).contiguous()


def _gate_up_weight_problems(rows: int, cols: int) -> list[str]:
    """The weight-axis conditions as messages; an empty list means admissible."""
    problems: list[str] = []
    if rows <= 0 or rows % GATE_UP_SCALE_BLOCK:
        problems.append(
            f"H={rows} is not a positive multiple of "
            f"GATE_UP_SCALE_BLOCK={GATE_UP_SCALE_BLOCK}; one contraction block "
            f"carries exactly one scale"
        )
    if cols <= 0 or cols % GATE_UP_SCALE_BLOCK:
        problems.append(
            f"I={cols} is not a positive multiple of "
            f"GATE_UP_SCALE_BLOCK={GATE_UP_SCALE_BLOCK}; one output block "
            f"carries exactly one scale"
        )
    return problems


def _refuse_gate_up(problems: list[str]) -> None:
    """Raise the named error if there are problems."""
    if problems:
        raise MoeBlockwiseFp8Error(
            "the gate/up kernel refuses this geometry: "
            + "; ".join(problems)
        )


def _require_gate_up_weight_blocked(rows: int, cols: int) -> None:
    """The weight-axis conditions alone: ``H`` and ``I`` are ``128``-blocked."""
    _refuse_gate_up(_gate_up_weight_problems(rows, cols))


def _require_gate_up_blocked(tokens: int, rows: int, cols: int) -> None:
    """Every extent condition the gate/up kernel's loops impose.

    ``tokens`` is ``B``, ``rows`` is ``H``, ``cols`` is ``I`` (one half). Only
    positivity and divisibility are checked: the kernel walks every axis in tiles,
    so it has no magnitude bound.
    """
    problems: list[str] = []
    if tokens <= 0 or tokens % TILE_SIZE:
        problems.append(
            f"B={tokens} is not a positive multiple of TILE_SIZE={TILE_SIZE}; "
            f"the kernel walks tokens in {TILE_SIZE}-row PSUM tiles"
        )
    problems += _gate_up_weight_problems(rows, cols)
    _refuse_gate_up(problems)


@dataclass
class _GateUpDispatchCounters:
    """Dispatch counts for the gate/up path, separate from the vendor route's."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_GATE_UP_COUNTERS = _GateUpDispatchCounters()


def reset_gate_up_dispatch_counters() -> None:
    """Zero the gate/up counters."""
    _GATE_UP_COUNTERS.nki_dispatch = 0
    _GATE_UP_COUNTERS.torch_fallback = 0


def gate_up_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the gate/up path since the reset.

    ``torch_fallback`` is always 0: this path has no torch route, and an
    inadmissible geometry raises.
    """
    return _GATE_UP_COUNTERS.nki_dispatch, _GATE_UP_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_gate_up_nki_dispatch() -> None:
    """Count one kernel dispatch off the traced graph; a traced store is a guard."""
    _GATE_UP_COUNTERS.nki_dispatch += 1


@nki.jit
def moe_gate_up_blockwise_fp8_kernel(
    hidden, weight_bank, scale_bank, row_index, expert_index, iota
):
    """``out[P, 2*I] = routed_rows(hidden) @ dequantise(weight_bank[expert])``, fp32.

    One expert's gate/up projection per token block, with the ``128 x 128`` block
    dequantisation folded into the tile loop and the result left pre-activation.
    Gate and up tiles for one ``i_block`` accumulate side by side, and each
    transposed hidden tile is loaded once and fed to both matmuls.

    Args:
        hidden: ``[T + 1, H]`` bf16, every token of the layer plus the appended
            padding row. The rows each block multiplies are chosen on device.
        weight_bank: the whole expert bank ``[E, H, 2 * I]`` fp8-e4m3, viewed by the
            caller as ``[E * H * n_col_blocks, GATE_UP_SCALE_BLOCK]``. ``[E, H, 2*I]``
            is contiguous, so one ``128``-column block of one row is one row of that
            view and no copy is made.
        scale_bank: ``[E * TILE_SIZE, n_blocks]`` fp32, the per-expert operands of
            :func:`to_gate_up_kernel_scale_operand` stacked.
        row_index: ``[P, 1]`` int32, ``token_position_to_id`` with ``-1`` for a
            padding position. ``P = n_blocks_total * block``.
        expert_index: ``[n_blocks_total, 1]`` int32, ``block_to_expert``.
        iota: ``[TILE_SIZE, 1]`` int32 holding ``0..TILE_SIZE-1``; see
            :func:`_row_iota`.

    Returns:
        ``[P, 2*I]`` fp32, pre-activation, in block order: position ``p`` of the
        routing map, not token ``p``. The caller scatters back with one ``index_add``.

    Only tensors cross ``wrap_nki`` (a non-tensor argument reaches the kernel as
    ``None`` at lowering), so every extent is read from an operand's shape. Each
    ``128``-row tile belongs to exactly one block because ``block`` is a multiple of
    ``TILE_SIZE``, so the block index is a trace-time integer and only the expert is
    a device value; rows arrive by indirect DMA and weight and scale tiles by an
    expert offset into the bank, so nothing is copied or read back to the host.
    Accumulation is PSUM within one contraction block and SBUF across blocks, so
    each block's scale multiplies exactly its own partial sum. ``nc_matmul``
    contracts the partition axis (128 wide); the stationary free axis is at most 128
    and the moving free axis at most 512, and every tile here sits inside those
    bounds. Loop bounds are trace-time ints, hence ``range`` and not
    ``nl.affine_range``.
    """
    n_blocks = expert_index.shape[0]
    _grid_ndim, n_prgs, prg_id = get_verified_program_sharding_info(
        "moe_gate_up_blockwise_fp8", (0, 1), NUM_SHARDS
    )
    first_block, end_block = _program_block_range(
        "moe_gate_up_blockwise_fp8", n_blocks, n_prgs, prg_id
    )
    positions = row_index.shape[0]
    h_extent = hidden.shape[1]
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK
    # One scale column per (H block, fused column block), so the width over the H
    # blocks is the fused column count in blocks.
    n_col_blocks = scale_bank.shape[1] // n_h_blocks
    fused_cols = n_col_blocks * GATE_UP_SCALE_BLOCK
    i_extent = fused_cols // GATE_UP_FUSION
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    # One expert per block: the routing operands' own lengths give the block length.
    block = positions // expert_index.shape[0]
    tiles_per_block = block // TILE_SIZE
    pad_row = hidden.shape[0] - 1

    out = nl.ndarray((positions, fused_cols), dtype=nl.float32, buffer=nl.shared_hbm)
    # Routed rows land here first; each token tile's hidden tiles are then transposed
    # onto partitions once and read by every intermediate block. Kernel-internal, so
    # ``private_hbm``.
    staged = nl.ndarray((positions, h_extent), dtype=hidden.dtype, buffer=nl.private_hbm)
    ramp = _row_iota(iota, TILE_SIZE)
    # The block is the dynamic axis: every tensor a body addresses by block leads with
    # it, so the tile offsets inside a body stay trace-time.
    row_index_b = row_index.reshape((n_blocks, block, 1))
    staged_b = staged.reshape((n_blocks, block, h_extent))
    out_b = out.reshape((n_blocks, block, fused_cols))

    def stage_block(at_block):
        for tile in range(tiles_per_block):
            t0 = tile * TILE_SIZE
            wanted = _padding_resolved(
                TILE_SIZE,
                _dense_column(row_index_b, TILE_SIZE, t0, at_block=at_block),
                pad_row,
            )
            _stored(
                staged_b.ap(
                    pattern=[[h_extent, TILE_SIZE], [1, h_extent]],
                    offset=t0 * h_extent,
                    scalar_offset=at_block,
                    indirect_dim=0,
                ),
                _gathered(hidden, wanted, TILE_SIZE, h_extent, hidden.dtype),
            )

    nl.fori_loop(first_block, end_block, stage_block)

    def project_block(at_block):
        # The expert is a property of the block, so its broadcast and the scale gather
        # that reads it stay outside the tile loop.
        expert = _broadcast_row(expert_index, 0, TILE_SIZE, at_block=at_block)
        scale_sb = _gathered(
            scale_bank,
            _bank_rows(TILE_SIZE, expert, ramp, TILE_SIZE, 0, 1),
            TILE_SIZE,
            n_h_blocks * n_col_blocks,
            nl.float32,
        )
        for token_tile in range(tiles_per_block):
            t0 = token_tile * TILE_SIZE
            # [H=TILE_SIZE partitions, B=TILE_SIZE free] per hidden tile, in the
            # source dtype.
            hidden_t = []
            for h_tile in range(n_h_blocks * GATE_UP_H_TILES_PER_BLOCK):
                tile = nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=hidden.dtype, buffer=nl.sbuf)
                _transpose_rows(
                    tile,
                    staged_b,
                    h_extent,
                    TILE_SIZE,
                    TILE_SIZE,
                    t0 * h_extent + h_tile * TILE_SIZE,
                    at_block=at_block,
                )
                hidden_t.append(tile)
            for i_block in range(n_i_blocks):
                gate_col = i_block * GATE_UP_SCALE_BLOCK
                up_col = i_extent + i_block * GATE_UP_SCALE_BLOCK
                gate_acc = _gate_up_sbuf()
                up_acc = _gate_up_sbuf()
                for h_block in range(n_h_blocks):
                    gate_psum = _gate_up_psum()
                    up_psum = _gate_up_psum()
                    for h_sub in range(GATE_UP_H_TILES_PER_BLOCK):
                        h0 = h_block * GATE_UP_SCALE_BLOCK + h_sub * TILE_SIZE
                        hidden_tile = hidden_t[h0 // TILE_SIZE]
                        # [H=TILE_SIZE partitions, I=GATE_UP_SCALE_BLOCK free]: row
                        # ``expert * H + h0 + i`` of the ``[E * H, 2 * I]`` bank, with
                        # the gate or up column block as a static offset into the row.
                        stride, row0, step, gate_column = gate_up_bank_address(
                            h_extent, h0, i_block
                        )
                        up_column = gate_up_bank_address(
                            h_extent, h0, n_i_blocks + i_block
                        )[3]
                        h_rows = _bank_rows(TILE_SIZE, expert, ramp, stride, row0, step)
                        gate_w = _gathered(
                            weight_bank,
                            h_rows,
                            TILE_SIZE,
                            GATE_UP_SCALE_BLOCK,
                            nl.bfloat16,
                            fused_cols,
                            gate_column,
                        )
                        up_w = _gathered(
                            weight_bank,
                            h_rows,
                            TILE_SIZE,
                            GATE_UP_SCALE_BLOCK,
                            nl.bfloat16,
                            fused_cols,
                            up_column,
                        )
                        # dst = stationary.T @ moving = [B, I]. ``accumulate`` is
                        # explicit so first-write-overwrites is visible here.
                        nisa.nc_matmul(
                            dst=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            stationary=hidden_tile,
                            moving=gate_w,
                            accumulate=(h_sub > 0),
                        )
                        nisa.nc_matmul(
                            dst=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            stationary=hidden_tile,
                            moving=up_w,
                            accumulate=(h_sub > 0),
                        )
                    # The flattening ``gate_up_flat_scale_index`` returns:
                    # ``h_block * n_col_blocks + (gate_or_up * n_i_blocks + i_block)``.
                    gate_flat = h_block * n_col_blocks + i_block
                    up_flat = h_block * n_col_blocks + n_i_blocks + i_block
                    if h_block == 0:
                        # The first block initialises the accumulator; no zeroing pass.
                        nisa.tensor_scalar(
                            dst=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            data=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            op0=nl.multiply,
                            operand0=scale_sb[0:TILE_SIZE, gate_flat : gate_flat + 1],
                        )
                        nisa.tensor_scalar(
                            dst=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            data=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            op0=nl.multiply,
                            operand0=scale_sb[0:TILE_SIZE, up_flat : up_flat + 1],
                        )
                    else:
                        nisa.scalar_tensor_tensor(
                            dst=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            data=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            op0=nl.multiply,
                            operand0=scale_sb[0:TILE_SIZE, gate_flat : gate_flat + 1],
                            op1=nl.add,
                            operand1=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        )
                        nisa.scalar_tensor_tensor(
                            dst=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            data=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            op0=nl.multiply,
                            operand0=scale_sb[0:TILE_SIZE, up_flat : up_flat + 1],
                            op1=nl.add,
                            operand1=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        )
                _stored(
                    out_b.ap(
                        pattern=[[fused_cols, TILE_SIZE], [1, GATE_UP_SCALE_BLOCK]],
                        offset=t0 * fused_cols + gate_col,
                        scalar_offset=at_block,
                        indirect_dim=0,
                    ),
                    gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                )
                _stored(
                    out_b.ap(
                        pattern=[[fused_cols, TILE_SIZE], [1, GATE_UP_SCALE_BLOCK]],
                        offset=t0 * fused_cols + up_col,
                        scalar_offset=at_block,
                        indirect_dim=0,
                    ),
                    up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                )

    nl.fori_loop(first_block, end_block, project_block)
    return out


def can_run_moe_gate_up_blockwise_fp8(
    hidden_states: Tensor, tokens: int, rows: int, cols: int
) -> bool:
    """Whether the NKI route is available for an admissible gate/up geometry.

    ``can_run_kernel`` answers whether a device or simulator is present;
    :func:`_require_gate_up_blocked` answers whether the kernel accepts these
    extents. An inadmissible geometry raises rather than returning False.

    Raises:
        MoeBlockwiseFp8Error: if the geometry is inadmissible.
    """
    _require_gate_up_blocked(tokens, rows, cols)
    return can_run_kernel(hidden_states)


def _partition_iota(device: torch.device) -> Tensor:
    """``[TILE_SIZE, 1]`` int32 holding ``0..TILE_SIZE-1``, the kernels' ramp operand.

    Built on the host because this NKI build has no ``nl.arange`` or ``nl.iota``. It
    is a constant of the compiled graph.
    """
    return torch.arange(TILE_SIZE, dtype=torch.int32, device=device).reshape(
        TILE_SIZE, 1
    )


_SWIGLU_BOUND_COLUMNS = 3


def _swiglu_bound_operand(
    gate_upper: float | None, up_upper: float | None, device: torch.device
) -> Tensor:
    """The activation bounds as one column operand: gate upper, ``-up``, ``up``.

    A one-column tensor means no bounds, and the kernel then emits no bound
    instruction. A set limit travels as a column because a non-tensor argument is
    replaced with ``None`` when the graph is captured; an unset limit beside a set
    one is infinity, which bounds nothing.
    """
    if gate_upper is None and up_upper is None:
        return torch.full((TILE_SIZE, 1), 0.0, dtype=torch.float32, device=device)
    unbounded = float("inf")
    gate = unbounded if gate_upper is None else float(gate_upper)
    up = unbounded if up_upper is None else float(up_upper)
    # ``torch.full`` rather than ``torch.tensor`` over a list: the latter materialises
    # a real tensor inside the capture backend's fake mode.
    return torch.cat(
        [
            torch.full((TILE_SIZE, 1), value, dtype=torch.float32, device=device)
            for value in (gate, -up, up)
        ],
        dim=1,
    )


def _require_routing(
    positions: int, block: int, experts: int, row_index: Tensor, expert_index: Tensor
) -> None:
    """Every condition the routed kernels' trace-time block arithmetic depends on.

    The block of a ``128``-row tile is ``m_tile // (block // TILE_SIZE)``, an integer
    only while ``block`` is a multiple of ``TILE_SIZE`` and ``positions`` a multiple
    of ``block``. A remainder would silently give a tile the previous block's expert.
    """
    problems: list[str] = []
    if block <= 0 or block % TILE_SIZE:
        problems.append(
            f"block={block} is not a positive multiple of TILE_SIZE={TILE_SIZE}; a "
            f"128-row tile must belong to exactly one block"
        )
    elif positions % block:
        problems.append(
            f"row_index carries {positions} positions, which is not a multiple of "
            f"block={block}"
        )
    else:
        blocks = positions // block
        if int(expert_index.shape[0]) != blocks:
            problems.append(
                f"expert_index carries {int(expert_index.shape[0])} entries against "
                f"{blocks} blocks of {block} positions; the mapping returns one "
                f"expert per block"
            )
    if experts < 1:
        problems.append(f"the bank carries {experts} experts")
    for name, tensor in (("row_index", row_index), ("expert_index", expert_index)):
        if tensor.dtype not in (torch.int32, torch.int64):
            problems.append(f"{name} must be an integer index, got {tensor.dtype}")
    if problems:
        raise MoeBlockwiseFp8Error(
            "the MoE routing refuses this mapping: " + "; ".join(problems)
        )


def moe_gate_up_blockwise_fp8(
    hidden_states: Tensor,
    weight_bank: Tensor,
    scale_bank: Tensor,
    row_index: Tensor,
    expert_index: Tensor,
    block: int,
) -> Tensor:
    """Routed gate/up projection: ``[T + 1, H]`` in, ``[P, 2*I]`` fp32 out.

    ``hidden_states`` is every token of the layer plus the appended padding row; the
    rows each block multiplies are chosen inside the kernel from ``row_index``.
    ``weight_bank`` is the whole expert bank, ``[E, H, 2, I]`` or ``[E, H, 2*I]``,
    contraction-major per expert as the checkpoint stores it, so one expert's weight
    reaches the kernel as an address and never as a copy. The result is
    pre-activation and in block order.
    """
    if hidden_states.ndim != 2:
        raise MoeBlockwiseFp8Error(
            f"hidden_states must be [T + 1, H]; got {tuple(hidden_states.shape)}"
        )
    if weight_bank.ndim not in (3, 4):
        raise MoeBlockwiseFp8Error(
            f"weight_bank must be [E, H, 2, I] or [E, H, 2*I]; got "
            f"{tuple(weight_bank.shape)}"
        )
    experts, rows = int(weight_bank.shape[0]), int(weight_bank.shape[1])
    fused_cols = int(weight_bank.shape[2]) * (
        int(weight_bank.shape[3]) if weight_bank.ndim == 4 else 1
    )
    if int(hidden_states.shape[1]) != rows:
        raise MoeBlockwiseFp8Error(
            f"weight_bank is contraction-major, so its axis 1 must equal "
            f"hidden_states.shape[1]; got bank {tuple(weight_bank.shape)} against "
            f"hidden {tuple(hidden_states.shape)}. A [E, 2*I, H] bank is the likely "
            f"cause -- this seam does not accept that orientation"
        )
    if fused_cols % GATE_UP_FUSION:
        raise MoeBlockwiseFp8Error(
            f"weight_bank has {fused_cols} fused columns, which is not a multiple "
            f"of GATE_UP_FUSION={GATE_UP_FUSION}; the gate half and the up half "
            f"must be equally wide"
        )
    cols = fused_cols // GATE_UP_FUSION
    positions = int(row_index.shape[0])
    _require_gate_up_blocked(positions, rows, cols)
    _require_routing(positions, block, experts, row_index, expert_index)

    expected = gate_up_kernel_scale_shape(rows, cols)
    if tuple(scale_bank.shape) != (experts,) + expected:
        raise MoeBlockwiseFp8Error(
            f"scale_bank has shape {tuple(scale_bank.shape)}, expected "
            f"{(experts,) + expected} at [H={rows}, I={cols}] over {experts} "
            f"experts. Build each expert's operand with "
            f"to_gate_up_kernel_scale_operand rather than by hand."
        )

    _count_gate_up_nki_dispatch()
    return wrap_nki(moe_gate_up_blockwise_fp8_kernel)[NUM_SHARDS](
        hidden_states.to(torch.bfloat16),
        # One ``128``-column block per row; ``[E, H, 2*I]`` is contiguous, so a view.
        weight_bank.reshape(-1, fused_cols),
        scale_bank.to(torch.float32).reshape(-1, expected[1]),
        row_index.to(torch.int32).reshape(-1, 1),
        expert_index.to(torch.int32).reshape(-1, 1),
        _partition_iota(row_index.device),
    )


def down_flat_scale_index(i_block: int, h_block: int, n_h_blocks: int) -> int:
    """Column of the down kernel's scale operand holding one block's scale.

    The order is the C-order flattening of the checkpoint's ``[I//128, H//128]``
    down grid: contraction-block major.
    """
    if n_h_blocks < 1:
        raise MoeBlockwiseFp8Error(f"n_h_blocks must be >= 1, got {n_h_blocks}")
    return i_block * n_h_blocks + h_block


def down_kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``[TILE_SIZE, n_blocks]``, the down scale operand shape for one expert.

    ``rows`` is ``I`` and ``cols`` is ``H``.
    """
    _require_gate_up_weight_blocked(rows, cols)
    n_blocks = (rows // GATE_UP_SCALE_BLOCK) * (cols // GATE_UP_SCALE_BLOCK)
    return (TILE_SIZE, n_blocks)


def to_down_kernel_scale_operand(
    checkpoint_scales: Tensor, rows: int, cols: int
) -> Tensor:
    """One expert's ``[I//128, H//128]`` down scales as the kernel operand.

    Raises:
        MoeBlockwiseFp8Error: if the grid is not the shape the extents imply. A
            mis-sized grid reshapes without error onto the wrong scales.
    """
    _require_gate_up_weight_blocked(rows, cols)
    expected = (rows // GATE_UP_SCALE_BLOCK, cols // GATE_UP_SCALE_BLOCK)
    if tuple(checkpoint_scales.shape) != expected:
        raise MoeBlockwiseFp8Error(
            f"mis-sized down scale grid: got {tuple(checkpoint_scales.shape)}, "
            f"expected {expected} at [I={rows}, H={cols}] and block "
            f"{GATE_UP_SCALE_BLOCK}."
        )
    flat = checkpoint_scales.to(torch.float32).reshape(1, -1)
    return flat.expand(TILE_SIZE, flat.shape[1]).contiguous()


def _require_down_blocked(tokens: int, rows: int, cols: int) -> None:
    """The down kernel's own extent conditions. ``rows`` is ``I``, ``cols`` is ``H``."""
    problems: list[str] = []
    if tokens <= 0 or tokens % TILE_SIZE:
        problems.append(
            f"B={tokens} is not a positive multiple of TILE_SIZE={TILE_SIZE}"
        )
    if rows <= 0 or rows % GATE_UP_SCALE_BLOCK:
        problems.append(
            f"I={rows} is not a positive multiple of "
            f"GATE_UP_SCALE_BLOCK={GATE_UP_SCALE_BLOCK}"
        )
    if cols <= 0 or cols % GATE_UP_SCALE_BLOCK:
        problems.append(
            f"H={cols} is not a positive multiple of "
            f"GATE_UP_SCALE_BLOCK={GATE_UP_SCALE_BLOCK}"
        )
    if GATE_UP_H_TILES_PER_BLOCK != 1:
        # The down kernel walks one contraction tile per scale block; a changed
        # constant fails here instead of silently dropping contraction tiles.
        problems.append(
            f"GATE_UP_H_TILES_PER_BLOCK={GATE_UP_H_TILES_PER_BLOCK}, and the down "
            f"kernel is written for exactly 1 contraction tile per scale block"
        )
    _refuse_gate_up(problems)


@dataclass
class _SwigluDispatchCounters:
    """Dispatch counts for the activation kernel."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


@dataclass
class _DownDispatchCounters:
    """Dispatch counts for the down kernel."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_SWIGLU_COUNTERS = _SwigluDispatchCounters()
_DOWN_COUNTERS = _DownDispatchCounters()


def reset_swiglu_dispatch_counters() -> None:
    """Zero the activation counters."""
    _SWIGLU_COUNTERS.nki_dispatch = 0
    _SWIGLU_COUNTERS.torch_fallback = 0


def swiglu_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the activation path; the second is 0."""
    return _SWIGLU_COUNTERS.nki_dispatch, _SWIGLU_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_swiglu_nki_dispatch() -> None:
    """Count one kernel dispatch off the traced graph; a traced store is a guard."""
    _SWIGLU_COUNTERS.nki_dispatch += 1


def reset_down_dispatch_counters() -> None:
    """Zero the down counters."""
    _DOWN_COUNTERS.nki_dispatch = 0
    _DOWN_COUNTERS.torch_fallback = 0


def down_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the down path; the second is 0."""
    return _DOWN_COUNTERS.nki_dispatch, _DOWN_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_down_nki_dispatch() -> None:
    """Count one kernel dispatch off the traced graph; a traced store is a guard."""
    _DOWN_COUNTERS.nki_dispatch += 1


@nki.jit
def moe_swiglu_transposed_kernel(gate_up, bounds):
    """``SiLU(clamp(gate)) * clamp(up)`` for one token block, as ``[I, B]`` fp32.

    Args:
        gate_up: ``[B, 2*I]`` pre-activation output, gate columns then up columns.
        bounds: ``[TILE_SIZE, _SWIGLU_BOUND_COLUMNS]`` fp32 (gate upper bound, up
            bound negated, up bound), or ``[TILE_SIZE, 1]`` when neither half is
            bounded. :func:`_swiglu_bound_operand` builds it; a bound is a value that
            no shape can carry, and a one-column operand elides every bound
            instruction at trace time.

    Returns:
        ``[I, B]`` fp32, transposed so the down projection can contract ``I`` on the
        partition axis without transposing anything itself: one on-chip
        ``nisa.nc_transpose`` per tile here, and the down kernel loads both operands
        straight from HBM.

    The asymmetry is the reference model's: ``gate`` is bounded from above only and
    ``up`` from both sides. The bounds are applied on the tile already loaded rather
    than in a torch pass over the whole pre-activation tensor. The activation is a
    separate kernel from the down projection because SiLU is transcendental: no torch
    reference reproduces a device sigmoid bit for bit, so keeping the matmul apart
    keeps its comparison exact, at the cost of one HBM round trip for the
    intermediate.
    """
    tokens, fused_cols = gate_up.shape
    n_tiles = tokens // TILE_SIZE
    _grid_ndim, n_prgs, prg_id = get_verified_program_sharding_info(
        "moe_swiglu_transposed", (0, 1), NUM_SHARDS
    )
    first_tile, end_tile = _program_block_range(
        "moe_swiglu_transposed", n_tiles, n_prgs, prg_id
    )
    i_extent = fused_cols // GATE_UP_FUSION
    # A one-column operand means no bounds, and no bound instruction is emitted.
    bounded_config = bounds.shape[1] == _SWIGLU_BOUND_COLUMNS
    out = nl.ndarray((i_extent, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
    if bounded_config:
        bounds_sb = nl.load(bounds[0:TILE_SIZE, 0:_SWIGLU_BOUND_COLUMNS])
    # The token tile is the dynamic axis. The input leads with it; the result carries
    # it on the free axis because this kernel returns ``[I, B]``.
    gate_up_b = gate_up.reshape((n_tiles, TILE_SIZE, fused_cols))
    out_b = out.reshape((i_extent, n_tiles, TILE_SIZE))
    column = [[fused_cols, TILE_SIZE], [1, GATE_UP_SCALE_BLOCK]]

    def activate_tile(at_tile):
        for i_block in range(i_extent // GATE_UP_SCALE_BLOCK):
            i0 = i_block * GATE_UP_SCALE_BLOCK
            gate = _loaded(
                gate_up_b.ap(
                    pattern=column,
                    offset=i0,
                    scalar_offset=at_tile,
                    indirect_dim=0,
                ),
                TILE_SIZE,
                GATE_UP_SCALE_BLOCK,
                nl.float32,
            )
            up = _loaded(
                gate_up_b.ap(
                    pattern=column,
                    offset=i_extent + i0,
                    scalar_offset=at_tile,
                    indirect_dim=0,
                ),
                TILE_SIZE,
                GATE_UP_SCALE_BLOCK,
                nl.float32,
            )
            # Bounds on the tiles just loaded, as ``minimum``/``maximum`` against a
            # per-partition column (see :func:`_swiglu_bound_operand`).
            if bounded_config:
                bounded = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=bounded,
                    data=gate,
                    op0=nl.minimum,
                    operand0=bounds_sb[0:TILE_SIZE, 0:1],
                )
                gate = bounded
                # Two calls rather than one fused pair: a ``[P, 1]`` column is known to
                # work as ``operand0``, not as ``operand1``.
                floored = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=floored,
                    data=up,
                    op0=nl.maximum,
                    operand0=bounds_sb[0:TILE_SIZE, 1:2],
                )
                bounded_up = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=bounded_up,
                    data=floored,
                    op0=nl.minimum,
                    operand0=bounds_sb[0:TILE_SIZE, 2:3],
                )
                up = bounded_up
            # sigmoid is one activation-engine op, not a composition.
            squashed = _gate_up_sbuf()
            nisa.activation(dst=squashed, data=gate, op=nl.sigmoid)
            # SiLU(gate) = gate * sigmoid(gate). The three-operand form with ``1.0`` as
            # the identity scalar stands in for an elementwise tensor-tensor multiply.
            silu = _gate_up_sbuf()
            nisa.scalar_tensor_tensor(
                dst=silu,
                data=gate,
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=squashed,
            )
            gated = _gate_up_sbuf()
            nisa.scalar_tensor_tensor(
                dst=gated,
                data=silu,
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=up,
            )
            # [B, I] -> [I, B], the orientation the down projection contracts on.
            transposed = _gate_up_psum()
            nisa.nc_transpose(dst=transposed, data=gated)
            out_sb = _gate_up_sbuf()
            nisa.tensor_copy(dst=out_sb, src=transposed)
            _stored(
                out_b.ap(
                    pattern=[[tokens, GATE_UP_SCALE_BLOCK], [1, TILE_SIZE]],
                    offset=i0 * tokens,
                    scalar_offset=at_tile,
                    indirect_dim=1,
                ),
                out_sb,
            )

    nl.fori_loop(first_tile, end_tile, activate_tile)
    return out


def moe_swiglu_transposed(
    gate_up: Tensor,
    gate_upper: float | None = None,
    up_upper: float | None = None,
) -> Tensor:
    """SwiGLU activation: ``[B, 2*I]`` in, ``[I, B]`` fp32 out.

    The two bounds are the reference model's, asymmetric: an upper bound on ``gate``
    and a symmetric one on ``up``. ``None`` means no bound and no instruction for it.
    There is no torch route; an inadmissible shape raises. The transposed return is
    part of the contract.
    """
    if gate_up.ndim != 2:
        raise MoeBlockwiseFp8Error(
            f"gate_up must be [B, 2*I]; got {tuple(gate_up.shape)}"
        )
    tokens, fused_cols = int(gate_up.shape[0]), int(gate_up.shape[1])
    if fused_cols % GATE_UP_FUSION:
        raise MoeBlockwiseFp8Error(
            f"gate_up has {fused_cols} columns, which is not a multiple of "
            f"GATE_UP_FUSION={GATE_UP_FUSION}"
        )
    problems: list[str] = []
    if tokens <= 0 or tokens % TILE_SIZE:
        problems.append(
            f"B={tokens} is not a positive multiple of TILE_SIZE={TILE_SIZE}"
        )
    half = fused_cols // GATE_UP_FUSION
    if half <= 0 or half % GATE_UP_SCALE_BLOCK:
        problems.append(
            f"I={half} is not a positive multiple of "
            f"GATE_UP_SCALE_BLOCK={GATE_UP_SCALE_BLOCK}"
        )
    _refuse_gate_up(problems)

    _count_swiglu_nki_dispatch()
    return wrap_nki(moe_swiglu_transposed_kernel)[NUM_SHARDS](
        gate_up.to(torch.float32),
        _swiglu_bound_operand(gate_upper, up_upper, gate_up.device),
    )


@nki.jit
def moe_down_blockwise_fp8_kernel(
    intermediate_t,
    weight_bank,
    scale_bank,
    affinity_bank,
    row_index,
    expert_index,
    iota,
):
    """``out[P, H] = (intermediate[P, I] @ dequantise(bank[expert])) * affinity``.

    Args:
        intermediate_t: ``[I, P]``, the activation kernel's transposed output for
            every block. The token axis is the free axis, so a block's columns are a
            trace-time slice and no row gather is needed here.
        weight_bank: the whole bank ``[E, I, H]`` fp8-e4m3, viewed by the caller as
            ``[E * I * n_h_blocks, GATE_UP_SCALE_BLOCK]``, a view and not a copy.
        scale_bank: ``[E * TILE_SIZE, n_blocks]`` fp32, the per-expert operands of
            :func:`to_down_kernel_scale_operand` stacked.
        affinity_bank: ``[(T + 1) * E, 1]`` fp32, token-major masked affinities with
            the padding token's zeros appended.
        row_index: ``[P, 1]`` int32, ``token_position_to_id``, ``-1`` for padding.
        expert_index: ``[n_blocks_total, 1]`` int32, ``block_to_expert``.
        iota: ``[TILE_SIZE, 1]`` int32 ramp; see :func:`_row_iota`.

    Returns:
        ``[P, H]`` fp32 in block order; the caller scatters back with one
        ``index_add``.

    The affinity index ``row * E_local + expert`` is computed on device from the
    block's expert scalar, so one gather replaces a torch advanced index over two
    tensors. The affinity is applied as a ``tensor_scalar`` ``operand0``, a
    per-partition column broadcast along the free axis, because the accumulator
    carries tokens on partitions. One contraction tile per scale block at this
    granularity, so ``accumulate`` is False on every matmul and each block's scale
    multiplies exactly its own partial sum; the wrapper refuses the geometry if that
    quotient is not one.
    """
    n_blocks = expert_index.shape[0]
    _grid_ndim, n_prgs, prg_id = get_verified_program_sharding_info(
        "moe_down_blockwise_fp8", (0, 1), NUM_SHARDS
    )
    first_block, end_block = _program_block_range(
        "moe_down_blockwise_fp8", n_blocks, n_prgs, prg_id
    )
    # Only tensors cross the wrapper, so every extent is read from an operand's shape.
    i_extent, positions = intermediate_t.shape
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    # One scale column per (I block, H block), so the width over the I blocks is H.
    n_h_blocks = scale_bank.shape[1] // n_i_blocks
    h_extent = n_h_blocks * GATE_UP_SCALE_BLOCK
    # One ``TILE_SIZE``-tall scale operand per expert; the count is the affinity row
    # stride.
    n_experts = scale_bank.shape[0] // TILE_SIZE
    block = positions // expert_index.shape[0]
    tiles_per_block = block // TILE_SIZE
    # The padding row is the appended one; the wrapper checks the affinity bank's
    # length against ``T`` so this quotient cannot move it.
    pad_row = affinity_bank.shape[0] // n_experts - 1

    out = nl.ndarray((positions, h_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    ramp = _row_iota(iota, TILE_SIZE)
    # The block is the dynamic axis. The output and the routing column lead with it;
    # the intermediate carries it on the free axis because the activation returns
    # ``[I, B]``.
    row_index_b = row_index.reshape((n_blocks, block, 1))
    intermediate_b = intermediate_t.reshape((i_extent, n_blocks, block))
    out_b = out.reshape((n_blocks, block, h_extent))

    def project_block(at_block):
        # The expert is the block's property, so its broadcast and the scale gather
        # that reads it stay outside the tile loop.
        expert = _broadcast_row(expert_index, 0, TILE_SIZE, at_block=at_block)
        scale_sb = _gathered(
            scale_bank,
            _bank_rows(TILE_SIZE, expert, ramp, TILE_SIZE, 0, 1),
            TILE_SIZE,
            n_i_blocks * n_h_blocks,
            nl.float32,
        )
        for token_tile in range(tiles_per_block):
            t0 = token_tile * TILE_SIZE
            # Loaded per token tile: a token is a partition and the partition axis
            # holds 128, so a whole-column load would bound this kernel to one tile.
            # The resolved row is the same padding-aware index the gate/up kernel
            # gathered its tokens with.
            affinity_sb = _gathered(
                affinity_bank,
                _affinity_rows(
                    TILE_SIZE,
                    _padding_resolved(
                        TILE_SIZE,
                        _dense_column(row_index_b, TILE_SIZE, t0, at_block=at_block),
                        pad_row,
                    ),
                    expert,
                    n_experts,
                ),
                TILE_SIZE,
                1,
                nl.float32,
            )
            for h_block in range(n_h_blocks):
                h0 = h_block * GATE_UP_SCALE_BLOCK
                acc = _gate_up_sbuf()
                for i_block in range(n_i_blocks):
                    i0 = i_block * GATE_UP_SCALE_BLOCK
                    psum = _gate_up_psum()
                    inter_tile = _loaded(
                        intermediate_b.ap(
                            pattern=[[positions, TILE_SIZE], [1, TILE_SIZE]],
                            offset=i0 * positions + t0,
                            scalar_offset=at_block,
                            indirect_dim=1,
                        ),
                        TILE_SIZE,
                        TILE_SIZE,
                        nl.bfloat16,
                    )
                    stride, row0, step, column = down_bank_address(i_extent, i0, h_block)
                    w_tile = _gathered(
                        weight_bank,
                        _bank_rows(TILE_SIZE, expert, ramp, stride, row0, step),
                        TILE_SIZE,
                        GATE_UP_SCALE_BLOCK,
                        nl.bfloat16,
                        h_extent,
                        column,
                    )
                    nisa.nc_matmul(
                        dst=psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=inter_tile,
                        moving=w_tile,
                        accumulate=False,
                    )
                    flat = i_block * n_h_blocks + h_block
                    if i_block == 0:
                        nisa.tensor_scalar(
                            dst=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            data=psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            op0=nl.multiply,
                            operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        )
                    else:
                        nisa.scalar_tensor_tensor(
                            dst=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            data=psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                            op0=nl.multiply,
                            operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                            op1=nl.add,
                            operand1=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        )
                # The affinity is a per-partition column and the partition axis is the
                # token axis, so one value per token is exactly this operand's shape.
                scaled = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=scaled[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    data=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    op0=nl.multiply,
                    operand0=affinity_sb[0:TILE_SIZE, 0:1],
                )
                _stored(
                    out_b.ap(
                        pattern=[[h_extent, TILE_SIZE], [1, GATE_UP_SCALE_BLOCK]],
                        offset=t0 * h_extent + h0,
                        scalar_offset=at_block,
                        indirect_dim=0,
                    ),
                    scaled[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                )

    nl.fori_loop(first_block, end_block, project_block)
    return out


def can_run_moe_down_blockwise_fp8(
    intermediate_t: Tensor, tokens: int, rows: int, cols: int
) -> bool:
    """Whether the NKI route is available for an admissible down geometry.

    Raises:
        MoeBlockwiseFp8Error: if the geometry is inadmissible.
    """
    _require_down_blocked(tokens, rows, cols)
    return can_run_kernel(intermediate_t)


def moe_down_blockwise_fp8(
    intermediate_t: Tensor,
    weight_bank: Tensor,
    scale_bank: Tensor,
    affinity_bank: Tensor,
    row_index: Tensor,
    expert_index: Tensor,
    block: int,
    tokens: int,
) -> Tensor:
    """Routed down projection: ``[I, P]`` in, ``[P, H]`` fp32 out, block order.

    ``weight_bank`` is ``[E, I, H]``, contraction-major per expert; ``affinity_bank``
    is the routing map's flat ``[(T + 1) * E, 1]`` masked affinities. Both are
    addressed inside the kernel from each block's expert, so neither is sliced here.
    """
    for name, tensor, rank in (
        ("intermediate_t", intermediate_t, 2),
        ("weight_bank", weight_bank, 3),
        ("affinity_bank", affinity_bank, 2),
    ):
        if tensor.ndim != rank:
            raise MoeBlockwiseFp8Error(
                f"{name} must have rank {rank}; got shape {tuple(tensor.shape)}"
            )
    rows, positions = int(intermediate_t.shape[0]), int(intermediate_t.shape[1])
    experts, bank_rows, cols = (int(extent) for extent in weight_bank.shape)
    if bank_rows != rows:
        raise MoeBlockwiseFp8Error(
            f"weight_bank is contraction-major, so its axis 1 must equal "
            f"intermediate_t.shape[0]; got bank {tuple(weight_bank.shape)} against "
            f"intermediate {tuple(intermediate_t.shape)}. A [E, H, I] bank is the "
            f"likely cause -- this seam does not accept that orientation"
        )
    _require_down_blocked(positions, rows, cols)
    _require_routing(positions, block, experts, row_index, expert_index)

    expected = down_kernel_scale_shape(rows, cols)
    if tuple(scale_bank.shape) != (experts,) + expected:
        raise MoeBlockwiseFp8Error(
            f"scale_bank has shape {tuple(scale_bank.shape)}, expected "
            f"{(experts,) + expected} at [I={rows}, H={cols}] over {experts} "
            f"experts. Build each expert's operand with "
            f"to_down_kernel_scale_operand rather than by hand."
        )
    if tuple(affinity_bank.shape) != ((tokens + 1) * experts, 1):
        raise MoeBlockwiseFp8Error(
            f"affinity_bank must be [(T + 1) * E_local, 1] = "
            f"{((tokens + 1) * experts, 1)} at T={tokens} and E_local={experts}; got "
            f"{tuple(affinity_bank.shape)}. It is the mapping's own flat token-major "
            f"emission with the padding token's zeros appended, not a per-block "
            f"slice, and its length is checked because the padding row is counted "
            f"from T rather than from this tensor."
        )

    _count_down_nki_dispatch()
    return wrap_nki(moe_down_blockwise_fp8_kernel)[NUM_SHARDS](
        intermediate_t.to(torch.float32),
        weight_bank.reshape(-1, cols),
        scale_bank.to(torch.float32).reshape(-1, expected[1]),
        affinity_bank.to(torch.float32).reshape(-1, 1),
        row_index.to(torch.int32).reshape(-1, 1),
        expert_index.to(torch.int32).reshape(-1, 1),
        _partition_iota(row_index.device),
    )


def _unwrap_nki(obj: Any) -> Any:
    """The plain Python function behind an ``nki.jit`` object, else ``obj``.

    ``nki.jit`` stores the decorated function on ``.func``, and the wrapper's
    ``__module__``/``__qualname__`` are not the kernel's. ``__wrapped__`` is tried
    second because ``functools.wraps`` sets it.
    """
    for attribute in ("func", "__wrapped__"):
        inner = getattr(obj, attribute, None)
        if inner is not None:
            return inner
    return obj


def _function_ast(obj: Any) -> tuple[Any, ast.FunctionDef]:
    """``(function, its def node)``, parsed from the function's own source."""
    fn = _unwrap_nki(obj)
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:  # no source: editable install broken
        raise MoeBlockwiseFp8Error(
            f"cannot read the source of {getattr(fn, '__name__', fn)!r}, so the "
            f"seam's dispatch target cannot be derived: {exc}"
        ) from exc
    name = getattr(fn, "__name__", None)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return fn, node
    raise MoeBlockwiseFp8Error(
        f"no `def {name}` found in the source read for it, so the seam's "
        f"dispatch target cannot be derived"
    )


def _resolved(fn: Any, node: ast.expr, what: str) -> Any:
    """Resolve a bare name in ``fn``'s source against ``fn``'s live globals.

    Reading ``__globals__`` rather than this module's namespace means a rebound
    module global changes the answer too, not only an edited call site.
    """
    if not isinstance(node, ast.Name):
        raise MoeBlockwiseFp8Error(
            f"{what} is not a plain name ({type(node).__name__}), so the object "
            f"it denotes cannot be resolved"
        )
    try:
        return fn.__globals__[node.id]
    except KeyError as exc:
        raise MoeBlockwiseFp8Error(
            f"{what} is {node.id!r}, which is not bound in "
            f"{fn.__module__!r}; the seam's dispatch target cannot be resolved"
        ) from exc


def _seam_wrapped_object() -> Any:
    """The object :func:`blockwise_fp8_moe` hands to ``wrap_nki``."""
    fn, tree = _function_ast(blockwise_fp8_moe)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "wrap_nki"
    ]
    if len(calls) != 1:
        raise MoeBlockwiseFp8Error(
            f"the seam makes {len(calls)} `wrap_nki(...)` calls, expected exactly "
            f"one; which object it wraps is therefore ambiguous"
        )
    if len(calls[0].args) != 1:
        raise MoeBlockwiseFp8Error(
            f"the seam's `wrap_nki(...)` takes {len(calls[0].args)} positional "
            f"arguments, expected exactly one"
        )
    return _resolved(fn, calls[0].args[0], "the seam's `wrap_nki` argument")


def _shim_forward_target() -> Any:
    """The object the wrapped shim returns the result of calling."""
    shim = _seam_wrapped_object()
    fn, tree = _function_ast(shim)
    returns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    ]
    if len(returns) != 1:
        raise MoeBlockwiseFp8Error(
            f"{getattr(fn, '__name__', fn)!r} returns the result of "
            f"{len(returns)} calls, expected exactly one; its forward target is "
            f"therefore ambiguous"
        )
    call = returns[0].value
    assert isinstance(call, ast.Call)
    return _resolved(fn, call.func, "the shim's forward target")


def seam_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the object the vendor route hands to ``wrap_nki``.

    This is the shim, not the vendor kernel. Both identity readings follow the live
    call chain from :func:`blockwise_fp8_moe` (its ``wrap_nki`` argument, then the
    shim's forward target) rather than this module's import, so substituting either
    hop changes the reading instead of leaving it unchanged.
    """
    obj = _unwrap_nki(_seam_wrapped_object())
    return obj.__module__, obj.__qualname__


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the NKI kernel the vendor route forwards to.

    Raises:
        MoeBlockwiseFp8Error: if the chain cannot be derived. There is deliberately
            no fallback to this module's import of the kernel.
    """
    target = _unwrap_nki(_shim_forward_target())
    return target.__module__, target.__qualname__


def _wrapped_object_of(seam: Any, what: str) -> Any:
    """The object ``seam``'s single ``wrap_nki(...)`` call wraps, resolved live."""
    fn, tree = _function_ast(seam)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "wrap_nki"
    ]
    if len(calls) != 1:
        raise MoeBlockwiseFp8Error(
            f"{what} makes {len(calls)} `wrap_nki(...)` calls, expected exactly "
            f"one; which object it wraps is therefore ambiguous"
        )
    if len(calls[0].args) != 1:
        raise MoeBlockwiseFp8Error(
            f"{what}'s `wrap_nki(...)` takes {len(calls[0].args)} positional "
            f"arguments, expected exactly one"
        )
    return _resolved(fn, calls[0].args[0], f"{what}'s `wrap_nki` argument")


def gate_up_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel the gate/up wrapper dispatches to.

    The unwrap matters: ``nki.jit`` returns a wrapper whose ``__module__`` is
    ``nki``'s, so reading the decorated object directly reports the same answer for
    a kernel defined here and one imported from ``nkilib``.

    Raises:
        MoeBlockwiseFp8Error: if the chain cannot be derived. There is deliberately
            no fallback to this module's own kernel name.
    """
    obj = _unwrap_nki(_wrapped_object_of(moe_gate_up_blockwise_fp8, "the gate/up seam"))
    return obj.__module__, obj.__qualname__


def swiglu_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel the activation wrapper dispatches to."""
    obj = _unwrap_nki(_wrapped_object_of(moe_swiglu_transposed, "the activation seam"))
    return obj.__module__, obj.__qualname__


def down_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel the down wrapper dispatches to."""
    obj = _unwrap_nki(_wrapped_object_of(moe_down_blockwise_fp8, "the down seam"))
    return obj.__module__, obj.__qualname__
