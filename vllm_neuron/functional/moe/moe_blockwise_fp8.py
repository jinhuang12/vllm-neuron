# SPDX-License-Identifier: Apache-2.0
"""MoE-half block-quantised fp8 matmul: the ADAPT of ``nkilib``'s ``bwmm_shard_on_I``.

`inc-glm53f-025`. This module is the campaign's block-quant MoE path. It is
**kernel-class** under P13: the per-token expert matmul with block
dequantisation folded in is the model's dominant FLOP path, so the arithmetic
runs in NKI. The torch code here is the CPU oracle and the
constraint-violation fallback -- the two roles the plan's substrate register
admits (`design/increment-plan.md` §4) -- and never the shipped
implementation.

ADAPT, not SCRATCH
------------------
``nkilib`` already provides this kernel at ``256 x 256`` block granularity:
``nkilib.core.moe.moe_cte.bwmm_shard_on_I.blockwise_mm_baseline_shard_intermediate``,
whose ``is_block_quant=True`` path is the only block-quant member of the
``bwmm_*`` family (1 of 10 files; measured in
``increments/wp6-scale-consumer-geometry.md``). GLM-5.3-Flash ships
``[128, 128]`` checkpoint scales, so the granularity gap is bridged on the
host by :mod:`vllm_neuron.functional.moe.blockwise_fp8_retile`
(`inc-glm53f-024`), and this module adapts the vendor kernel to the retiled
layout through a seam this repository owns. Nothing here re-authors kernel
numerics; the NKI member is called, not replaced.

The scale layout this module consumes -- SETTLED, not assumed
------------------------------------------------------------
The producer emits a flat ``(E, n_blocks * TILE_SIZE)`` tensor
(:func:`~vllm_neuron.functional.moe.blockwise_fp8_retile.consumer_scale_shape`).
The kernel and its torch reference both consume a *logical* tensor whose
**last** axis is ``TILE_SIZE``:

* ``gate_up_proj_scale`` -- ``[E, H//256, 2, I_TP//256, TILE_SIZE]``
* ``down_proj_scale``    -- ``[E, I_TP//256, H//256, TILE_SIZE]``

so the bridge is a plain C-order reshape and the ``TILE_SIZE`` replicas are
**contiguous**. Three independent instruments in the installed substrate agree
(``nkilib`` sha256 ``b2b5f7530f7bb46aad0f0e871343b7fdae6b4509712f163a9b3df2d8769c935d``):

1. the DMA access pattern ``pattern=[[1, TILE_SIZE], [TILE_SIZE, k]]`` at
   ``bwmm_shard_on_I.py:1135`` (gate/up) and ``:2007`` (down) -- the partition
   axis walks with stride **1**, the block axis with stride **TILE_SIZE**;
2. the host offset arithmetic at ``:1131`` and ``:2001``-``:2003``, which
   advances by ``TILE_SIZE`` per ``256``-block;
3. ``moe_cte_torch.py:195`` and ``:275``, where the vendor torch reference
   slices ``[..., 0]`` off a trailing ``TILE_SIZE`` axis.

A partition-major layout is therefore **refuted**, not merely unchosen, and
:func:`to_kernel_scale_layout` is the one place the bridge is written.

Route
-----
Acceptance is Tier N: the NKI simulator, reached through this module's own
:func:`blockwise_fp8_moe` seam
(``wrap_nki -> NKIHOPCaller -> HOP -> DispatchKey.CPU ->
nki.simulator.simulate_kernel``). The seam counts its dispatches so a silent
fall back to the torch path cannot be read as a kernel run: under a torch
oracle on both sides a numeric comparison passes green regardless of which
route ran, which is why the counters below are acceptance criteria and not
diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate,
)
from nkilib.core.moe.moe_cte.bwmm_shard_on_I_torch import (
    blockwise_mm_baseline_shard_intermediate_torch_ref,
)

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    DOWN,
    GATE_UP,
    TILE_SIZE,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Logical-core count the vendor kernel shards ``I`` over. ``NUM_SHARDS`` is not
#: a parameter: the kernel reads it as ``nl.num_programs(axes=0)``
#: (``bwmm_shard_on_I.py:80``), i.e. from the SPMD launch grid, and
#: ``:628`` refuses anything but ``2`` on the dynamic-control-flow path
#: ("shard-on-I with dynamic control flow only work on TRN2"). The campaign's
#: target is trn2 at LNC2, so the grid is fixed here rather than exposed.
NUM_SHARDS = 2

#: ``H`` bounds, from the kernel's own compatibility asserts at
#: ``bwmm_shard_on_I.py:668`` (``512 <= H <= 8192``).
MIN_HIDDEN = 512
MAX_HIDDEN = 8192

#: ``H`` must additionally be a multiple of ``PSUM_SIZE`` (``512``), stated in
#: the kernel docstring at ``:204``; ``moe_cte_utils.py:59`` is the defining
#: assignment of ``PSUM_SIZE``.
PSUM_SIZE = 512

#: ``GATE_UP`` and ``DOWN`` are re-exported from the producer rather than
#: re-declared: the two modules must agree on the selector string, and one
#: definition cannot drift from itself.
__all__ = [
    "BLOCK_QUANT_SIZE",
    "DOWN",
    "GATE_UP",
    "NUM_SHARDS",
    "TILE_SIZE",
    "MoeBlockwiseFp8Error",
    "blockwise_fp8_moe",
    "blockwise_fp8_moe_torch_oracle",
    "can_run_blockwise_fp8_moe",
    "dispatch_counters",
    "kernel_identity",
    "kernel_scale_shape",
    "reset_dispatch_counters",
    "to_kernel_scale_layout",
]


class MoeBlockwiseFp8Error(ValueError):
    """A configuration this module refuses, named rather than silently coerced.

    Raised in preference to letting the vendor kernel trap: ``kernel_assert``
    fires at trace time with a message this repository does not control, and a
    refusal that names the offending extent is what a caller can act on.
    """


# --------------------------------------------------------------------------- #
# The layout bridge -- the single place the settled byte order is written.      #
# --------------------------------------------------------------------------- #
def kernel_scale_shape(
    num_experts: int, rows: int, cols: int, projection: str = DOWN
) -> tuple[int, ...]:
    """The *logical* scale shape the kernel and its torch reference consume.

    ``rows`` is the ``H`` axis and ``cols`` the ``I_TP`` axis of one expert's
    weight, both global (unsharded): the host tensor is sized on the full
    ``I_TP`` even though the device buffer is sharded
    (``bwmm_shard_on_I.py:1127`` uses ``I_blocks_total``, ``:1126`` uses
    ``I_blocks_sharded``).

    Returns the shape whose **last** axis is ``TILE_SIZE`` -- see the module
    docstring for the three instruments that settle that.
    """
    _require_blocked(rows, cols)
    if num_experts < 1:
        raise MoeBlockwiseFp8Error(f"num_experts must be >= 1, got {num_experts}")
    h_256 = rows // BLOCK_QUANT_SIZE
    i_256 = cols // BLOCK_QUANT_SIZE
    if projection == DOWN:
        # :1987 comment; :1994-:1995 allocation.
        return (num_experts, i_256, h_256, TILE_SIZE)
    if projection == GATE_UP:
        # :1127 allocation; the 2 is the gate/up fusion.
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
    """Reshape the producer's flat scale tensor into the kernel's logical view.

    ``consumer_scales`` is what
    :func:`~vllm_neuron.functional.moe.blockwise_fp8_retile.retile_block_scales`
    emits: ``(E, n_blocks * TILE_SIZE)``, flat.

    This is a **C-order** reshape, and that is the settled byte order rather
    than a default taken for convenience. Under C order the producer's flat
    offset for block ``b`` and replica ``t`` is ``b * TILE_SIZE + t``, which is
    exactly the offset the kernel's DMA reads (``:1131`` + ``:1135``, ``:2001``
    + ``:2007``). The producer's own
    :func:`~vllm_neuron.functional.moe.blockwise_fp8_retile.flat_scale_index`
    returns ``b``, so no index arithmetic is repeated here.

    Raises:
        MoeBlockwiseFp8Error: if ``consumer_scales`` does not have the flat
            shape the declared extents imply. Checked rather than trusted: a
            reshape of a wrongly sized tensor either raises deep inside torch
            or, worse, succeeds with a different block-to-scale assignment.
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
    """Every extent condition the kernel's own asserts impose, checked here.

    Sources, each an assert in ``bwmm_shard_on_I.py``: ``:668`` H range,
    ``:669`` ``H % TILE_SIZE``, ``:680`` ``H % 256``, ``:670`` ``I_TP % 16``,
    ``:681`` ``I_TP % 256``, ``:672`` ``I_TP % NUM_SHARDS``; plus the
    docstring's ``H % PSUM_SIZE`` at ``:204``.
    """
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
    # The scale index divides I_TP_sharded (:1123, :1381, :1988) while the
    # asserts only constrain I_TP (:681), so an odd multiple of 256 truncates
    # the sharded block extent at NUM_SHARDS=2 and the index walks off its
    # block row. Refused here because the kernel does not refuse it.
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


# --------------------------------------------------------------------------- #
# The route seam and its counters.                                             #
# --------------------------------------------------------------------------- #
@dataclass
class _DispatchCounters:
    """What route actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` seam;
    ``torch_fallback`` counts entries into the torch path. They are separate
    counters, not one flag, so "the kernel ran" and "the fallback did not run"
    are two independent readings and a test can require both.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


def can_run_blockwise_fp8_moe(
    hidden_states: Tensor, rows: int, cols: int
) -> bool:
    """Is the NKI route available *and* admissible for this geometry?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_blocked`
    answers "does the kernel accept these extents". A geometry the kernel
    refuses is an error rather than a silent fallback -- falling back would
    ship torch for kernel-class work.

    Raises:
        MoeBlockwiseFp8Error: if the geometry is inadmissible.
    """
    _require_blocked(rows, cols)
    return can_run_kernel(hidden_states)


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
    """Block-quantised fp8 MoE matmul. The seam the route predicate counts.

    Args:
        hidden_states: ``[T+1, H]``. The trailing row is the padding-token slot
            (``bwmm_shard_on_I.py:157``).
        expert_affinities_masked: ``[(T+1) * E, 1]``.
        gate_up_proj_weight: ``[E, H, 2, I_TP]``, fp8.
        down_proj_weight: ``[E, I_TP, H]``, fp8.
        block_size: tokens per block, a multiple of ``256`` (``:667``).
        token_position_to_id: ``[N * B]``, int32.
        block_to_expert: ``[N, 1]``, int32.
        gate_up_proj_scale: logical ``[E, H//256, 2, I_TP//256, TILE_SIZE]``.
        down_proj_scale: logical ``[E, I_TP//256, H//256, TILE_SIZE]``.
        **kernel_kwargs: forwarded verbatim to the vendor kernel and, on the
            fallback path, to its torch reference, so the two routes cannot
            drift apart in configuration.

    Returns:
        ``[T+1, H]`` output hidden states.
    """
    rows = hidden_states.shape[-1]
    cols = down_proj_weight.shape[-2]

    if not can_run_blockwise_fp8_moe(hidden_states, rows, cols):
        _COUNTERS.torch_fallback += 1
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

    _COUNTERS.nki_dispatch += 1
    # `wrap_nki(...)[NUM_SHARDS]` is the SPMD launch grid, not an output arity:
    # the kernel reads `nl.num_programs(axes=0)` as NUM_SHARDS (:80).
    wrapped = wrap_nki(blockwise_mm_baseline_shard_intermediate)
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
    """The vendor kernel's *own* torch reference, block-quant path.

    Sourced from ``nkilib`` rather than written here on purpose: an oracle this
    repository authored would be this module's own arithmetic restated, and a
    comparison against it could not detect a shared misreading of the kernel's
    scale layout. The reference used is
    ``bwmm_shard_on_I_torch.blockwise_mm_baseline_shard_intermediate_torch_ref``,
    whose ``is_block_quant`` path is implemented at ``moe_cte_torch.py:193``
    (gate/up) and ``:273`` (down).

    The wrapper is called rather than ``_moe_cte_torch_ref_impl`` directly
    because the two disagree on a default that changes the numbers: the
    wrapper defaults ``expert_affinities_scaling_mode`` to ``PRE_SCALE``,
    matching the kernel (``bwmm_shard_on_I.py:122``), while the impl defaults
    to ``POST_SCALE`` (``moe_cte_torch.py:49``).

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


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the adapted NKI member, read off the object.

    Exposed so a test can assert *which* kernel the seam dispatches to without
    re-importing it, and so a substitution shows up as a changed reading rather
    than as silence.
    """
    func: Optional[Any] = getattr(
        blockwise_mm_baseline_shard_intermediate, "func", None
    )
    target = func if func is not None else blockwise_mm_baseline_shard_intermediate
    return target.__module__, target.__qualname__
