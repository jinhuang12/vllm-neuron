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
layout through a seam this repository owns. **That is the `-025` limb, and on
it nothing re-authors kernel numerics: the NKI member is called, not
replaced.** `inc-glm53f-113a` adds a SECOND, INDEPENDENT limb that does author
its own kernel, at the checkpoint's own granularity; the section that carries
it says why, and the two limbs share this module without sharing a route.

The `-113a` gate/up limb, in one paragraph
-----------------------------------------
`inc-glm53f-113a` authors this campaign's own NKI kernel for the gate/up
projection of one expert's token block, indexing the checkpoint's ``128 x 128``
scales directly, and returns the **pre-activation** fp32 result. It is a
separate seam with its own counters and its own identity reading, and this
commit changes NO byte of the route above: :func:`blockwise_fp8_moe` still
enters the vendor member, because the block's overall return also needs the
down projection, the activation and the affinity scaling, which are
`inc-glm53f-113b`. The single sentence "the vendor member stops being called on
the block-quant limb" is therefore `-113b`'s to make true, not this
increment's, and nothing here pretends otherwise.

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

#: The `-113a` limb's scale-block extent: the granularity the CHECKPOINT itself
#: stores, one fp32 scale per ``128 x 128`` block of the weight. It is declared
#: as ``TILE_SIZE`` rather than as a literal ``128`` and it is deliberately NOT
#: ``BLOCK_QUANT_SIZE``: that ``256`` is the vendor limb's consumer granularity
#: and is not this kernel's business. `inc-glm53f-026`'s dense module carries the
#: same declaration for the same reason (``blockwise_fp8_mm.py:109``).
GATE_UP_SCALE_BLOCK = TILE_SIZE

#: Contraction tiles per scale block, re-derived from the two extents rather than
#: written as ``1``, so the pair cannot drift from the quotient. At this
#: granularity the quotient IS ``1``, which is why no partial sum in the kernel
#: below ever spans two different scales.
GATE_UP_H_TILES_PER_BLOCK = GATE_UP_SCALE_BLOCK // TILE_SIZE

#: The fusion width of the gate/up weight: gate and up, in that order, on one
#: axis. Named rather than written as ``2`` at four sites.
GATE_UP_FUSION = 2

#: ``GATE_UP`` and ``DOWN`` are re-exported from the producer rather than
#: re-declared: the two modules must agree on the selector string, and one
#: definition cannot drift from itself.
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
    # The vendor's own ``skip_dma: SkipMode = SkipMode()`` becomes these two
    # flat booleans; the object is rebuilt in the kernel body below.
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

    The kernel this seam dispatches to defaults one parameter to a live vendor
    object -- ``skip_dma: SkipMode = SkipMode()``
    (``bwmm_shard_on_I.py:119``). ``wrap_nki`` folds a kernel's stored default
    set at trace time, and Dynamo can turn that object neither into a Python
    constant nor into a graph proxy, so tracing the seam through the raw kernel
    dies before the graph is built. Every default here is ``None``, a primitive
    or an enum member instead, and the object is built inside the kernel body,
    where the NKI parser reads it.

    The vendor states the rule this implements in the target kernel's own
    comment at ``bwmm_shard_on_I.py:113-114``: flat booleans rather than a
    nested object, "because the NKI parser frontend cannot read attributes off
    a nested NKIObject inside a kernel." The same construction is already
    landed twice on this vendor family, at ``moe_cte.py:677`` and ``:766``.

    ``skip_token`` and ``skip_weight`` default to ``False``, which is the
    switch state ``SkipMode()`` already produced, so no caller's behaviour
    moves.
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


# --------------------------------------------------------------------------- #
# `inc-glm53f-113a`: THIS CAMPAIGN'S OWN gate/up kernel, at [128, 128].         #
# --------------------------------------------------------------------------- #
# WHAT THIS LIMB IS. One expert's gate/up projection over one block of tokens::
#
#     out[B, 2 * I] = hidden[B, H] @ dequantise(fused_gate_up_weight[H, 2 * I])
#
# with the block dequantisation folded into the tile loop and the result returned
# PRE-ACTIVATION in fp32. The scales it indexes are the checkpoint's own
# ``128 x 128`` grid, so no host-side retile stands between the checkpoint and the
# arithmetic -- which is the whole reason this increment exists.
#
# WHY IT IS A KERNEL AND NOT TORCH GLUE (P13). The routed-expert matmul is the
# model's dominant FLOP path, so its arithmetic runs in NKI. This limb carries NO
# torch projection route at all: an inadmissible geometry RAISES
# (:func:`_require_gate_up_blocked`), exactly as the landed
# ``functional/kda/gate_clamp.py`` and ``functional/attention/mla_projections.py``
# do, because a torch fallback for kernel-class work is the defect and not the
# remedy. The reference this limb is measured against lives in the acceptance
# test, not here.
#
# WHY IT HAS ITS OWN BODY AND DOES NOT CALL `inc-glm53f-026`'S DENSE KERNEL. At
# ``[128,128]`` the two arithmetics coincide -- one blocked fp8 matmul -- and that
# is stated here rather than left for a reader to notice. Three properties this
# body has and a call to the dense kernel could not give:
#
#   1. THE TWO HALVES ARE LIVE TOGETHER. Gate and up tiles for the same
#      ``i_block`` sit in two accumulators at once, which is what lets
#      `inc-glm53f-113b` apply ``SiLU(gate) * up`` inside this kernel instead of
#      writing both halves to HBM and reading them back. Two dense calls cannot
#      be fused after the fact.
#   2. ONE ACTIVATION TILE FEEDS BOTH MATMULS. The transposed hidden tile is
#      loaded once per contraction tile and used twice, so the activation DMA is
#      half what two separate projections would move.
#   3. THE DENSE KERNEL STAYS THE DENSE KERNEL. Growing it a fused-half axis it
#      has no caller for would put MoE geometry into a module whose acceptance is
#      already landed against the dense case.
#
# WHAT THIS LIMB DELIBERATELY DOES NOT DO, each with its owner. The token gather
# through ``token_position_to_id`` and the block-to-expert dispatch are NOT here:
# they need an indirect DMA this campaign has not measured on this image, and the
# seam that would need them is the block seam above. The activation and the
# expert-affinity scaling are `inc-glm53f-113b`. The down projection is
# `inc-glm53f-113b`. So this commit adds a limb and changes no route: a caller
# reaches it only by name.
#
# THE THREE TILE BOUNDS ARE THIS IMAGE'S. A ``nc_matmul`` contracts the PARTITION
# axis, so both operands present the contraction extent there: ``nl.tile_size.pmax``
# = 128 on the partition axis, ``gemm_stationary_fmax`` = 128 on the stationary
# free axis, ``gemm_moving_fmax`` = 512 on the moving free axis. Each was measured
# by refusal on this image at `inc-glm53f-039a` (``probe-039a-matmul.out``), and
# every tile below sits at or inside them.
#
# NO VENDOR CONSTANT IS INHERITED, AND THAT IS THE POINT. ``NUM_SHARDS``,
# ``MIN_HIDDEN``, ``MAX_HIDDEN`` and ``PSUM_SIZE`` at the top of this file are
# statements about the VENDOR kernel, traced to its own asserts; they still hold
# for the limb that calls it and they say nothing about this one. This kernel
# walks every axis in tiles, so it has no magnitude bound at all: its
# admissibility is positivity plus the two divisibility conditions its loops
# actually need. Re-imposing a bound this body does not have would defeat the
# reason the campaign authored it.
def _gate_up_sbuf(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """One fp32 accumulator tile in SBUF."""
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _gate_up_psum(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """One fp32 matmul destination in PSUM.

    Allocated inside the contraction-block loop, following the landed dense
    kernel: PSUM is reclaimed per block, which is what lets that block's scale be
    applied before the next block accumulates. Two are live at a time -- gate and
    up -- and each dies at the fold below.
    """
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def gate_up_flat_scale_index(
    h_block: int, gate_or_up: int, i_block: int, n_i_blocks: int
) -> int:
    """Column of the kernel's scale operand that holds one block's scale.

    The operand is ``[TILE_SIZE, n_blocks]``: one column per ``128 x 128`` weight
    block, replicated down the partition axis because ``nisa.tensor_scalar``
    broadcasts only along the free dimension. The column order is the C-order
    flattening of the checkpoint's own ``[H//128, 2, I//128]`` grid, so a caller
    that already holds that grid needs no index arithmetic of its own:
    :func:`to_gate_up_kernel_scale_operand` is a reshape and nothing else.
    """
    if n_i_blocks < 1:
        raise MoeBlockwiseFp8Error(f"n_i_blocks must be >= 1, got {n_i_blocks}")
    if not 0 <= gate_or_up < GATE_UP_FUSION:
        raise MoeBlockwiseFp8Error(
            f"gate_or_up must be 0 (gate) or 1 (up), got {gate_or_up}"
        )
    return (h_block * GATE_UP_FUSION + gate_or_up) * n_i_blocks + i_block


def gate_up_kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``[TILE_SIZE, n_blocks]`` -- the scale operand shape the kernel consumes.

    ``rows`` is ``H`` and ``cols`` is ``I`` (one half's width, not the fused
    width). Per expert: this limb takes one expert's weight, so the expert axis is
    the caller's loop and not an operand axis.
    """
    _require_gate_up_weight_blocked(rows, cols)
    n_blocks = (rows // GATE_UP_SCALE_BLOCK) * GATE_UP_FUSION * (
        cols // GATE_UP_SCALE_BLOCK
    )
    return (TILE_SIZE, n_blocks)


def to_gate_up_kernel_scale_operand(
    checkpoint_scales: Tensor, rows: int, cols: int
) -> Tensor:
    """One expert's ``[H//128, 2, I//128]`` checkpoint scales -> the kernel operand.

    A C-order reshape and a partition-axis broadcast, and nothing else: the
    column order IS the checkpoint grid's own order, which is what
    :func:`gate_up_flat_scale_index` states.

    Raises:
        MoeBlockwiseFp8Error: if the grid is not the shape the declared extents
            imply. Checked rather than trusted -- a mis-sized grid reshapes
            without error onto a different block-to-scale assignment, which is a
            wrong answer rather than a failure.
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
    """The two WEIGHT-axis conditions, as messages. Empty list means admissible.

    Split out from :func:`_require_gate_up_blocked` because the scale-operand
    helpers know the weight extents and do NOT know the token count: passing them
    a stand-in ``B`` would put a number in a refusal message that no caller
    supplied.
    """
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
    """Raise the one named error, or return. The message form is written once."""
    if problems:
        raise MoeBlockwiseFp8Error(
            "the campaign gate/up kernel refuses this geometry: "
            + "; ".join(problems)
        )


def _require_gate_up_weight_blocked(rows: int, cols: int) -> None:
    """The weight-axis conditions alone: ``H`` and ``I`` are ``128``-blocked."""
    _refuse_gate_up(_gate_up_weight_problems(rows, cols))


def _require_gate_up_blocked(tokens: int, rows: int, cols: int) -> None:
    """Every extent condition the kernel's own loops impose, in one place.

    ``tokens`` is ``B``, ``rows`` is ``H``, ``cols`` is ``I`` (one half). Only
    positivity and divisibility are checked, and the ABSENCE of an upper bound is
    deliberate -- see the section comment above.
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
    """What route the gate/up limb actually took, counted rather than inferred.

    A SECOND counter pair in this module, which the campaign's own convention
    admits when one summed pair could not tell two entry points apart: the landed
    ``functional/dsa/causal_bound.py`` carries two pairs for exactly that reason
    (``test_dsa_layer.py:3665``). ``-027``'s three readers --
    :data:`_COUNTERS`, :func:`reset_dispatch_counters` and
    :func:`dispatch_counters` -- keep their names, shapes and module, and this
    pair is disjoint from them.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_GATE_UP_COUNTERS = _GateUpDispatchCounters()


def reset_gate_up_dispatch_counters() -> None:
    """Zero the gate/up limb's counters. Called before a case's first call."""
    _GATE_UP_COUNTERS.nki_dispatch = 0
    _GATE_UP_COUNTERS.torch_fallback = 0


def gate_up_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the gate/up limb since the reset.

    ``torch_fallback`` can only ever read ``0``, because this limb has no torch
    projection route to increment it -- an inadmissible geometry raises (P13).
    The counter is kept so a test can STATE that reading instead of assuming it,
    which is what makes the zero a measurement.
    """
    return _GATE_UP_COUNTERS.nki_dispatch, _GATE_UP_COUNTERS.torch_fallback


@nki.jit
def moe_gate_up_blockwise_fp8_kernel(hidden, fused_weight, scale_operand):
    """``out[B, 2*I] = hidden[B, H] @ dequantise(fused_weight[H, 2*I])``, fp32.

    Args:
        hidden: ``[B, H]``, bf16. Loaded through ``nl.load_transpose2d`` so the
            contraction extent lands on the partition axis without an on-chip
            transpose.
        fused_weight: ``[H, 2*I]`` fp8-e4m3, gate columns first then up columns --
            the plain reshape of the checkpoint's ``[H, 2, I]`` for one expert, so
            no host copy stands between the two. Upcast to bf16 on the DMA.
        scale_operand: ``[TILE_SIZE, n_blocks]`` fp32 from
            :func:`to_gate_up_kernel_scale_operand`.

    Returns:
        ``[B, 2*I]`` fp32, PRE-ACTIVATION. The activation and the expert-affinity
        scaling are `inc-glm53f-113b`'s and are applied inside this kernel there,
        never in torch between two kernels.

    The accumulation is in PSUM within one contraction block and in SBUF across
    blocks, so each block's scale multiplies exactly the partial sum it belongs
    to: ``nisa.tensor_scalar`` initialises the accumulator on the first block and
    ``nisa.scalar_tensor_tensor`` multiplies-and-adds on every later one. Every
    loop bound is a trace-time int read off a tensor shape, which is why these are
    ``range`` loops and not ``nl.affine_range``.
    """
    tokens, h_extent = hidden.shape
    _, fused_cols = fused_weight.shape
    i_extent = fused_cols // GATE_UP_FUSION
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    n_col_blocks = GATE_UP_FUSION * n_i_blocks

    out = nl.ndarray((tokens, fused_cols), dtype=nl.float32, buffer=nl.shared_hbm)
    # One load: the scale operand is (partitions x blocks) and tiny.
    scale_sb = nl.load(scale_operand)

    for m_tile in range(tokens // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
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
                    # [H=TILE_SIZE partitions, B=TILE_SIZE free]
                    hidden_t = nl.load_transpose2d(
                        hidden[m0 : m0 + TILE_SIZE, h0 : h0 + TILE_SIZE]
                    )
                    # [H=TILE_SIZE partitions, I=GATE_UP_SCALE_BLOCK free]
                    gate_w = nl.load(
                        fused_weight[
                            h0 : h0 + TILE_SIZE,
                            gate_col : gate_col + GATE_UP_SCALE_BLOCK,
                        ],
                        dtype=nl.bfloat16,
                    )
                    up_w = nl.load(
                        fused_weight[
                            h0 : h0 + TILE_SIZE,
                            up_col : up_col + GATE_UP_SCALE_BLOCK,
                        ],
                        dtype=nl.bfloat16,
                    )
                    # dst = stationary.T @ moving = [B, I]. The accumulate flag is
                    # explicit rather than inferred, so first-write-overwrites is
                    # visible here.
                    nisa.nc_matmul(
                        dst=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=hidden_t,
                        moving=gate_w,
                        accumulate=(h_sub > 0),
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=hidden_t,
                        moving=up_w,
                        accumulate=(h_sub > 0),
                    )
                # The same flattening `gate_up_flat_scale_index` returns, written
                # as the block walk that produces it: `h_block * n_col_blocks +
                # (gate_or_up * n_i_blocks + i_block)`.
                gate_flat = h_block * n_col_blocks + i_block
                up_flat = h_block * n_col_blocks + n_i_blocks + i_block
                if h_block == 0:
                    # The first block initialises the accumulator, so there is no
                    # zeroing pass over SBUF.
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
            nl.store(
                out[m0 : m0 + TILE_SIZE, gate_col : gate_col + GATE_UP_SCALE_BLOCK],
                value=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
            nl.store(
                out[m0 : m0 + TILE_SIZE, up_col : up_col + GATE_UP_SCALE_BLOCK],
                value=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
    return out


# --------------------------------------------------------------------------- #
# `inc-glm53f-113b`: the activation and the down projection, both in NKI.        #
# --------------------------------------------------------------------------- #
# WHAT THESE TWO LIMBS ARE. Together with `-113a` they complete the block's
# arithmetic: ``SiLU(gate) * up`` on the gate/up result, then the down projection
# with its own ``128 x 128`` block dequantisation folded in, then the expert
# affinity on the down result -- which is where the plan block puts it. No torch
# arithmetic stands between any two stages; each stage is a kernel and the seam
# only passes tensors.
#
# WHY TWO SEAMS AND NOT ONE FUSED KERNEL. The plan pins the compared tensor as the
# down projection's own fp32 matmul output, and ``SiLU`` is transcendental: no
# reference this repository can write reproduces a device sigmoid bit for bit, so
# a kernel that computed the activation and the matmul in one pass could not be
# read by an equality at all. Splitting them keeps the matmul's equality EXACT and
# gives the activation its own reading at the file's already-declared tolerance.
# One consequence is stated rather than hidden: the intermediate goes out to HBM
# between the two, which a later fusion increment may remove -- and cannot remove
# by weakening this equality.
#
# WHY THE INTERMEDIATE COMES BACK TRANSPOSED. ``nc_matmul`` contracts the
# PARTITION axis, and the down projection contracts ``I``. The activation produces
# its result with tokens on partitions, so somebody must transpose; doing it once
# inside the activation kernel with the landed ``nisa.nc_transpose`` idiom
# (``functional/kda/gate_clamp.py:166``) costs one on-chip transpose per tile and
# lets the down kernel load both operands straight from HBM.
#
# WHY ``nisa.tensor_tensor`` IS NOWHERE HERE. An elementwise tensor-times-tensor
# is what the activation needs twice, and this image's parameter names for that
# member are not measured by this campaign. The three-operand form IS measured --
# `-113a` folds a scale with it -- so it is used with ``1.0`` as the identity
# scalar. That is one wasted multiply and a construct this seat can check, which
# is the trade an unmeasured API does not deserve.
#
# WHY THE AFFINITY IS A ``tensor_scalar`` OPERAND. ``operand0`` is a PER-PARTITION
# column, broadcast along the free axis only, and the down accumulator carries
# tokens on partitions -- so one affinity value per token is exactly the operand
# shape that member takes (``gate_clamp.py:178-195`` records the refusal that
# happens when it is not).
#
# WHAT IS STILL NOT HERE, AND WHY IT IS NOT A GAP IN THIS COMMIT. The block seam
# ``blockwise_fp8_moe`` still enters the vendor member. Switching it needs three
# device-side constructs -- an indirect row gather for the tokens, a per-block
# device scalar for the expert index, and an indirect row scatter for the output --
# which ARE landed idioms in this campaign (`inc-glm53f-044`'s paged gather and
# `-045`'s ragged pack, both reading ``.ap(vector_offset=..., indirect_dim=0)`` off
# the vendor's own ``scatter_add``). `-045`'s module docstring also records the
# campaign's standing rule about them: the mechanism is measured on this image, at
# the dtype and the width it will be used at, BEFORE the file that uses it is
# authored. That measurement is a simulator round this seat cannot run, so the
# switch waits for it rather than being guessed here.
def can_run_moe_gate_up_blockwise_fp8(
    hidden_states: Tensor, tokens: int, rows: int, cols: int
) -> bool:
    """Is the NKI route available *and* is this geometry admissible?

    Two independent conditions, deliberately not merged, following the vendor
    limb's own predicate above: ``can_run_kernel`` answers "is there a device or a
    simulator" and :func:`_require_gate_up_blocked` answers "does the kernel
    accept these extents". An inadmissible geometry raises rather than reading
    False, because falling back would ship torch for kernel-class work.

    Raises:
        MoeBlockwiseFp8Error: if the geometry is inadmissible.
    """
    _require_gate_up_blocked(tokens, rows, cols)
    return can_run_kernel(hidden_states)


def moe_gate_up_blockwise_fp8(
    hidden_states: Tensor,
    fused_gate_up_weight: Tensor,
    gate_up_scale_operand: Tensor,
) -> Tensor:
    """The counted gate/up seam. ``[B, H]`` and ``[H, 2*I]`` in, ``[B, 2*I]`` fp32 out.

    ``fused_gate_up_weight`` is CONTRACTION-MAJOR -- ``H`` on axis 0 -- which is
    the orientation the checkpoint already stores for this projection
    (``[E, H, 2, I]``), so one expert's weight reaches the kernel as a reshape and
    never as a copy. The result is PRE-ACTIVATION.
    """
    if hidden_states.ndim != 2:
        raise MoeBlockwiseFp8Error(
            f"hidden_states must be [B, H]; got {tuple(hidden_states.shape)}"
        )
    if fused_gate_up_weight.ndim != 2:
        raise MoeBlockwiseFp8Error(
            f"fused_gate_up_weight must be [H, 2*I]; got "
            f"{tuple(fused_gate_up_weight.shape)}"
        )
    tokens, rows = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    if int(fused_gate_up_weight.shape[0]) != rows:
        raise MoeBlockwiseFp8Error(
            f"fused_gate_up_weight is contraction-major, so its axis 0 must equal "
            f"hidden_states.shape[1]; got weight "
            f"{tuple(fused_gate_up_weight.shape)} against hidden "
            f"{tuple(hidden_states.shape)}. A [2*I, H] weight is the likely "
            f"cause -- this seam does not accept that orientation"
        )
    fused_cols = int(fused_gate_up_weight.shape[1])
    if fused_cols % GATE_UP_FUSION:
        raise MoeBlockwiseFp8Error(
            f"fused_gate_up_weight has {fused_cols} columns, which is not a "
            f"multiple of GATE_UP_FUSION={GATE_UP_FUSION}; the gate half and the "
            f"up half must be equally wide"
        )
    cols = fused_cols // GATE_UP_FUSION
    _require_gate_up_blocked(tokens, rows, cols)

    expected = gate_up_kernel_scale_shape(rows, cols)
    if tuple(gate_up_scale_operand.shape) != expected:
        raise MoeBlockwiseFp8Error(
            f"gate_up_scale_operand has shape "
            f"{tuple(gate_up_scale_operand.shape)}, expected {expected} at "
            f"[B={tokens}, H={rows}, I={cols}]. Build it with "
            f"to_gate_up_kernel_scale_operand rather than by hand."
        )

    _GATE_UP_COUNTERS.nki_dispatch += 1
    return wrap_nki(moe_gate_up_blockwise_fp8_kernel)(
        hidden_states.to(torch.bfloat16),
        fused_gate_up_weight,
        gate_up_scale_operand.to(torch.float32),
    )


def down_flat_scale_index(i_block: int, h_block: int, n_h_blocks: int) -> int:
    """Column of the down kernel's scale operand holding one block's scale.

    The order is the C-order flattening of the checkpoint's own ``[I//128, H//128]``
    down grid -- contraction-block major -- which is the same relationship the
    ``256``-era logical shape ``[E, I_256, H_256, TILE_SIZE]`` records at the top of
    this file.
    """
    if n_h_blocks < 1:
        raise MoeBlockwiseFp8Error(f"n_h_blocks must be >= 1, got {n_h_blocks}")
    return i_block * n_h_blocks + h_block


def down_kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``[TILE_SIZE, n_blocks]`` for the down projection. ``rows`` is ``I``, ``cols`` is ``H``."""
    _require_gate_up_weight_blocked(rows, cols)
    n_blocks = (rows // GATE_UP_SCALE_BLOCK) * (cols // GATE_UP_SCALE_BLOCK)
    return (TILE_SIZE, n_blocks)


def to_down_kernel_scale_operand(
    checkpoint_scales: Tensor, rows: int, cols: int
) -> Tensor:
    """One expert's ``[I//128, H//128]`` down scales -> the kernel operand.

    Raises:
        MoeBlockwiseFp8Error: if the grid is not the shape the extents imply. A
            mis-sized grid reshapes without error onto a different block-to-scale
            assignment, which is a wrong answer rather than a failure.
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
        # The kernel below walks one contraction TILE per scale BLOCK. The quotient
        # is 1 at this granularity and the guard is here so a changed constant
        # fails loudly instead of dropping contraction tiles silently.
        problems.append(
            f"GATE_UP_H_TILES_PER_BLOCK={GATE_UP_H_TILES_PER_BLOCK}, and the down "
            f"kernel is written for exactly 1 contraction tile per scale block"
        )
    _refuse_gate_up(problems)


@dataclass
class _SwigluDispatchCounters:
    """The activation limb's route, counted separately from the two matmul limbs."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


@dataclass
class _DownDispatchCounters:
    """The down limb's route, counted separately from the other two."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_SWIGLU_COUNTERS = _SwigluDispatchCounters()
_DOWN_COUNTERS = _DownDispatchCounters()


def reset_swiglu_dispatch_counters() -> None:
    """Zero the activation limb's counters."""
    _SWIGLU_COUNTERS.nki_dispatch = 0
    _SWIGLU_COUNTERS.torch_fallback = 0


def swiglu_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the activation limb. The second is 0 by construction."""
    return _SWIGLU_COUNTERS.nki_dispatch, _SWIGLU_COUNTERS.torch_fallback


def reset_down_dispatch_counters() -> None:
    """Zero the down limb's counters."""
    _DOWN_COUNTERS.nki_dispatch = 0
    _DOWN_COUNTERS.torch_fallback = 0


def down_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the down limb. The second is 0 by construction."""
    return _DOWN_COUNTERS.nki_dispatch, _DOWN_COUNTERS.torch_fallback


@nki.jit
def moe_swiglu_transposed_kernel(gate_up):
    """``SiLU(gate) * up`` for one token block, returned as ``[I, B]`` fp32.

    Args:
        gate_up: ``[B, 2*I]`` -- `-113a`'s pre-activation output, gate columns then
            up columns.

    Returns:
        ``[I, B]`` fp32, transposed so the down projection can contract ``I`` on the
        partition axis without transposing anything itself.
    """
    tokens, fused_cols = gate_up.shape
    i_extent = fused_cols // GATE_UP_FUSION
    out = nl.ndarray((i_extent, tokens), dtype=nl.float32, buffer=nl.shared_hbm)

    for m_tile in range(tokens // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for i_block in range(i_extent // GATE_UP_SCALE_BLOCK):
            i0 = i_block * GATE_UP_SCALE_BLOCK
            gate = nl.load(
                gate_up[m0 : m0 + TILE_SIZE, i0 : i0 + GATE_UP_SCALE_BLOCK],
                dtype=nl.float32,
            )
            up = nl.load(
                gate_up[
                    m0 : m0 + TILE_SIZE,
                    i_extent + i0 : i_extent + i0 + GATE_UP_SCALE_BLOCK,
                ],
                dtype=nl.float32,
            )
            # sigmoid is one activation-engine op on this image, not a composition.
            squashed = _gate_up_sbuf()
            nisa.activation(dst=squashed, data=gate, op=nl.sigmoid)
            # SiLU(gate) = gate * sigmoid(gate). ``1.0`` is the identity scalar the
            # three-operand form needs; see the section comment on tensor_tensor.
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
            nl.store(
                out[i0 : i0 + GATE_UP_SCALE_BLOCK, m0 : m0 + TILE_SIZE],
                value=out_sb,
            )
    return out


def moe_swiglu_transposed(gate_up: Tensor) -> Tensor:
    """The counted activation seam. ``[B, 2*I]`` in, ``[I, B]`` fp32 out.

    No torch route: an inadmissible shape raises (P13). The transposed return is
    part of the contract, not an implementation detail -- the section comment says
    why the transpose lives here.
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

    _SWIGLU_COUNTERS.nki_dispatch += 1
    return wrap_nki(moe_swiglu_transposed_kernel)(gate_up.to(torch.float32))


@nki.jit
def moe_down_blockwise_fp8_kernel(intermediate_t, down_weight, scale_operand, affinity):
    """``out[B, H] = (intermediate[B, I] @ dequantise(down_weight[I, H])) * affinity``.

    Args:
        intermediate_t: ``[I, B]`` -- the activation seam's transposed output.
            Loaded as bf16, which is the compute dtype the vendor kernel uses too.
        down_weight: ``[I, H]`` fp8-e4m3, contraction-major. Upcast on the DMA.
        scale_operand: ``[TILE_SIZE, n_blocks]`` fp32 from
            :func:`to_down_kernel_scale_operand`.
        affinity: ``[B, 1]`` fp32 -- one expert affinity per token, applied to the
            down result, which is where the plan block puts it.

    Returns:
        ``[B, H]`` fp32.

    One contraction tile per scale block at this granularity, so ``accumulate`` is
    False on every matmul and each block's scale multiplies exactly the partial sum
    it belongs to. The seam refuses the geometry if that quotient ever stops being
    one.
    """
    i_extent, tokens = intermediate_t.shape
    _, h_extent = down_weight.shape
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK

    out = nl.ndarray((tokens, h_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    scale_sb = nl.load(scale_operand)

    for m_tile in range(tokens // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        # LOADED PER TOKEN TILE, never hoisted out of this loop: a token is a
        # PARTITION, the partition axis serves 128 of them, and a whole-column load
        # would bound this kernel to one tile. The causal-bound kernel re-loads its
        # per-row length column inside its own row loop for the same reason.
        affinity_sb = nl.load(affinity[m0 : m0 + TILE_SIZE, 0:1], dtype=nl.float32)
        for h_block in range(n_h_blocks):
            h0 = h_block * GATE_UP_SCALE_BLOCK
            acc = _gate_up_sbuf()
            for i_block in range(n_i_blocks):
                i0 = i_block * GATE_UP_SCALE_BLOCK
                psum = _gate_up_psum()
                inter_tile = nl.load(
                    intermediate_t[i0 : i0 + TILE_SIZE, m0 : m0 + TILE_SIZE],
                    dtype=nl.bfloat16,
                )
                w_tile = nl.load(
                    down_weight[i0 : i0 + TILE_SIZE, h0 : h0 + GATE_UP_SCALE_BLOCK],
                    dtype=nl.bfloat16,
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
            nl.store(
                out[m0 : m0 + TILE_SIZE, h0 : h0 + GATE_UP_SCALE_BLOCK],
                value=scaled[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
    return out


def can_run_moe_down_blockwise_fp8(
    intermediate_t: Tensor, tokens: int, rows: int, cols: int
) -> bool:
    """Is the NKI route available *and* is this down geometry admissible?

    Raises:
        MoeBlockwiseFp8Error: if the geometry is inadmissible.
    """
    _require_down_blocked(tokens, rows, cols)
    return can_run_kernel(intermediate_t)


def moe_down_blockwise_fp8(
    intermediate_t: Tensor,
    down_weight: Tensor,
    down_scale_operand: Tensor,
    affinity: Tensor,
) -> Tensor:
    """The counted down seam. ``[I, B]``, ``[I, H]``, operand and ``[B, 1]`` in; ``[B, H]`` fp32 out."""
    for name, tensor, rank in (
        ("intermediate_t", intermediate_t, 2),
        ("down_weight", down_weight, 2),
        ("affinity", affinity, 2),
    ):
        if tensor.ndim != rank:
            raise MoeBlockwiseFp8Error(
                f"{name} must have rank {rank}; got shape {tuple(tensor.shape)}"
            )
    rows, tokens = int(intermediate_t.shape[0]), int(intermediate_t.shape[1])
    if int(down_weight.shape[0]) != rows:
        raise MoeBlockwiseFp8Error(
            f"down_weight is contraction-major, so its axis 0 must equal "
            f"intermediate_t.shape[0]; got weight {tuple(down_weight.shape)} "
            f"against intermediate {tuple(intermediate_t.shape)}. A [H, I] weight "
            f"is the likely cause -- this seam does not accept that orientation"
        )
    cols = int(down_weight.shape[1])
    _require_down_blocked(tokens, rows, cols)

    expected = down_kernel_scale_shape(rows, cols)
    if tuple(down_scale_operand.shape) != expected:
        raise MoeBlockwiseFp8Error(
            f"down_scale_operand has shape {tuple(down_scale_operand.shape)}, "
            f"expected {expected} at [B={tokens}, I={rows}, H={cols}]. Build it "
            f"with to_down_kernel_scale_operand rather than by hand."
        )
    if tuple(affinity.shape) != (tokens, 1):
        raise MoeBlockwiseFp8Error(
            f"affinity must be [B, 1] = {(tokens, 1)}; got "
            f"{tuple(affinity.shape)}. One expert affinity per token, which is the "
            f"per-partition operand shape the kernel applies it as."
        )

    _DOWN_COUNTERS.nki_dispatch += 1
    return wrap_nki(moe_down_blockwise_fp8_kernel)(
        intermediate_t.to(torch.float32),
        down_weight,
        down_scale_operand.to(torch.float32),
        affinity.to(torch.float32),
    )


# --------------------------------------------------------------------------- #
# The two identity readings, DERIVED THROUGH THE SEAM.                          #
# --------------------------------------------------------------------------- #
# WHAT WAS WRONG (`B26-M1`, repaired at `inc-glm53f-077`). This function used to
# read the MODULE-LEVEL import at the top of this file:
#
#     func = getattr(blockwise_mm_baseline_shard_intermediate, "func", None)
#     target = func if func is not None else blockwise_mm_baseline_shard_intermediate
#     return target.__module__, target.__qualname__
#
# That reads a name this module imports, NOT the object the seam sends work to.
# The seam wraps the shim (`blockwise_fp8_moe`, the `wrap_nki(...)` call below
# it), and the shim forwards to the vendor kernel from inside its own body. So
# every substitution the reading exists to catch was invisible to it: change
# what the shim forwards to, or change which object `wrap_nki` wraps, and the
# reading stayed byte-identical. `evidence-077.md` §5 recorded that silence as
# "identical" and read it as reassurance, which is the defect.
#
# WHAT IS DIFFERENT NOW. Both readings start at the seam and follow the real
# call chain, resolving each step through the LIVE module binding:
#
#     blockwise_fp8_moe  --wrap_nki(...)-->  the shim  --return-->  the kernel
#                              |                            |
#                        seam_identity()             kernel_identity()
#
# so `seam_identity()` moves when the `wrap_nki` argument is substituted, and
# `kernel_identity()` moves when EITHER hop is substituted. The readings are
# graded rather than equal, which is what lets one arm separate the two hazards.
#
# THE SHIM'S BODY IS NOT TOUCHED. Deriving by introspection rather than by an
# indirection the shim calls keeps every line of the NKI-traced body identical,
# so no kernel numerics and no trace-time behaviour moves for this repair.
#
# A BROKEN DERIVATION RAISES. There is deliberately no fall back to the
# module-level import: falling back is exactly the silence this repair removes,
# and a reading that cannot be derived must say so rather than return a
# plausible answer.
def _unwrap_nki(obj: Any) -> Any:
    """The plain Python function behind an ``nki.jit`` object, else ``obj``.

    ``nki.jit`` stores the decorated function on ``.func``; the vendor kernel and
    this module's shim are both such objects, and ``__module__``/``__qualname__``
    of the wrapper are not the kernel's. ``__wrapped__`` is tried second because
    it is what ``functools.wraps`` sets, and which of the two a given ``nki``
    build populates is read here rather than assumed.
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

    Going through ``__globals__`` rather than through this module's own
    namespace is what makes the reading move: a rebound module global moves it
    too, not only an edited call site.
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
    """``(module, qualname)`` of the object ``wrap_nki`` actually wraps.

    This is the shim, not the vendor kernel. Substituting the seam's ``wrap_nki``
    argument moves this reading; substituting what the shim forwards to does not.
    """
    obj = _unwrap_nki(_seam_wrapped_object())
    return obj.__module__, obj.__qualname__


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the NKI member the seam ultimately forwards to.

    Derived through the seam and then through the shim's own forward target, so a
    substitution at either hop moves the reading instead of leaving it silent.

    Raises:
        MoeBlockwiseFp8Error: if the chain cannot be derived. There is no fall
            back to this module's import of the kernel -- that fall back is the
            silence `B26-M1` found.
    """
    target = _unwrap_nki(_shim_forward_target())
    return target.__module__, target.__qualname__


# --------------------------------------------------------------------------- #
# `inc-glm53f-113a`'s identity reading, derived through ITS OWN seam.            #
# --------------------------------------------------------------------------- #
# WHY A SECOND DERIVATION AND NOT A PARAMETER ON THE FIRST. `_seam_wrapped_object`
# above is read BY NAME and called with no arguments by a landed acceptance item,
# so giving it a parameter would edit landed evidence-bearing code to save ten
# lines. The rule it applies is written once, here, and the landed function is
# left byte-identical on purpose. This is also why the reading below is DERIVED
# rather than returned off the module-level kernel name: `B26-M1` is the finding
# that a reading taken off an import stays byte-identical when the seam is
# substituted, which is the silence the derivation removes.
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
    """``(module, qualname)`` of the kernel the gate/up seam dispatches to.

    Read by the acceptance so the kernel under test is known to be authored in
    this campaign rather than imported from the substrate. THE UNWRAP IS THE WHOLE
    READING: ``nki.jit`` returns a wrapper whose own ``__module__`` is the
    substrate's, so reading the attribute off the decorated object reports the
    same answer for an authored kernel and an imported one alike.

    Raises:
        MoeBlockwiseFp8Error: if the chain cannot be derived. There is
            deliberately no fall back to this module's own kernel name.
    """
    obj = _unwrap_nki(_wrapped_object_of(moe_gate_up_blockwise_fp8, "the gate/up seam"))
    return obj.__module__, obj.__qualname__


def swiglu_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel the activation seam dispatches to.

    Derived through the seam by the same rule as the reading above, so a
    substitution moves it instead of leaving it silent.
    """
    obj = _unwrap_nki(_wrapped_object_of(moe_swiglu_transposed, "the activation seam"))
    return obj.__module__, obj.__qualname__


def down_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the kernel the down seam dispatches to."""
    obj = _unwrap_nki(_wrapped_object_of(moe_down_blockwise_fp8, "the down seam"))
    return obj.__module__, obj.__qualname__
