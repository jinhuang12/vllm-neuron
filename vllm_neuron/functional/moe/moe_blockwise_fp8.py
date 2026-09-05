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

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

import nki
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
    "seam_identity",
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
