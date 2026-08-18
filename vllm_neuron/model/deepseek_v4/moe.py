# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 MoE block
=====================

Three modules, in the order the data flows through them:

* :class:`DeepseekV4MoE` — the router. Owns ``gate_weight``, the optional
  ``gate_bias`` and, on the three hash layers only, the ``tid2eid`` table.
  It produces the per-token expert selection and weights, drives the routed
  experts and the shared expert, and sums the two contributions.
* :class:`DeepseekV4RoutedExperts` — the 256 routed experts, MXFP8 group-32,
  four of them resident per core at ``ep_degree=64``.
* :class:`DeepseekV4SharedExpert` — the single always-on expert, block-128x128
  FP8, sharded 16 ways and replicated four times across the TP group.

ANNOTATION GUIDE (llama3 / gpt_oss house style):
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    DeepSeek-V4-specific. Change when porting.

WHY THE ROUTER IS HAND-ROLLED RATHER THAN A KERNEL ROUTER
--------------------------------------------------------
``NF.router``, ``NF.rmsnorm_router_topk_tkg`` and ``NF.moe_block_tkg`` all
embed their own router, and every one of them offers only ``SOFTMAX`` or
``SIGMOID`` scoring with the bias folded into the *logits*. DeepSeek-V4 needs
three things none of them can express:

1. ``sqrt(softplus(logits))`` scoring (``dsv4_ref/model.py:576``),
2. bias-corrected *selection* against *uncorrected* weights — the
   ``noaux_tc`` rule, where ``gate_bias`` shifts the score used to pick the
   experts but never the score used to weight them
   (``dsv4_ref/model.py:577-585``), and
3. on layers 0-2, selection by a vocabulary-indexed table lookup with no
   logits involved at all (``dsv4_ref/model.py:581-582``).

So routing is torch here, and only the expert MLPs go to a kernel. Every
routing formula below cites the line of DeepSeek's own reference
implementation (``dsv4_ref/model.py``, shipped in the pinned checkpoint
repo) that fixes it — that reference, not the derived spec reports,
is the authority on this arithmetic.
"""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

import vllm_neuron.functional as NF

from .config import DeepseekV4Config

logger = logging.getLogger(__name__)


# <-- MODEL-SPECIFIC: MX scale group extent for the routed experts. The
# checkpoint stores one E8M0 byte per 32 elements along K
# (``dsv4_ref/model.py:143-145``: FP4 scale is ``[out, in // fp4_block_size]``
# with ``fp4_block_size == 32``). The MXFP8 upcast the loader performs keeps
# the group extent; only the element width changes.
_MX_GROUP_SIZE = 32

# Number of MX elements packed into one machine word on the expert path. The
# MoE kernels take their weights bit-cast to a wide integer dtype and
# reinterpret it (``functional/moe/moe_tkg_wrapper.py``: ``uint32 ->
# float8_e4m3fn_x4``), so four FP8 elements travel as one ``uint32``.
_MX_ELEMS_PER_WORD = 4

# MX tile extents the MoE kernels document. ``gate_up`` is tiled
# ``[E, 128, 2, ceil(H/512), I]``; ``down`` is tiled ``[E, I_p, ceil(I/512), H]``
# with ``I_p = 128`` once ``I > 512`` (``functional/moe/moe_tkg.py:62,70-72``).
_MX_TILE_ROWS = 128
_MX_TILE_K = 512

# >>> PARALLELISM: block extent for the prefill blockwise mapping. Must be a
# multiple of 128 (``functional/moe/moe_cte.py:198``). 256 is what gpt_oss
# uses at the same hidden size, so the block-count arithmetic is already
# exercised at this shape. <<<
_MOE_CTE_BLOCK_SIZE = 256

# The block-FP8 contract for the shared expert, frozen by the family
# interface contract §5. Every leg of the shared-expert MLP is called with
# exactly these arguments.
_BLOCK_FP8_BLOCK_SIZE = (128, 128)
_BLOCK_FP8_ACT_GROUP_SIZE = 128


def _moe_kernel_enums() -> tuple:
    """Resolve the enum members the MoE entry points take.

    These live in ``nkilib``, which this family reaches *only* through the
    ``functional/moe`` wrappers that already import it — hence the attribute
    reads off those modules rather than a direct ``nkilib`` import. The
    import is deferred to call time so this module also imports on a host
    without the Neuron toolchain installed.

    Returns:
        ``(MoECTEImplementation, ActFnType, ExpertAffinityScaleMode,
        MoEAllToAllVStrategy)``.
    """
    from vllm_neuron.functional.moe import moe_cte as _cte_mod
    from vllm_neuron.functional.moe import moe_tkg as _tkg_mod

    return (
        _cte_mod.MoECTEImplementation,
        _cte_mod.ActFnType,
        _cte_mod.ExpertAffinityScaleMode,
        _tkg_mod.MoEAllToAllVStrategy,
    )


def _resolve_ep(config: DeepseekV4Config) -> tuple[int, int, object]:
    """Return ``(ep_degree, ep_rank, ep_group)`` for the routed experts.

    Read from the live Neuron parallel state, exactly as ``gpt_oss`` does
    (``model_mxfp4.py:880-901``): ``ep_degree`` is *derived* there
    (``world_size // ep_tp_group.world_size``) and is not a leaf-module
    config field, so the accessor is the only correct source. Falls back to
    ``ep_degree=1`` when EP is not initialized, so a bare construction in a
    unit test still builds.
    """
    try:
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_group,
            get_neuron_ep_rank,
        )

        ep_degree = get_neuron_ep_degree()
        if ep_degree > 1:
            return ep_degree, get_neuron_ep_rank(), get_neuron_ep_group()
    except Exception:  # noqa: BLE001 - absent parallel state is not a failure
        logger.debug("Neuron EP state unavailable; DeepseekV4 MoE falls back to EP=1.")

    ep_degree = config.neuron_config.ep_degree if config.neuron_config else 1
    return max(1, int(ep_degree)), 0, None


# =============================================================================
# Section 1: Routed experts (MXFP8 group-32, expert-parallel)
# =============================================================================


class DeepseekV4RoutedExperts(nn.Module):
    """The 256 routed experts, four resident per core at ``ep_degree=64``.

    >>> PARALLELISM: pure EP. Each core owns a disjoint contiguous quarter-
    percent of the expert set with the FULL intermediate dimension (2048),
    so no intra-expert TP sharding and no gather/scatter of activations is
    needed — only one all-reduce of the summed expert output at the end. <<<

    <-- MODEL-SPECIFIC: 256 experts, top-6, SwiGLU with the asymmetric
    ``swiglu_limit`` clamp, ``w1`` = gate / ``w3`` = up / ``w2`` = down.

    <-- MXFP8: the checkpoint stores these experts as MXFP4
    (``float4_e2m1fn_x2`` elements, ``[out, in // 32]`` E8M0 scales —
    ``dsv4_ref/model.py:140-145``). Trainium2 has no FP4 datapath, so the
    loader upcasts the elements to ``float8_e4m3fn`` and preserves the
    group-32 scale grid. This class consumes ONLY that upcast form.

    Parameter shapes are the family contract's, i.e. the *logical*
    ``[E_local, out, in]`` orientation. The MoE kernels want the same bytes
    in their tiled MX layout, and the two are byte-for-byte the same size at
    this model's dimensions (see :meth:`_gate_up_weight`), so the loader
    writes tiled bytes into these buffers and this class reinterprets them
    with pure ``view`` calls — no runtime repacking.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        ep_degree: int | None = None,
        ep_rank: int | None = None,
        ep_group: object | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # >>> PARALLELISM: EP placement <<<
        if ep_degree is None or ep_rank is None:
            resolved_degree, resolved_rank, resolved_group = _resolve_ep(config)
            ep_degree = resolved_degree if ep_degree is None else ep_degree
            ep_rank = resolved_rank if ep_rank is None else ep_rank
            ep_group = resolved_group if ep_group is None else ep_group
        self.ep_degree = int(ep_degree)
        self.ep_rank = int(ep_rank)
        self.ep_group = ep_group

        if config.n_routed_experts % self.ep_degree != 0:
            raise ValueError(
                f"ep_degree={self.ep_degree} does not divide "
                f"n_routed_experts={config.n_routed_experts}; expert placement "
                "would be ragged."
            )

        self.total_num_experts = config.n_routed_experts
        self.num_local_experts = config.n_routed_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        # <-- MODEL-SPECIFIC: asymmetric SwiGLU clamp, see :meth:`_clamps`.
        self.swiglu_limit = float(config.swiglu_limit)

        # >>> PARALLELISM: contiguous EP placement, matching
        # ``NF.calculate_local_expert_indices`` and the reference's own
        # ``experts_start_idx = rank * n_local_experts``
        # (``dsv4_ref/model.py:625-626``). <<<
        self.local_expert_start = self.ep_rank * self.num_local_experts

        num_mx_groups_h = self.hidden_size // _MX_GROUP_SIZE
        num_mx_groups_i = self.intermediate_size // _MX_GROUP_SIZE

        # <-- MXFP8: w1 = gate projection, w3 = up projection
        # (``dsv4_ref/model.py:602-603``: ``gate = self.w1(x)``,
        # ``up = self.w3(x)``). Both are ``[out=I, in=H]``, i.e. the
        # ``[out, in]`` orientation the checkpoint already uses — do NOT
        # transpose.
        expert_shape = (self.num_local_experts, self.intermediate_size, self.hidden_size)
        scale_shape = (self.num_local_experts, self.intermediate_size, num_mx_groups_h)
        self.w1_weight = nn.Parameter(
            torch.empty(*expert_shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w1_scale = nn.Parameter(
            torch.empty(*scale_shape, dtype=torch.uint8), requires_grad=False
        )
        self.w3_weight = nn.Parameter(
            torch.empty(*expert_shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w3_scale = nn.Parameter(
            torch.empty(*scale_shape, dtype=torch.uint8), requires_grad=False
        )

        # <-- MXFP8: w2 = down projection, ``[out=H, in=I]``.
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.w2_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                num_mx_groups_i,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

        # MX tile extents, derived once so forward stays free of arithmetic.
        self._h_tiles = math.ceil(self.hidden_size / _MX_TILE_K)
        self._i_tiles = math.ceil(self.intermediate_size / _MX_TILE_K)
        self._i_p = (
            _MX_TILE_ROWS
            if self.intermediate_size > _MX_TILE_K
            else self.intermediate_size // _MX_ELEMS_PER_WORD
        )
        # One E8M0 byte per 32 elements, four elements per packed word, so a
        # tile row of packed words covers eight scale groups.
        self._scale_rows_per_tile = _MX_GROUP_SIZE // _MX_ELEMS_PER_WORD

    # ------------------------------------------------------------------
    # MX layout reinterpretation (pure views; no repacking at runtime)
    # ------------------------------------------------------------------
    def _packed(self, weight: Tensor) -> Tensor:
        """Bit-cast an FP8 expert weight to the ``uint32`` word view.

        The two-step ``view(uint8).view(uint32)`` is the idiom every existing
        loader in the repo uses for this (``weight_loaders_mx_fp8.py:315``);
        it is a reinterpretation, never a numeric conversion, so the E4M3
        bytes survive intact. ``view`` (not ``reshape``) is deliberate: it
        raises rather than silently copying if the parameter is ever handed
        over non-contiguous.
        """
        return weight.view(torch.uint8).view(torch.uint32)

    def _gate_up_weight(self) -> Tensor:
        """Fuse ``w1``/``w3`` into the ``[E_L, 128, 2, H/512, I]`` MX buffer.

        Both MoE entry points take gate and up as one tensor with the
        gate/up axis at dim 2 (``functional/moe/moe_tkg.py:70``,
        ``functional/moe/moe_cte.py:119``). At this model's dimensions the
        byte counts line up exactly — ``w1`` alone is
        ``4 x 2048 x 4096`` FP8 bytes = ``4 x 128 x 1 x 8 x 2048`` uint32
        words — so each parameter reinterprets to one half of the fused
        buffer and the fuse is a single concatenation.

        RECORDED COST: that concatenation is a real copy on every forward
        (~64 MiB per layer). It exists only because the interface contract
        fixes ``w1_weight`` and ``w3_weight`` as separate parameters; a fused
        ``gate_up`` parameter emitted by ``attach_moe_loaders`` would remove
        it entirely. Flagged rather than silently absorbed.
        """
        gate = self._packed(self.w1_weight).view(
            self.num_local_experts,
            _MX_TILE_ROWS,
            1,
            self._h_tiles,
            self.intermediate_size,
        )
        up = self._packed(self.w3_weight).view(
            self.num_local_experts,
            _MX_TILE_ROWS,
            1,
            self._h_tiles,
            self.intermediate_size,
        )
        return torch.cat([gate, up], dim=2)

    def _gate_up_scale(self) -> Tensor:
        """Fuse the group-32 scales into ``[E_L, 16, 2, H/512, I]``.

        Same axis convention as :meth:`_gate_up_weight`; the leading tile
        extent is ``128 / 8 = 16`` because one packed word carries four
        elements and one scale covers 32 (``functional/moe/moe_tkg.py:86-88``).
        """
        rows = _MX_TILE_ROWS // self._scale_rows_per_tile
        gate = self.w1_scale.view(
            self.num_local_experts, rows, 1, self._h_tiles, self.intermediate_size
        )
        up = self.w3_scale.view(
            self.num_local_experts, rows, 1, self._h_tiles, self.intermediate_size
        )
        return torch.cat([gate, up], dim=2)

    def _down_weight(self) -> Tensor:
        """Reinterpret ``w2`` as ``[E_L, I_p, ceil(I/512), H]`` (no copy)."""
        return self._packed(self.w2_weight).view(
            self.num_local_experts, self._i_p, self._i_tiles, self.hidden_size
        )

    def _down_scale(self) -> Tensor:
        """Reinterpret the ``w2`` scales as ``[E_L, I_p/8, ceil(I/512), H]``."""
        return self.w2_scale.view(
            self.num_local_experts,
            self._i_p // self._scale_rows_per_tile,
            self._i_tiles,
            self.hidden_size,
        )

    # ------------------------------------------------------------------
    # Activation clamp
    # ------------------------------------------------------------------
    def _clamps(self) -> tuple[float | None, float | None, float | None, float | None]:
        """Return ``(gate_hi, gate_lo, up_hi, up_lo)`` for the SwiGLU clamp.

        <-- MODEL-SPECIFIC: the clamp is ASYMMETRIC, and this is the one
        place a symmetric reading would silently change numerics.
        ``dsv4_ref/model.py:605-607`` reads::

            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)

        i.e. ``up`` is clamped on BOTH sides to ``[-10, 10]`` while ``gate``
        is clamped on the UPPER side only — it keeps its unbounded negative
        tail, which matters because ``silu`` is applied to it next
        (``dsv4_ref/model.py:608``). ``swiglu_limit <= 0`` disables the
        clamp entirely, same guard as the reference.
        """
        if self.swiglu_limit <= 0.0:
            return None, None, None, None
        return self.swiglu_limit, None, self.swiglu_limit, -self.swiglu_limit

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,
        expert_affinities: Tensor,
        expert_index: Tensor,
        is_prefill: bool,
    ) -> Tensor:
        """Run the local experts and reduce across the EP group.

        Args:
            hidden_states: ``[T, hidden_size]`` bf16, replicated on every rank.
            expert_affinities: ``[T, n_routed_experts]`` fp32, dense — the
                routing weight at the selected experts and zero elsewhere,
                already carrying ``routed_scaling_factor``.
            expert_index: ``[T, num_experts_per_tok]`` int32, global expert ids.
            is_prefill: Python-level phase flag (never a tensor), so the
                branch is resolved at trace time and each phase compiles to
                its own graph.

        Returns:
            ``[T, hidden_size]`` bf16, the summed routed contribution,
            reduced over the EP group so every rank holds the full sum
            (``dsv4_ref/model.py:645-646``).
        """
        if is_prefill:
            output = self._forward_prefill(
                hidden_states, expert_affinities, expert_index
            )
        else:
            output = self._forward_decode(
                hidden_states, expert_affinities, expert_index
            )

        # >>> PARALLELISM: EP combine. Every rank computed the partial sum
        # over its own experts against a fully replicated token set, so the
        # combine is a plain all-reduce — no all-to-all, no reduce-scatter.
        # This mirrors the reference's own `dist.all_reduce(y)`
        # (``dsv4_ref/model.py:645-646``). <<<
        if self.ep_group is not None and self.ep_degree > 1:
            output = self.ep_group.all_reduce(output)
        return output.to(hidden_states.dtype)

    def _forward_prefill(
        self, hidden_states: Tensor, expert_affinities: Tensor, expert_index: Tensor
    ) -> Tensor:
        """Prefill via ``NF.moe_cte`` on the blockwise mapping.

        ``NF.moe_cte`` is the only MoE entry point that takes a *block*
        schedule rather than a per-token one, which is what makes it the
        prefill path: at ``T`` in the thousands the token-generation kernels
        would either exceed their ``T <= 128`` selective-mode ceiling
        (``functional/moe/moe_block_tkg.py:435-437``) or pay all-expert cost
        per token.
        """
        impl, act_fn, scale_mode, _ = _moe_kernel_enums()
        gate_hi, gate_lo, up_hi, up_lo = self._clamps()

        # >>> PARALLELISM: slice the dense global affinities down to this
        # rank's experts; ``build_blockwise_mapping`` wants ``[T, E_local]``
        # (``functional/moe/moe_blockwise.py:36``). <<<
        local_affinities = expert_affinities
        if self.ep_degree > 1:
            local_expert_indices = torch.arange(
                self.local_expert_start,
                self.local_expert_start + self.num_local_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            local_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_token,
            block_size=_MOE_CTE_BLOCK_SIZE,
            moe_group=self._ep_tp_group(),
            # >>> PARALLELISM: tp_degree=1 — under pure EP each rank owns
            # whole experts, so there is no intermediate-dim sharding for
            # the mapping to coordinate. <<<
            tp_degree=1,
            # No padding mask: this module's forward signature carries no
            # ``positions``, so real-vs-padding tokens cannot be
            # distinguished here. Padding tokens route like any other token
            # and their output rows are discarded downstream.
            padding_mask=None,
        )

        # ``moe_cte`` reads ``conditions`` as ``[N + 2]`` (two trailing
        # zeros), while ``build_blockwise_mapping`` returns ``[N]``
        # (``functional/moe/moe_cte.py:152-154``).
        conditions = torch.cat(
            [
                conditions,
                torch.zeros(2, dtype=conditions.dtype, device=conditions.device),
            ]
        )

        return NF.moe_cte(
            implementation=impl.shard_on_block_mx,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self._gate_up_weight(),
            down_proj_weight=self._down_weight(),
            gate_up_proj_scale=self._gate_up_scale(),
            down_proj_scale=self._down_scale(),
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            block_size=_MOE_CTE_BLOCK_SIZE,
            conditions=conditions,
            # <-- MODEL-SPECIFIC: plain SiLU, not the Swish/1.702 variant
            # gpt_oss uses (``dsv4_ref/model.py:608``: ``F.silu(gate) * up``).
            activation_function=act_fn.SiLU,
            # The reference multiplies the routing weight into the
            # intermediate BEFORE the down projection
            # (``dsv4_ref/model.py:609-610``). The down projection is
            # linear, so POST_SCALE is algebraically identical — moe_cte's
            # own docstring states that equivalence
            # (``functional/moe/moe_cte.py:485-492``).
            expert_affinities_scaling_mode=scale_mode.POST_SCALE,
            gate_clamp_upper_limit=gate_hi,
            gate_clamp_lower_limit=gate_lo,
            up_clamp_upper_limit=up_hi,
            up_clamp_lower_limit=up_lo,
            skip_token=True,
            is_tensor_update_accumulating=True,
        )

    def _forward_decode(
        self, hidden_states: Tensor, expert_affinities: Tensor, expert_index: Tensor
    ) -> Tensor:
        """Decode via ``NF.moe_tkg``.

        ``NF.moe_tkg`` is chosen over ``NF.moe_block_tkg`` because it is the
        only decode entry point that accepts a routing decision computed
        outside the kernel: it takes ``expert_affinities`` and
        ``expert_index`` as arguments (``functional/moe/moe_tkg.py:20-21``),
        whereas ``moe_block_tkg`` fuses RMSNorm + its own router and exposes
        only ``router_act_fn`` in ``{SOFTMAX, SIGMOID}``
        (``functional/moe/moe_block_tkg.py:325``) — which cannot express
        ``sqrtsoftplus`` scoring, the ``noaux_tc`` bias-corrected selection,
        or the ``tid2eid`` hash gather. Passing our own routing through
        ``moe_block_tkg`` is not possible; passing it through ``moe_tkg`` is
        exactly what its signature is for.

        ``all_to_all_v_strategy`` stays ``DISABLED``: tokens are already
        replicated on every rank (the router is replicated), so there is
        nothing to dispatch. ``is_all_expert=True`` with ``rank_id`` lets the
        kernel slice the dense global affinities to the local experts itself
        (``functional/moe/moe_tkg.py:74-76``).
        """
        _, act_fn, scale_mode, a2a = _moe_kernel_enums()
        gate_hi, gate_lo, up_hi, up_lo = self._clamps()

        # >>> PARALLELISM: rank_id is a tensor, not a Python int, so the EP
        # rank is not baked in as a graph constant and one compiled artifact
        # serves every rank. <<<
        rank_id = torch.tensor(
            [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
        )

        return NF.moe_tkg(
            hidden_input=hidden_states,
            expert_gate_up_weights=self._gate_up_weight(),
            expert_down_weights=self._down_weight(),
            expert_gate_up_weights_scale=self._gate_up_scale(),
            expert_down_weights_scale=self._down_scale(),
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            is_all_expert=True,
            rank_id=rank_id,
            # See ``_forward_prefill`` for why POST_SCALE reproduces the
            # reference's pre-down-projection scaling exactly.
            expert_affinities_scaling_mode=scale_mode.POST_SCALE,
            activation_fn=act_fn.SiLU,
            gate_clamp_upper_limit=gate_hi,
            gate_clamp_lower_limit=gate_lo,
            up_clamp_upper_limit=up_hi,
            up_clamp_lower_limit=up_lo,
            # Static control flow: the dynamic variant additionally requires
            # ``block_size`` to divide T into >= 2 blocks, which would tie
            # this module to the runner's bucket shapes.
            is_all_expert_dynamic=False,
            all_to_all_v_strategy=a2a.DISABLED,
            output_dtype=hidden_states.dtype,
        )

    def _ep_tp_group(self):
        """Return the group ``build_blockwise_mapping`` coordinates over.

        Under pure EP this is the (single-rank) TP sub-group *inside* the EP
        partition, not the world group — the mapping's ``moe_group`` is the
        group whose ranks split one expert's intermediate dimension
        (``functional/moe/moe_blockwise.py:41-52``), and under pure EP
        nothing is split, so the group is degenerate.
        """
        try:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_tp_group,
            )

            return get_neuron_ep_tp_group()
        except Exception:  # noqa: BLE001 - absent parallel state is not a failure
            from vllm.distributed.parallel_state import get_tp_group

            return get_tp_group()


# =============================================================================
# Section 2: Shared expert (block-128x128 FP8, 16-way subgroup)
# =============================================================================


class DeepseekV4SharedExpert(nn.Module):
    """The single always-on expert, run on a 16-way TP subgroup.

    >>> PARALLELISM: 16-way, NOT 64-way, and replicated four times across
    the TP group. This is a deliberate, plan-level choice, and the reason is
    numerical, not performance: ``down_proj`` is row-parallel, so a 64-way
    split would give ``K_local = 2048 / 64 = 32``, which cuts the
    128-element group the block-FP8 path quantizes activations over
    (``config.py:111-119``). 16 ways gives ``K_local = 128`` — exactly one
    activation quantization group, so the group boundary sits where the
    reference's does and the numerics do not drift.

    Consequently the ``down_proj`` result is all-reduced over
    ``shared_group`` (16 ranks) and NOT over the full 64-rank TP group. The
    four subgroups each compute the same complete sum independently, so
    every rank ends up with the identical replicated result — which is what
    lets the caller add it straight onto the EP-all-reduced routed output
    with no further collective. <<<

    <-- MODEL-SPECIFIC: same SwiGLU body as a routed expert
    (``dsv4_ref/model.py:632`` constructs it as an ``Expert``), including
    the asymmetric ``swiglu_limit`` clamp, but with ``weights=None`` so no
    routing weight and no ``routed_scaling_factor`` is applied
    (``dsv4_ref/model.py:648``: ``y += self.shared_experts(x)``).
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        shared_group,
        shared_group_rank: int,
        shared_group_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # >>> PARALLELISM: the subgroup is built ONCE in the model backbone
        # and passed down; never construct a process group per layer. <<<
        self.shared_group = shared_group
        self.shared_group_rank = int(shared_group_rank)
        self.shared_group_size = int(shared_group_size)

        self.hidden_size = config.hidden_size
        # <-- MODEL-SPECIFIC: n_shared_experts == 1, asserted upstream
        # (``dsv4_ref/model.py:631``), so the shared intermediate is one
        # ``moe_intermediate_size``.
        self.intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.swiglu_limit = float(config.swiglu_limit)

        if self.intermediate_size % self.shared_group_size != 0:
            raise ValueError(
                f"shared_group_size={self.shared_group_size} must divide the shared "
                f"expert intermediate size {self.intermediate_size}."
            )
        self.intermediate_size_per_rank = (
            self.intermediate_size // self.shared_group_size
        )

        block_n, block_k = _BLOCK_FP8_BLOCK_SIZE

        # <-- MODEL-SPECIFIC: w1 = gate, w3 = up, both column-parallel over
        # the intermediate dim; ``[out, in]`` orientation as stored
        # (``dsv4_ref/model.py:146-150``).
        gate_up_shape = (self.intermediate_size_per_rank, self.hidden_size)
        gate_up_scale_shape = (
            math.ceil(self.intermediate_size_per_rank / block_n),
            math.ceil(self.hidden_size / block_k),
        )
        self.w1_weight = nn.Parameter(
            torch.empty(*gate_up_shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w1_scale = nn.Parameter(
            torch.empty(*gate_up_scale_shape, dtype=torch.float32), requires_grad=False
        )
        self.w3_weight = nn.Parameter(
            torch.empty(*gate_up_shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w3_scale = nn.Parameter(
            torch.empty(*gate_up_scale_shape, dtype=torch.float32), requires_grad=False
        )

        # <-- MODEL-SPECIFIC: w2 = down, row-parallel over the intermediate
        # dim. K_local = 128 by construction; see the class docstring.
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.intermediate_size_per_rank,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.w2_scale = nn.Parameter(
            torch.empty(
                math.ceil(self.hidden_size / block_n),
                math.ceil(self.intermediate_size_per_rank / block_k),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def _linear(self, x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
        """One block-FP8 leg, in the exact frozen call form of contract §5."""
        return NF.block_fp8_linear(
            x,
            weight,
            scale,
            block_size=_BLOCK_FP8_BLOCK_SIZE,
            act_group_size=_BLOCK_FP8_ACT_GROUP_SIZE,
            accum_dtype=torch.float32,
            out_dtype=torch.bfloat16,
            bias=None,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return the shared-expert contribution, ``[T, hidden_size]`` bf16.

        Reproduces ``dsv4_ref/model.py:601-612`` step for step: gate and up
        in fp32, the asymmetric clamp, ``silu(gate) * up``, a cast back to
        the input dtype before the down projection, and no routing weight.
        """
        gate = self._linear(hidden_states, self.w1_weight, self.w1_scale).to(
            torch.float32
        )
        up = self._linear(hidden_states, self.w3_weight, self.w3_scale).to(torch.float32)

        # <-- MODEL-SPECIFIC: asymmetric clamp — ``up`` on both sides,
        # ``gate`` on the upper side only (``dsv4_ref/model.py:605-607``).
        if self.swiglu_limit > 0.0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)

        # ``dsv4_ref/model.py:608`` then ``:611`` — the cast back to the
        # activation dtype happens BEFORE the down projection, so the
        # down-projection input is quantized from bf16, not from fp32.
        act = (F.silu(gate) * up).to(hidden_states.dtype)

        output = self._linear(act, self.w2_weight, self.w2_scale)

        # >>> PARALLELISM: reduce over the 16-rank subgroup ONLY. Each rank
        # holds a 128-wide slice of the 2048 intermediate, so the 16 partial
        # down-projections sum to the full result; the four subgroups each
        # produce that same full result independently, leaving it replicated
        # across all 64 ranks with no 64-way collective. <<<
        if self.shared_group is not None and self.shared_group_size > 1:
            output = output.contiguous()
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.shared_group)
        return output


# =============================================================================
# Section 3: The MoE block (router + routed experts + shared expert)
# =============================================================================


class DeepseekV4MoE(nn.Module):
    """Router plus both expert paths for one decoder layer.

    <-- MODEL-SPECIFIC: two routing regimes coexist in one architecture.

    * Layers 0-2 (``config.is_hash_moe_layer``) are *hash* layers. Expert
      selection is a pure vocabulary-indexed gather, ``tid2eid[input_ids]``
      (``dsv4_ref/model.py:581-582``) — no top-k, no comparison, no bias.
      That is exactly why ``ffn.gate.bias`` is ABSENT from layers 0, 1 and 2
      in the checkpoint while ``tid2eid`` is present only there: there is
      nothing for a selection bias to bias. The router *weight* is still
      present and still used, because the routing WEIGHTS come from the
      score at the hash-selected experts (``dsv4_ref/model.py:585``).
    * Layers 3-42 select with ``noaux_tc``: score with
      ``sqrt(softplus(logits))``, add ``gate_bias`` to a copy, take the top
      6 of that biased copy, then read the weights back out of the
      UNBIASED scores.

    Both regimes then renormalize and scale identically.

    >>> PARALLELISM: the router is fully replicated — ``gate_weight`` is
    ``[256, 4096]`` on every rank and every rank computes the same routing
    decision for every token. That is what makes the EP combine a plain
    all-reduce with no all-to-all dispatch. <<<
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        shared_group,
        shared_group_rank: int,
        shared_group_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.scoring_func = config.scoring_func
        self.is_hash_layer = config.is_hash_moe_layer(layer_idx)

        # <-- MODEL-SPECIFIC: the reference renormalizes the top-k weights
        # whenever scoring is NOT softmax (``dsv4_ref/model.py:586-587``);
        # it has no ``norm_topk_prob`` field at all. This checkpoint's
        # ``norm_topk_prob=True`` agrees with that rule for
        # ``sqrtsoftplus``, so the two readings coincide here. The
        # reference's condition is the one implemented, and a checkpoint
        # that disagreed would be caught below rather than silently
        # diverging.
        self._renormalize = self.scoring_func != "softmax"
        if self._renormalize != bool(config.norm_topk_prob):
            raise ValueError(
                f"norm_topk_prob={config.norm_topk_prob} contradicts "
                f"scoring_func={self.scoring_func!r}. DeepSeek-V4's reference "
                "implementation renormalizes the top-k weights exactly when "
                "scoring is not softmax; a checkpoint that disagrees needs an "
                "explicit decision, not a silent choice."
            )
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"DeepseekV4 on Neuron implements scoring_func='sqrtsoftplus'; "
                f"got {self.scoring_func!r}."
            )

        # >>> PARALLELISM: router weights replicated on all ranks <<<
        self.gate_weight = nn.Parameter(
            torch.empty(
                self.n_routed_experts, self.hidden_size, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )

        # <-- MODEL-SPECIFIC: selection bias exists ONLY on the non-hash
        # layers; ``None`` here is load-bearing, not a default.
        if self.is_hash_layer:
            self.gate_bias = None
            # <-- MODEL-SPECIFIC: ``[vocab_size, num_experts_per_tok]``. The
            # checkpoint stores it int64; the reference declares int32
            # (``dsv4_ref/model.py:564``). Only the storage width differs —
            # both are exact expert ids and index identically.
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int64,
                ),
                requires_grad=False,
            )
        else:
            self.gate_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            self.tid2eid = None

        ep_degree, ep_rank, ep_group = _resolve_ep(config)
        self.experts = DeepseekV4RoutedExperts(
            config,
            layer_idx,
            ep_degree=ep_degree,
            ep_rank=ep_rank,
            ep_group=ep_group,
        )
        self.shared_expert = DeepseekV4SharedExpert(
            config,
            layer_idx,
            shared_group=shared_group,
            shared_group_rank=shared_group_rank,
            shared_group_size=shared_group_size,
        )

        self._attach_loaders(
            ep_degree=ep_degree,
            ep_rank=ep_rank,
            shared_group_rank=shared_group_rank,
            shared_group_size=shared_group_size,
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _attach_loaders(
        self,
        *,
        ep_degree: int,
        ep_rank: int,
        shared_group_rank: int,
        shared_group_size: int,
    ) -> None:
        """Attach the checkpoint loaders for this whole MoE subtree.

        All loader logic lives in ``weight_loaders.py``; this module only
        declares the parameters and names the one public entry point that
        binds to them. The call covers the router, the routed experts and
        the shared expert in one pass, so it must run after every
        ``nn.Parameter`` above exists.

        The import is deferred and the absence of the module tolerated: it
        is authored concurrently, and a missing loader module must not stop
        this module from importing or from being constructed in a test.
        """
        try:
            from .weight_loaders import attach_moe_loaders
        except ImportError:  # pragma: no cover - concurrent authoring window
            logger.warning(
                "vllm_neuron.model.deepseek_v4.weight_loaders.attach_moe_loaders is "
                "not available; DeepseekV4 MoE parameters for layer %d have no "
                "checkpoint loaders attached and will not load.",
                self.layer_idx,
            )
            return

        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        attach_moe_loaders(
            self,
            self.config,
            layer_idx=self.layer_idx,
            tp_size=tp_group.world_size,
            tp_rank=tp_group.rank_in_group,
            ep_degree=ep_degree,
            ep_rank=ep_rank,
            shared_tp_size=shared_group_size,
            shared_tp_rank=shared_group_rank,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _route(
        self, hidden_states: Tensor, input_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(dense_affinities [T, E] fp32, expert_index [T, K] int32)``.

        Line-for-line transcription of ``Gate.forward``
        (``dsv4_ref/model.py:569-589``). Every step is annotated with the
        reference line that fixes it, because several of them are easy to
        get subtly wrong:

        1. ``:570`` the router matmul runs in fp32 on fp32-upcast weights.
        2. ``:576`` ``scores = softplus(logits).sqrt()``.
        3. ``:577`` the UNBIASED scores are kept aside.
        4. ``:579-580`` ``gate_bias`` is added to a separate tensor.
        5. ``:581-584`` selection: hash gather, or top-k of the BIASED score.
        6. ``:585`` the weights are gathered from the UNBIASED scores.
        7. ``:586-587`` renormalize so the K weights sum to 1.
        8. ``:588`` multiply by ``routed_scaling_factor``.

        Every op is static-shape: ``matmul``, ``softplus``, ``sqrt``,
        ``index_select``, ``topk``, ``gather``, ``scatter``. No ``.item()``,
        no ``nonzero()``, no boolean-mask indexing.
        """
        # Step 1 (:570) — fp32 router matmul. The weight is stored
        # ``[E, H]``, so the contraction transposes it.
        logits = torch.matmul(
            hidden_states.to(torch.float32), self.gate_weight.to(torch.float32).t()
        )

        # Step 2 (:576) — sqrtsoftplus.
        scores = torch.sqrt(F.softplus(logits))

        # Step 3 (:577) — keep the uncorrected scores; they alone produce
        # the routing weights.
        original_scores = scores

        # Step 4 (:579-580) — the bias shifts the SELECTION score only.
        if self.gate_bias is not None:
            scores = scores + self.gate_bias.to(torch.float32)

        # Step 5 (:581-584) — selection.
        if self.is_hash_layer:
            # <-- MODEL-SPECIFIC: a pure gather. No logits, no comparison,
            # no bias — which is why layers 0-2 carry no ``gate.bias``.
            # ``index_select`` on a flattened int64 index keeps the shape
            # static and avoids advanced indexing.
            flat_ids = input_ids.reshape(-1).to(torch.int64)
            expert_index = self.tid2eid.index_select(0, flat_ids)
        else:
            expert_index = torch.topk(scores, self.num_experts_per_tok, dim=-1)[1]
        expert_index = expert_index.to(torch.int64)

        # Step 6 (:585) — weights come from the UNCORRECTED scores.
        weights = original_scores.gather(1, expert_index)

        # Step 7 (:586-587).
        if self._renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # Step 8 (:588) — ``routed_scaling_factor`` (the reference's
        # ``route_scale``, 1.5) is applied unconditionally, to the routed
        # weights only. It never touches the shared expert.
        weights = weights * self.routed_scaling_factor

        # Densify to ``[T, E]`` because both MoE entry points take the
        # affinity as a dense per-expert vector, zero at unselected experts
        # (``functional/moe/moe_cte.py:114-116``,
        # ``functional/moe/moe_tkg.py:74``). ``scatter`` is out-of-place and
        # static-shape.
        dense = torch.zeros(
            hidden_states.shape[0],
            self.n_routed_experts,
            dtype=torch.float32,
            device=hidden_states.device,
        ).scatter(1, expert_index, weights.to(torch.float32))

        return dense, expert_index.to(torch.int32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, hidden_states: Tensor, input_ids: Tensor, is_prefill: bool
    ) -> Tensor:
        """Route, run both expert paths, and sum them.

        Args:
            hidden_states: ``[T, hidden_size]``, already ``ffn_norm``-ed by
                the decoder layer. This is the hc-REDUCED single stream, not
                the ``hc_mult``-wide residual.
            input_ids: ``[T]`` (or any shape flattening to ``T``) token ids.
                Every layer receives them; only the three hash layers read
                them (``dsv4_ref/model.py:637``).
            is_prefill: Python bool selecting the expert kernel. Never a
                tensor — the branch must resolve at trace time.

        Returns:
            ``[T, hidden_size]`` in the input dtype.
        """
        if self.is_hash_layer and input_ids is None:
            raise ValueError(
                f"DeepseekV4 layer {self.layer_idx} is a hash-MoE layer and routes "
                "through tid2eid[input_ids]; input_ids must not be None."
            )

        expert_affinities, expert_index = self._route(hidden_states, input_ids)

        routed = self.experts(
            hidden_states, expert_affinities, expert_index, is_prefill
        )
        shared = self.shared_expert(hidden_states)

        # <-- MODEL-SPECIFIC: a plain, unscaled add
        # (``dsv4_ref/model.py:648``: ``y += self.shared_experts(x)``).
        # ``routed_scaling_factor`` is already inside ``routed`` via the
        # routing weights and must NOT be applied again here or to
        # ``shared``.
        return (routed + shared).to(hidden_states.dtype)
