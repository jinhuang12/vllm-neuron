# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 MoE block
=====================

Three modules, in the order the data flows through them:

* :class:`DeepseekV4MoE` — the router. Owns ``gate_weight``, the optional
  ``gate_bias`` and, on the three hash layers only, the ``tid2eid`` table.
  It produces the per-token expert selection and weights, drives the routed
  experts and the shared expert, and sums the two contributions.
* :class:`DeepseekV4RoutedExperts` — the 256 routed experts, 1-byte FP8 with
  one power-of-two scale per output channel (LD-80 restored the operator's
  recorded fp8 container; the LD-67 bf16 fold is retired — NOT MX: the MX
  expert kernels are NeuronCore-v4 and this venue is v3), four of them
  resident per core at ``ep_degree=64``.
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


# RETIRED (LD-21): ``_MX_GROUP_SIZE``, ``_MX_ELEMS_PER_WORD``,
# ``_MX_TILE_ROWS`` and ``_MX_TILE_K`` parameterized the MX tiled layouts.
# The checkpoint's group-32 E8M0 grid still exists on disk — the loader reads
# it (``weight_loaders.py`` ``_MX_GROUP``) — but this module never sees it: the
# groups are folded into one per-output-channel scale at load time (LD-23), so
# no group extent reaches the forward. The MX expert kernels are
# NeuronCore-v4 and cannot execute on this campaign's trn2 (= v3) venue at all
# (R-13), so the routed experts now take the gen3-legal PLAIN layouts and there
# is no tiling left to parameterize. Kept as a comment, not as dead constants
# that read like a live contract.

# >>> PARALLELISM: block extent for the prefill blockwise mapping. Must be a
# multiple of 128 (``functional/moe/moe_cte.py:198``). 256 is what gpt_oss
# uses at the same hidden size, so the block-count arithmetic is already
# exercised at this shape. <<<
#
# LD-21: 256 is now also MANDATORY, not merely convenient. The gen3-legal
# quantized prefill implementation is ``shard_on_i``, whose own compatibility
# check asserts ``block_size % 256 == 0``
# (``nkilib/core/moe/moe_cte/bwmm_shard_on_I.py:660``, reproduced verbatim in
# ``artifacts/repairs/author_model_family-iter3/iter3-moe-gen3-probe.txt``
# finding B). Lowering this to 128 does not degrade — it fails kernel
# validation.
_MOE_CTE_BLOCK_SIZE = 256

# The block-FP8 contract for the shared expert, frozen by the family
# interface contract §5. Every leg of the shared-expert MLP is called with
# exactly these arguments.
_BLOCK_FP8_BLOCK_SIZE = (128, 128)
_BLOCK_FP8_ACT_GROUP_SIZE = 128

# LD-60 / LD-59 -- the arithmetic realization of ``scoring_func="sqrtsoftplus"``
# at :_route step 2. NOT a semantics knob: ``config.py:124`` still pins
# ``scoring_func="sqrtsoftplus"`` and the reject in :DeepseekV4MoE.__init__ (the
# ``if self.scoring_func != "sqrtsoftplus": raise``) still stands. These
# constants exist so the same MATHEMATICAL function can be computed without
# asking the [Act] engine for a ``Softplus``, a ``Sqrt`` or a ``Ln`` table
# (NCC_INLA001 "the number of activation tables must be <= 8"; port-plan.md
# §3.3-3.4, ladder decisions LD-59 and LD-60).
#
# ``atanh(t)/t == 1 + t**2/3 + t**4/5 + ... `` -- the reciprocal odd integers,
# truncated after ``t**12/13``. Plain Taylor, NOT minimax-tuned: on the reduced
# range ``t**2 <= 1/9`` the truncated tail is already about ``t**14/15 ==
# 1.7e-08``, below fp32 resolution (6e-08), and a tuned set would trade an
# auditable derivation for a marginal bound. Graded by G1 (port-plan.md §3.5).
_ATANH_C0 = 1.0
_ATANH_C1 = 1.0 / 3.0
_ATANH_C2 = 1.0 / 5.0
_ATANH_C3 = 1.0 / 7.0
_ATANH_C4 = 1.0 / 9.0
_ATANH_C5 = 1.0 / 11.0
_ATANH_C6 = 1.0 / 13.0

# LD-59 zero guard for ``y * rsqrt(y)``. This is the SMALLEST NORMAL fp32 value,
# and that is the largest choice that is also the smallest: it must be normal
# (so it survives a subnormal flush-to-zero, which a device may do and which no
# off-hardware run can rule out), and every value above it costs accuracy for
# nothing, because a normal ``y`` needs no clamping at all.
#
# THE BOUND THE PLAN REQUIRES RECORDED. ``r_max = rsqrt(2**-126) == 2**63 ==
# 9.223372e+18``, so even if the compiler reassociates the Newton step into a
# literal ``r*r``, that product is ``2**126 == 8.507059e+37``, under fp32 max
# 3.4028235e+38 with 4x headroom. The source below parenthesises ``(y * r)``
# first, so the product ``r*r`` is not formed AS WRITTEN either -- ``y * r`` is
# approximately ``sqrt(y)`` and cannot overflow. Both readings are safe.
#
# WHY NOT ``1e-30``, WHICH THIS FILE SHIPPED FIRST: ``1e-30`` is a NORMAL fp32
# number, so it clamped normal, accurately-representable ``y`` values across the
# whole band ``1e-38 < y < 1e-30`` (``-87 < logits < -69``) where the fp32
# reference is trustworthy to 2.8e-08. Measured: it drove the relative error to
# ~1.0 there. That was an authoring defect, found by the G1 leg of the gate, and
# it is this constant alone. See ``artifacts/repairs/author_model_family-iter17/``.
#
# The remaining deviation is ``0 < y < 2**-126``, i.e. SUBNORMAL ``y`` only,
# reached at ``logits < -87.3``. There the result is ``y * 1.5 * 2**63`` rather
# than ``sqrt(y)``. That band is exactly the band in which the fp32 reference
# ``torch.sqrt(F.softplus(x))`` is NOT A FUNCTION OF ITS INPUT -- measured, it
# returns 28 grid units for a large tensor and 27 for a size-1 tensor at the
# same ``x = -100.0`` -- so nothing there is worth matching. Both the deviating
# and the reference value are below 1e-19, and G4 measures 0 top-k flips of 4096
# in all 12 probed regimes, which is the only thing ``scores`` decides.
# At EXACTLY ``y == 0`` the outer ``y *`` restores the exact ``0.0`` (G3).
_SQRT_GUARD = 2.0 ** -126


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


def _nki_output_dtype(torch_dtype: torch.dtype):
    """Map a torch dtype to the NKI dtype ``nkilib``'s ``moe_tkg`` requires.

    ``moe_tkg`` sizes its output allocation with
    ``sizeinbytes(output_dtype)`` (``nkilib/core/moe/moe_tkg/moe_tkg.py:264``),
    which ``kernel_assert``s "dtype size unknown!" on a torch dtype
    (``nkilib/core/utils/allocator.py:54``). Deferred import for the same
    reason as :func:`_moe_kernel_enums`: this module must import on a host with
    no Neuron toolchain.

    Only the dtypes this family's activations can actually be are mapped. An
    unmapped dtype raises here rather than reaching the kernel's assert, so the
    error names the model's dtype instead of the allocator's internals.
    """
    import nki.language as nl

    mapping = {
        torch.bfloat16: nl.bfloat16,
        torch.float32: nl.float32,
        torch.float16: nl.float16,
    }
    try:
        return mapping[torch_dtype]
    except KeyError:
        raise ValueError(
            f"DeepseekV4 MoE decode cannot map activation dtype {torch_dtype} "
            "to an NKI output dtype for moe_tkg. Expected bfloat16 (the served "
            "dtype), float32 or float16."
        ) from None


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
# Section 1: Routed experts (FP8 per-output-channel, expert-parallel)
# =============================================================================


class DeepseekV4RoutedExperts(nn.Module):
    """The 256 routed experts, four resident per core at ``ep_degree=64``.

    >>> PARALLELISM: pure EP. Each core owns a disjoint contiguous quarter-
    percent of the expert set with the FULL intermediate dimension (2048),
    so no intra-expert TP sharding and no gather/scatter of activations is
    needed — only one all-reduce of the summed expert output at the end. <<<

    <-- MODEL-SPECIFIC: 256 experts, top-6, SwiGLU with the asymmetric
    ``swiglu_limit`` clamp, ``w1`` = gate / ``w3`` = up / ``w2`` = down.

    <-- FP8 PER-CHANNEL (LD-21/LD-23; residency RESTORED by LD-80, retiring
    the LD-67 bf16 fold): the checkpoint stores these experts as MXFP4
    (``float4_e2m1fn_x2`` elements, ``[out, in // 32]`` E8M0 scales —
    ``dsv4_ref/model.py:140-145``). Trainium2 has no FP4 datapath AND no MX
    datapath: ``nisa.nc_matmul_mx`` / ``nisa.quantize_mx`` are NeuronCore-v4
    instructions, and this venue is v3 (R-13). So the loader requantizes to
    1-byte legacy-E4M3 elements plus ONE fp32 power-of-two scale per output
    channel, which is the form the gen3-legal MoE entry points take:
    ``NF.moe_cte`` non-MX ``shard_on_i`` (PER_CHANNEL) at prefill and
    ``NF.moe_tkg`` ``QuantizationType.ROW`` at decode. Both were proven on the
    installed wheel before this class was written —
    ``artifacts/repairs/author_model_family-iter3/iter3-moe-gen3-probe.txt``.
    The LD-67 fold (Amendment 7) briefly emitted these weights as bf16 with
    the scales folded in; it cost +4.31 GiB/core of resident weight I/O, its
    flood-class justification was void on the record (F-218 — the LD-73
    sentinel mask on the index stream is the actual containment, and it
    STANDS), and LD-80 (plan §21.2) retired it by restoring this container.

    THE 1-BYTE DTYPE IS A CARRIER, NOT AN ENCODING CLAIM. The parameters are
    declared ``torch.float8_e4m3fn`` because that is the only 1-byte float
    torch has, but the BYTES inside them are legacy ``nl.float8_e4m3`` (bias 7,
    amax 240), written by the loader from a byte table and exponent-field
    arithmetic. That is correct rather than merely tolerated: on trn2 the
    plugin maps ``torch.float8_e4m3fn`` to ``nl.float8_e4m3`` in both
    directions (``nki/nki_dtype.py:43,51-53``), so the kernel decodes exactly
    the encoding the loader wrote. Never convert these tensors through torch's
    own fp8 cast — that is the OCP ``e4m3fn`` encoding (amax 448) and would
    reinterpret every exponent.

    Parameter shapes are the *contraction-first* ``[E_local, in, out]``
    orientation, which is what the kernels take (``gate_up`` ``[E, H, 2, I]``,
    ``down`` ``[E, I, H]``). The loader writes them already transposed, because
    this class reinterprets with pure ``view`` calls — no runtime repacking —
    and a ``view`` cannot transpose.
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

        # LADDER-DECISION LD-74 (E1, assessment §16.3; plan §18.2): the EP rank
        # enters the traced graph as a NON-PERSISTENT module buffer, never as a
        # per-call ``torch.tensor([[int]])`` whose VALUE bakes into the render
        # as a per-rank literal. A buffer renders as a value-free ``get_attr``
        # node, so the 8 ranks of one oproj tile trace byte-identical graphs
        # and share one compile key (F-235). ``device="cpu"`` IS LOAD-BEARING
        # (plan §18.2 item 5): the runner constructs this family inside
        # ``torch.device("meta")`` (``neuron_model_runner.py:1195``) and a
        # non-persistent buffer has NO checkpoint key to load over — an
        # unnamed device would leave it a valueless meta tensor and
        # ``.to(device)`` would raise after load. Same convention as the RoPE
        # tables (``model.py:216-232``).
        self.register_buffer(
            "ep_rank_buf",
            torch.tensor([[self.ep_rank]], dtype=torch.int32, device="cpu"),
            persistent=False,
        )

        # <-- FP8 PER-CHANNEL: w1 = gate projection, w3 = up projection
        # (``dsv4_ref/model.py:602-603``: ``gate = self.w1(x)``,
        # ``up = self.w3(x)``). Logically ``[out=I, in=H]``; stored
        # CONTRACTION-FIRST as ``[in=H, out=I]`` because the kernels take
        # ``gate_up`` as ``[E, H, 2, I]`` and this class only ``view``s.
        # The loader writes the transpose (LD-23; expert residency restored
        # to this fp8-per-channel container by LD-80, retiring the LD-67
        # bf16 fold).
        expert_shape = (self.num_local_experts, self.hidden_size, self.intermediate_size)
        # ONE fp32 power-of-two multiplier per OUTPUT channel. Not a group
        # grid: the gen3-legal kernels take PER_CHANNEL / ROW scales only, and
        # the group-32 exponents were folded into these at load time.
        scale_shape = (self.num_local_experts, self.intermediate_size)
        self.w1_weight = nn.Parameter(
            torch.empty(*expert_shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w1_scale = nn.Parameter(
            torch.empty(*scale_shape, dtype=torch.float32), requires_grad=False
        )
        self.w3_weight = nn.Parameter(
            torch.empty(*expert_shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w3_scale = nn.Parameter(
            torch.empty(*scale_shape, dtype=torch.float32), requires_grad=False
        )

        # <-- FP8 PER-CHANNEL: w2 = down projection, logically ``[out=H, in=I]``,
        # stored contraction-first as ``[in=I, out=H]`` — which is exactly the
        # ``down`` layout the kernels document, so no fuse and no view here.
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.w2_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts, self.hidden_size, dtype=torch.float32
            ),
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    # Kernel layout reinterpretation (views + one fuse; no repacking)
    # ------------------------------------------------------------------
    def _gate_up_weight(self) -> Tensor:
        """Fuse ``w1``/``w3`` into the ``[E_L, H, 2, I]`` kernel buffer.

        Both MoE entry points take gate and up as one tensor with the gate/up
        axis at dim 2 (``functional/moe/moe_tkg.py:70``,
        ``functional/moe/moe_cte.py:119``). Each parameter is already
        ``[E, H, I]``, so the fuse is one ``view`` to insert the axis plus one
        concatenation.

        RECORDED COST: that concatenation is a real copy on every forward
        (~32 MiB per layer at 1 byte per element). It exists only because the
        interface contract fixes ``w1_weight`` and ``w3_weight`` as separate
        parameters; a fused ``gate_up`` parameter emitted by
        ``attach_moe_loaders`` would remove it entirely. Flagged rather than
        silently absorbed.
        """
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        gate = self.w1_weight.view(e, h, 1, i)
        up = self.w3_weight.view(e, h, 1, i)
        return torch.cat([gate, up], dim=2)

    def _down_weight(self) -> Tensor:
        """``w2`` already IS the ``[E_L, I, H]`` ``down`` layout — no view."""
        return self.w2_weight

    def _gate_up_scale_row(self) -> Tensor:
        """``moe_tkg`` ROW gate/up scale: ``[E_L, 2, I]``.

        ``QuantizationType.ROW`` is selected by the mere PRESENCE of a weight
        scale (``functional/moe/moe_tkg.py:390-410``), and its gate/up scale
        carries the gate/up axis at dim 1 — the same axis convention as the
        weight, one dimension shorter.
        """
        return torch.stack([self.w1_scale, self.w3_scale], dim=1)

    def _down_scale_row(self) -> Tensor:
        """``moe_tkg`` ROW down scale: ``[E_L, H]`` — the parameter as stored."""
        return self.w2_scale

    def _gate_up_scale_per_channel(self) -> Tensor:
        """``moe_cte`` PER_CHANNEL gate/up scale: ``[E_L, 1, 2*I]``.

        The kernel flattens the fused weight's last two axes, so the fused
        output-channel axis runs ``[gate_0..gate_{I-1}, up_0..up_{I-1}]`` and
        the scale must concatenate in exactly that order — a ``stack`` would
        interleave and silently mis-scale every channel.
        """
        return torch.cat([self.w1_scale, self.w3_scale], dim=1).unsqueeze(1)

    def _down_scale_per_channel(self) -> Tensor:
        """``moe_cte`` PER_CHANNEL down scale: ``[E_L, 1, H]``."""
        return self.w2_scale.unsqueeze(1)

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
        # R-17 GUARD (restored by LD-80 with the fp8-per-channel container).
        # ``_can_use_moe_cte_kernel``'s non-MX branch checks
        # neither dtype nor scales (``functional/moe/moe_cte.py:584-615``): it
        # returns ``can_run_kernel(hidden_states)``, which is also False under
        # ``VLLM_NEURON_DISABLE_NKI_KERNELS``. When it is False the call falls
        # through to ``_torch_moe_impl`` (``moe_cte.py:448-461``), which takes
        # NO scale arguments at all — so fp8 experts would be multiplied as raw
        # E4M3 mantissas with every per-channel multiplier dropped. That is
        # SILENTLY WRONG output, not an error, and the port plan records the
        # rule as hard: never route fp8 weights to ``_torch_moe_impl``.
        # Convert the silence into a refusal here, at the only call site that
        # can.
        # The guard is conditioned on the WEIGHTS being quantized rather than
        # on the venue alone, because the harm is specific to dropped scales: a
        # bf16 expert set has no scales to drop and reaching the torch fallback
        # with one is correct, which is what makes the bf16 dataflow A/B
        # (port-plan.md section 2 check 2) possible on CPU at all.
        from vllm_neuron.functional.moe import moe_cte as _cte_mod

        gate_up_weight = self._gate_up_weight()
        gate_up_scale = self._gate_up_scale_per_channel()
        if gate_up_scale is not None and not _cte_mod.can_run_kernel(hidden_states):
            raise RuntimeError(
                "DeepseekV4RoutedExperts prefill cannot run the NKI moe_cte "
                "kernel on this venue, and its torch fallback "
                "(_torch_moe_impl) accepts no weight scales. These experts are "
                f"{gate_up_weight.dtype} with per-output-channel scales "
                f"{tuple(gate_up_scale.shape)}, so falling back would silently "
                "drop every multiplier and produce wrong output with no error "
                "(R-17). Refusing. Run the prefill path on a Neuron device "
                "with the NKI kernels enabled, or exercise this module under "
                "the NKI CPU simulator (VLLM_NEURON_CPU_MODE=1 with "
                "NKI_SIMULATOR=1)."
            )

        impl, act_fn, scale_mode, _ = _moe_kernel_enums()
        gate_hi, gate_lo, up_hi, up_lo = self._clamps()

        # >>> PARALLELISM: slice the dense global affinities down to this
        # rank's experts; ``build_blockwise_mapping`` wants ``[T, E_local]``
        # (``functional/moe/moe_blockwise.py:36``). <<<
        local_affinities = expert_affinities
        if self.ep_degree > 1:
            # LADDER-DECISION LD-74 (E2): the index vector is derived from the
            # ``ep_rank_buf`` buffer (value-free ``get_attr``) with STATIC
            # arange arguments, replacing a rank-VALUED
            # ``torch.arange(local_expert_start, …)`` whose endpoints rendered
            # as per-rank literals — one traced graph per oproj tile instead
            # of one per rank. The consuming
            # ``NF.get_local_expert_affinities`` gather ALREADY exists below:
            # NO new indirect op on this path (plan §18.2 item 2). Numerics:
            # index-set equality is exhaustive over all 64 ranks (§18.3
            # item 3(i)).
            local_expert_indices = self.ep_rank_buf.reshape(
                ()
            ) * self.num_local_experts + torch.arange(
                0,
                self.num_local_experts,
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

        # --- LD-73 SENTINEL MASK, prefill (CTE) family --------------------
        # LADDER-DECISION LD-73 (assessment §15.3; plan §17.2). DEPLOYED-PIN
        # WORKAROUND for defect (A), never a fix: F-229 — router-path
        # dense-affinity-zero -> all-padding blockwise mapping (the
        # -1/0xFFFFFFFF sentinel class) -> ISA-mandated vector-DGE OOB
        # account -> NRT_EXEC_OOB=1006 on every completed prefill warmup
        # execute (serve-26). The kernel-side mask route is vendor-locked at
        # the deployed pin (F-231: ``bwmm_shard_on_I`` weight gathers are
        # forced ``oob_mode=error``, kernel_assert :669); the vendor's
        # ``ignore`` oob_mode is post-pin (aws-neuron-sdk#1335 family). The
        # in-fork precedent for this fork-side class is the gpt_oss
        # clamp-plus-collapse (``gpt_oss/model_bf16.py:1535-1554``, naming
        # "nrta 1006" verbatim) — never a bare clamp.
        #
        # TENSOR SET, derived from the kernel's own contract (the pinned
        # descriptor census ITER-60-RAW-P609, ``bwmm_shard_on_I.py`` sha
        # 16ca33d6…):
        #  * ``token_position_to_id`` keys the vector-DGE descriptors —
        #    hidden gather :816, affinity gather :1775 (address
        #    ``t*E_local + block_expert``, ``functional/moe/moe_cte.py:
        #    114-116``), old-block loads :1851/:1867, output scatter :2491 —
        #    THE sentinel carrier. Every sentinel (negative int32, i.e. the
        #    -1/0xFFFFFFFF class) and any out-of-domain id is remapped to
        #    the kernel's OWN documented padding-sink index T
        #    (``functional/moe/moe_cte.py:112-113,127-128``: hidden
        #    ``[T+1, H]`` "reserves the last row as a padding sink";
        #    "Padding slots should map to index T").
        #  * ``hidden_states`` gains one ZERO sink row -> ``[T+1, H]``.
        #  * ``expert_affinities_masked`` gains ``E_local`` ZERO entries
        #    (rows ``T*E_local .. T*E_local+E_local-1`` — exactly the sink
        #    row's affinities under the ``[t*E + e]`` contract).
        #  * ``block_to_expert`` keys the ERROR-mode weight gathers
        #    (:1285/:1681/:1693/:2426): clamped into ``[0, E_local-1]`` to
        #    make the scalar-offset domain explicit. NOT a bare clamp: a
        #    degenerate block's positions are all sink by the tp2id remap,
        #    so no real output row can receive a substituted-expert
        #    contribution.
        #  * ``conditions`` / ``local_expert_indices``: flags resp. arange —
        #    no sentinel channel; untouched.
        #
        # EXACT-ZERO COLLAPSE: a masked position gathers the zero sink
        # hidden row, its affinity gather reads an exact zero, POST_SCALE
        # multiplies the contribution to exactly 0, and its output scatter
        # lands in the discarded sink row — real rows are untouched by
        # masked positions. The kernel launch below is unchanged (same
        # kernel, same flags; with no sentinel present the ``skip_token``
        # arm is simply never exercised). HONESTY BOUND (assessment §15.3
        # item 4): NO computation-neutrality claim — where (A)'s degenerate
        # mapping expresses, the masked graph computes DEFINED zero expert
        # contributions where the unmasked graph's behavior is undefined;
        # the detector is the parity verdict (F-224), never an in-graph
        # witness (Amendment 4).
        _t_real = hidden_states.shape[0]
        token_position_to_id = torch.clamp(
            torch.where(
                token_position_to_id < 0,
                torch.full_like(token_position_to_id, _t_real),
                token_position_to_id,
            ),
            min=0,
            max=_t_real,
        )
        block_to_expert = torch.clamp(block_to_expert, 0, self.num_local_experts - 1)
        hidden_states = torch.cat(
            [
                hidden_states,
                torch.zeros(
                    1,
                    hidden_states.shape[1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ),
            ],
            dim=0,
        )
        expert_affinities_masked = torch.cat(
            [
                expert_affinities_masked,
                torch.zeros(
                    self.num_local_experts,
                    1,
                    dtype=expert_affinities_masked.dtype,
                    device=expert_affinities_masked.device,
                ),
            ],
            dim=0,
        )
        # --- end LD-73 SENTINEL MASK ---------------------------------------

        output = NF.moe_cte(
            # LD-21: ``shard_on_i``, NOT ``shard_on_block``. Both accept fp8
            # weights with per-channel scales, but ``shard_on_block`` pins
            # ``gup_scale = down_scale = None`` under a "# Placeholder for FP8"
            # comment (``nkilib/core/moe/moe_cte/bwmm_shard_on_block.py:245-246``)
            # while still setting ``is_quant`` from scale presence (:216) — it
            # accepts the scales and never applies them. Measured on the
            # installed wheel: rel_err 2.1e+09 vs 5.2e-03 for ``shard_on_i`` at
            # the same shapes and inputs, with NO kernel error either way
            # (probe findings A and B). ``shard_on_i`` has the complete
            # plumbing (``bwmm_shard_on_I.py:304-305,999-1008,1137,1973-2040``).
            implementation=impl.shard_on_i,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_weight,
            down_proj_weight=self._down_weight(),
            gate_up_proj_scale=gate_up_scale,
            down_proj_scale=self._down_scale_per_channel(),
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
            # F-302 (SUPERSEDES LD-21). This was FALSE, and the comment that
            # justified it was right about the crash and wrong about the cost.
            # The plugin's trim branch under this flag
            # (``functional/moe/moe_cte.py``) keyed on the FLAG ALONE, so
            # ``shard_on_i``'s plain ``[T, H]`` return hit a trim shaped for
            # ``shard_on_block``'s ``[T, 2, H+E]`` and raised
            # ``IndexError: too many indices for tensor of dimension 2``.
            # Passing False dodged that crash -- and in doing so ALSO disabled
            # the vendor kernel's ACCUMULATION and its ZERO-INIT
            # (``bwmm_shard_on_I.py``: ``output[token_indices] += block_output``
            # only under this flag, and ``output_initialization(output)`` is
            # SKIPPED without it; the vendor's own default is True).
            #
            # Two consequences, one root cause: a token routed to >= 2 experts
            # hosted on the same rank kept ONLY THE LAST expert's contribution
            # (ep19 mechanism (a)), and rows no block wrote kept the PREVIOUS
            # execution's data, which the EP all-reduce then summed across 64
            # ranks into every token (mechanism (b), worked around downstream by
            # the zero-mask in :meth:`_forward_prefill`). Measured on the NKI CPU
            # simulator: rel_F 0.727 / 0.836 / 0.786 / 0.096 on the four
            # multi-expert cases, against a 2e-2 bound.
            #
            # The trim is now keyed on the IMPLEMENTATION, so True is correct
            # here rather than a crash.
            is_tensor_update_accumulating=True,
        )
        # LD-73: discard the padding-sink row. The kernel contract returns
        # rows matching its token extent; under the ``[T+1, H]`` sink
        # convention the last row is the masked positions' write target and
        # is dropped here. If the kernel returns ``[T, H]`` the slice is the
        # identity — safe either way. The CPU fallback (``_torch_moe_impl``)
        # derives its token count from the affinity extent (T+1) and returns
        # ``[T+1, H]``, so the trim keeps the fallback numerically matched.
        #
        # --- ep19 P35 mechanism (b) ZERO-MASK (ITER-07 §6) -----------------
        # On device, ``moe_cte`` (``bwmm_shard_on_I``) leaves output rows its
        # blockwise mapping does not write holding the PREVIOUS execution's
        # data (ITER-07-RAW-P35.attempt2: E_zero as the allocation's 1st
        # execution is exact zero; the same inputs as its 3rd execution read
        # back max_abs 4.156 on the idx-local rows), and the EP all-reduce in
        # :meth:`forward` SUMS 64 ranks' stale rows into every token. A token
        # whose local affinity row is all-zero contributes exactly zero from
        # this rank, so mask those rows to zero on EVERY execution,
        # independent of kernel buffer state. Plain traced torch ops — no
        # python-side conditional on data — so the mask lives in the compiled
        # graph. On CPU the fallback already writes zeros on those rows, so
        # this is a numerical no-op on the CPU leg (graded by the apply_fix
        # A/B). SCOPE: mechanism (b) only; the mapped-row numerics defect (a)
        # (rel_F 0.037 > the 2e-2 C1 bound, inside the shard_on_i compute)
        # is NOT addressed here and stays open at kernel granularity.
        _has_local = (local_affinities != 0).any(dim=-1, keepdim=True)
        return torch.where(
            _has_local, output[:_t_real], torch.zeros_like(output[:_t_real])
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

        # >>> PARALLELISM — LADDER-DECISION LD-74 (E1): ``rank_id`` is the
        # ``ep_rank_buf`` module buffer (value-free ``get_attr`` render), so
        # the EP rank is not baked into the graph as a VALUE literal and one
        # compiled artifact serves every rank of an oproj tile. The kernel
        # receives the identical ``[[ep_rank]]`` int32 runtime tensor. <<<
        rank_id = self.ep_rank_buf

        # --- LD-73 SENTINEL MASK, decode (TKG) family ----------------------
        # LADDER-DECISION LD-73 (assessment §15.3; plan §17.2) — the same
        # sentinel sanitization as the prefill family, same commit, applied
        # at this family's kernel boundary. Decode has never executed under
        # LD-71; if defect (A) expresses here the same way (F-229), an
        # unmasked decode path would move the NRT_EXEC_OOB=1006 expression
        # from prefill warmup to decode warmup.
        #
        # TENSOR SET, derived from the kernel's own contract:
        #  * ``expert_index`` ``[T, K]`` int32 global-domain is the ONE
        #    sentinel-capable index tensor handed to ``moe_tkg`` (the
        #    aws-neuron-sdk#1335 class: max-int sentinels from topk-family
        #    kernels read as 0xFFFFFFFF vector-DGE addresses). Clamped into
        #    ``[0, n_routed_experts-1]``. The zero-contribution collapse is
        #    STRUCTURAL here, not a bare clamp: per the kernel contract
        #    (``functional/moe/moe_tkg.py:99-101``) ``expert_index`` is
        #    consumed for compute only under ``mask_unselected_experts=
        #    True``; this call passes the default False with
        #    ``is_all_expert=True``, so contributions are driven entirely by
        #    ``expert_affinities`` (dense VALUES, zero at unselected — no
        #    index domain), which this mask never touches.
        #  * ``rank_id`` ``[1, 1]``: built from ``self.ep_rank`` (a Python
        #    int in ``[0, ep_degree)``) — in-bounds by construction; the
        #    kernel's affinity slice it keys (``functional/moe/moe_tkg.py:
        #    74-76``) has no sentinel channel. Untouched.
        # ``moe_tkg`` carries ZERO error-mode oob lines at the deployed pin
        # (ITER-60-RAW-P609 §5 control) and the decode flood-class was
        # closed statically at Q-D2/ITER-58 — this arm is the plan's
        # same-idiom defensive mask, keeping serve-27 single-variable at the
        # commit level. HONESTY BOUND: as with prefill, no
        # computation-neutrality claim; parity v3r1 is the arbiter (F-224).
        expert_index = torch.clamp(expert_index, 0, self.total_num_experts - 1)
        # --- end LD-73 SENTINEL MASK ---------------------------------------

        return NF.moe_tkg(
            hidden_input=hidden_states,
            expert_gate_up_weights=self._gate_up_weight(),
            expert_down_weights=self._down_weight(),
            # LD-21: presenting these two selects ``QuantizationType.ROW``
            # (``functional/moe/moe_tkg.py:390-410``), which the installed wheel
            # dequantizes correctly on a 1-byte fp8 carrier — rel_err 5.068e-03
            # against the dequantized reference, versus 4.470e-03 for the bf16
            # control at the same shapes (probe finding C, leg P3c).
            # (ROW scales restored by LD-80, retiring the LD-67 scale-free form.)
            expert_gate_up_weights_scale=self._gate_up_scale_row(),
            expert_down_weights_scale=self._down_scale_row(),
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
            # This must be an NKI dtype, NOT a torch dtype. ``nkilib``'s
            # ``moe_tkg`` calls ``sizeinbytes(output_dtype)``
            # (``nkilib/core/moe/moe_tkg/moe_tkg.py:264``) and a torch dtype
            # trips ``kernel_assert`` there with "dtype size unknown!
            # torch.bfloat16" (``nkilib/core/utils/allocator.py:54``). Line 264
            # precedes every other validation in that kernel, so this is the
            # FIRST thing it would fail on. Found by the LD-21 probe; recorded
            # in iter3-moe-gen3-probe.txt section 2.5.
            output_dtype=_nki_output_dtype(hidden_states.dtype),
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
        key_prefix: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Checkpoint namespace override. ``None`` = the main stack's
        # ``layers.{layer_idx}``; the DSpark drafter passes ``"mtp.{stage}"``
        # while still reporting the out-of-range ``layer_idx`` that makes
        # ``is_hash_moe_layer`` False. See ``weight_loaders._layer_key_prefix``.
        self.key_prefix = key_prefix

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
            key_prefix=self.key_prefix,
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
        2. ``:576`` ``scores = softplus(logits).sqrt()`` -- the FUNCTION. Its
           ARITHMETIC REALIZATION here calls neither ``F.softplus`` nor
           ``torch.sqrt``; see the LD-59 / LD-60 block at step 2 below for the
           polynomial and the ``rsqrt`` form, and why.
        3. ``:577`` the UNBIASED scores are kept aside.
        4. ``:579-580`` ``gate_bias`` is added to a separate tensor.
        5. ``:581-584`` selection: hash gather, or top-k of the BIASED score.
        6. ``:585`` the weights are gathered from the UNBIASED scores.
        7. ``:586-587`` renormalize so the K weights sum to 1.
        8. ``:588`` multiply by ``routed_scaling_factor``.

        Every op is static-shape: ``matmul``, the step-2 elementwise chain
        (``abs``, ``exp``, ``div``, ``mul``, ``add``, ``maximum``, ``clamp_min``,
        ``rsqrt``), ``index_select``, ``topk``, ``gather``, ``scatter``, and
        the FP-69 sanitization chain between steps 5 and 6 (``ge``, ``lt``,
        ``where``, ``clamp``, ``full_like``). No ``.item()``, no
        ``nonzero()``, no boolean-mask indexing, and no Python-level branch
        on a tensor value (``self.is_hash_layer`` is a Python bool fixed at
        construction).
        """
        # Step 1 (:570) — fp32 router matmul. The weight is stored
        # ``[E, H]``, so the contraction transposes it.
        logits = torch.matmul(
            hidden_states.to(torch.float32), self.gate_weight.to(torch.float32).t()
        )

        # Step 2 (:576) — sqrtsoftplus. LD-59 + LD-60 changed the ARITHMETIC
        # REALIZATION of this line, not the function it computes. ``scoring_func``
        # is still ``"sqrtsoftplus"`` (``config.py:124``), the ``raise`` in
        # :DeepseekV4MoE.__init__ that rejects every other value still stands, and
        # the reference line this transcribes is still
        # ``dsv4_ref/model.py:576``. What changed is that neither ``Softplus``
        # nor ``Sqrt`` nor ``Ln`` is asked of the [Act] engine any more: the
        # compiler admits at most 8 DISTINCT [Act] activation functions per core
        # module (``NCC_INLA001`` at ``lower_act.cpp:348``), three failing cores
        # carried 11, and the demonstrated-passing 8-set is the 11 minus exactly
        # ``{Ln, Softplus, Sqrt}``. See port-plan.md §3.3-3.4.
        #
        # PARITY, STATED HERE BECAUSE IT IS NOT NEUTRAL. This is the MoE gate, and
        # step 4 below adds ``gate_bias`` AFTER this nonlinearity, so selection depends
        # on the VALUE and no monotone-invariance escape exists -- a perturbation
        # here can change which expert is selected, not only its weight. The
        # perturbation is bounded at the fp32 rounding floor and is gated
        # off-hardware by G1-G5 (port-plan.md §3.5); the verdict is
        # ``judge_parity``'s, by measurement, against the recorded standard.
        #
        # LD-60 -- softplus WITHOUT a logarithm. Every CLOSED-FORM identity for
        # softplus routes through a log (``log1p(exp x)``, ``x + log1p(exp -x)``,
        # ``-log sigmoid(-x)``); a POLYNOMIAL does not. Range-reduce with the libm
        # identity ``log1p(u) == 2*atanh(u/(2+u))``, then evaluate ``atanh(t)/t``
        # as its own series in ``t**2``:
        #
        #     softplus(x) == max(x,0) + log1p(exp(-|x|))   branch-free and stable
        #     u = exp(-|x|) in (0,1]  =>  t = u/(2+u) in (0,1/3]  =>  t**2 <= 1/9
        #
        # The range reduction is NOT optional: evaluating ``log1p(exp x)``
        # directly overflows ``exp`` for large positive ``x``, and this form never
        # evaluates ``exp`` on a positive argument at all. The ONLY [Act] function
        # it uses is ``Exp``, which every core already carries, so it costs no new
        # table; ``abs`` and ``maximum`` are measured Act-free (``func=Abs`` is 0
        # in all four histograms, and ``max`` is in 0 of 21 activation rosters, so
        # it cannot be an Act function at all).
        a = torch.abs(logits)
        u = torch.exp(-a)
        t = u / (2.0 + u)
        t2 = t * t
        p = _ATANH_C0 + t2 * (
            _ATANH_C1 + t2 * (
                _ATANH_C2 + t2 * (
                    _ATANH_C3 + t2 * (
                        _ATANH_C4 + t2 * (_ATANH_C5 + t2 * _ATANH_C6)
                    )
                )
            )
        )
        y = torch.maximum(logits, torch.zeros_like(logits)) + 2.0 * t * p

        # LD-59 -- sqrt WITHOUT ``Sqrt``, as ``y * rsqrt(y)``. ``torch.rsqrt`` as
        # THIS build lowers it consumes no [Act] table, and that is measured
        # rather than assumed: ``func=Rsqrt`` is 0 in all four BIR histograms,
        # pass-matched by the fact that the same ``DeepseekV4RMSNorm.forward``
        # body which calls ``rsqrt`` on every layer of both graphs has its OWN
        # ``.pow(2)`` counted as ``Square`` in those histograms. The roster does
        # contain ``reciprocal_sqrt``, so the claim is scoped to this lowering,
        # and ``func=Rsqrt > 0`` in any post-fix census falsifies it (F-b).
        #
        # BOTH guards below are load-bearing and NEITHER is optional:
        #  * ``clamp_min`` -- ``softplus(x) > 0`` holds in R but NOT in fp32,
        #    where ``y`` underflows to EXACTLY ``0.0`` for ``x`` below about
        #    -104. Then ``y * rsqrt(y)`` is ``0 * inf`` = NaN, and a NaN in
        #    ``scores`` propagates through step 4's ``+ gate_bias`` into ``topk``
        #    and corrupts EXPERT SELECTION silently. The guard is free at zero
        #    because the outer ``y *`` restores the exact ``0.0``.
        #  * ONE Newton step -- ``sqrt(y) == y * rsqrt(y)`` is NOT exact.
        #    ``sqrt`` is exactly-rounded under IEEE-754; ``rsqrt`` is not an
        #    IEEE-required operation and its accuracy on this build is
        #    UNMEASURABLE off hardware. One iteration takes relative error ``e``
        #    to ``1.5*e**2``, so ``1e-04 -> 1.5e-08`` and ``1e-07 -> 1.5e-14``,
        #    i.e. below fp32 resolution for any plausible hardware rsqrt. Four
        #    ops buy the removal of an unverifiable premise; take them.
        #  * The ``(y * r)`` parenthesisation is deliberate, not cosmetic: it
        #    keeps the intermediate at approximately ``sqrt(y)``, so no
        #    subexpression can overflow for any admitted ``y``. See the
        #    ``_SQRT_GUARD`` bound above, which holds under the reassociated
        #    reading too.
        r = torch.rsqrt(torch.clamp_min(y, _SQRT_GUARD))
        r = r * (1.5 - 0.5 * (y * r) * r)
        scores = y * r

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
            # LD-09 / port plan §3.9. Routed through ``NF.topk`` rather than
            # ``torch.topk`` so the rotational NKI top-k kernel can take this
            # call: sweep finding F-2 predicts a bare ``torch.topk`` lowers to
            # an HLO ``sort`` that ``neuronx-cc`` rejects (``NCC_EVRF029``,
            # exit 70). This is the family's ONLY ``topk``/``sort`` call site
            # and it is live on both prefill and decode.
            #
            # ``process_group=None`` is deliberate and semantics-preserving:
            # the router matmul above is replicated, not expert-sharded, so
            # ``scores`` is already the global ``[T, n_routed_experts]``. With
            # no group the op takes its ``tp_degree == 1`` fast path, which is
            # exactly the local selection ``torch.topk`` was doing. On that
            # path ``gather_dim`` is unread; it is passed as ``-1`` to match
            # ``dim`` rather than left to a default, because the op has none.
            #
            # NOT a guarantee on its own: when ``_can_use_nki_topk`` declines
            # the shape, the op falls back to ``torch.topk`` and F-2's rejected
            # lowering returns. ``_rotational_topk_config_compiles`` is the
            # authoritative device-free gate and it is graded per compiled
            # bucket, off hardware.
            #
            # R-39, parity-relevant: the NKI path and ``torch.topk`` may break
            # ties differently. This is EXPERT selection, so a different
            # tie-break selects a different expert and can change the output
            # token relative to a GPU baseline that used ``torch.topk``
            # semantics -- a divergence that is not a port bug.
            expert_index = NF.topk(
                scores, self.num_experts_per_tok, dim=-1, gather_dim=-1
            )[1]
        expert_index = expert_index.to(torch.int64)

        # --- FP-69 ROUTER-BOUNDARY SENTINEL MASK (LD-73 idiom, re-sited) --
        # ITER-69 / H-ROUTER-RAW (confirmed): the rotational-topk NKI kernel
        # emits uint32 indices that can carry the aws-neuron-sdk#1335
        # max-int sentinel class (0xFFFFFFFF), and defect (A) (F-229) can
        # surface them HERE, upstream of the LD-73 kernel-boundary mask:
        # this raw index stream feeds torch ``gather`` (step 6) and
        # ``scatter`` (the densify below) directly, in every topk layer —
        # the serve-27 residual OOB consumer class. The 3 hash layers key
        # the parameter-backed ``tid2eid`` table and are in-domain by table
        # construction (``valid = None`` skips the mask there); the DSA
        # indexer topk class sanitizes in-graph already and is untouched.
        # Same clamp-plus-collapse idiom as LD-73 (precedent: gpt_oss
        # ``model_bf16.py:1535-1554``), same honesty bound: this masks
        # defect (A)'s EXPRESSION only (F-224) — no health or correctness
        # claim; parity v3r1 remains the untouched arbiter. Invalid slots
        # contribute ZERO routing weight (the sanitized index 0 gathers a
        # score whose weight is then zero-collapsed at step 6, BEFORE step
        # 7's renormalization), and their densify writes are directed to a
        # sink column ``E`` that the trim discards, so no in-domain expert
        # slot can be read through, or clobbered by, an out-of-domain index.
        if self.is_hash_layer:
            valid = None
        else:
            raw_index = expert_index
            valid = (raw_index >= 0) & (raw_index < self.n_routed_experts)
            expert_index = torch.clamp(
                torch.where(valid, raw_index, torch.zeros_like(raw_index)),
                0,
                self.n_routed_experts - 1,
            )
        # --- end FP-69 ROUTER-BOUNDARY SENTINEL MASK -----------------------

        # Step 6 (:585) — weights come from the UNCORRECTED scores. FP-69:
        # the gather is keyed only by in-domain indices, and invalid slots
        # are zero-collapsed before renormalization so they carry no mass.
        weights = original_scores.gather(1, expert_index)
        if valid is not None:
            weights = torch.where(valid, weights, torch.zeros_like(weights))

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
        #
        # FP-69, topk layers only: the scatter target is widened to ``E+1``
        # columns and invalid slots are directed to the sink column ``E``,
        # which the ``[:, :E]`` trim discards — the LD-73 sink-collapse
        # idiom in column form. A fully-invalid row renormalizes 0/0 to NaN
        # at step 7, but every one of its writes lands in the trimmed sink
        # column, so the in-domain ``[T, E]`` result holds its zero init and
        # no NaN escapes. The hash branch keeps the original ``[T, E]``
        # scatter byte-for-byte.
        if valid is None:
            dense = torch.zeros(
                hidden_states.shape[0],
                self.n_routed_experts,
                dtype=torch.float32,
                device=hidden_states.device,
            ).scatter(1, expert_index, weights.to(torch.float32))
        else:
            dense = torch.zeros(
                hidden_states.shape[0],
                self.n_routed_experts + 1,
                dtype=torch.float32,
                device=hidden_states.device,
            ).scatter(
                1,
                torch.where(
                    valid,
                    expert_index,
                    torch.full_like(expert_index, self.n_routed_experts),
                ),
                weights.to(torch.float32),
            )[:, : self.n_routed_experts]

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
