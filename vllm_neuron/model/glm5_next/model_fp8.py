# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (``glm5_next``) blockwise-FP8 modeling skeleton.

``inc-glm53f-013`` -- WP1: model skeleton and ``KVSpec``. This module is the
file D14 declares a **coordinated merge point, rank-1-equivalent for the
duration of the campaign**: this increment creates the module header, the
imports, the ``Glm5NextForConditionalGeneration`` tree, ``get_kv_spec``, and
**forward stubs raising** :class:`NotImplementedError`, so that every later
increment's declared scope is a name that already exists. Each section below
names its D14 owner increment. **Creating a name is not implementing it.**

WHAT THIS INCREMENT CLAIMS, AND NOTHING MORE
--------------------------------------------
The KV geometry: ``get_kv_spec()`` returns one :class:`LayerSpec` per layer of
the hybrid 45-layer stack, 11 carrying the MLA latent geometry and 34 carrying
the KDA geometry. Every compute site here is a stub. That is deliberate and is
why this increment is **NON-KERNEL-CLASS** (P13): a stub computes nothing, so
there is no kernel-class functionality to place, and each compute site carries
its own substrate declaration where it lands.

NO PARAMETER IS ALLOCATED, AND THAT IS LOAD-BEARING
---------------------------------------------------
The stack this skeleton describes is 45 layers of 4096 hidden with 288 routed
experts. Allocating it is not a CPU-mode unit test. So every parameter is
**declared** rather than materialised: :func:`_declare_parameters` calls
``register_parameter(name, None)``, which is torch's own way to reserve a
parameter attribute path that a later increment fills in. The attribute path
genuinely exists on the module (``module.out_proj_weight`` returns ``None``),
the name is enumerable through :meth:`declared_parameter_names`, and the tree
allocates **zero** ``torch.nn.Parameter`` objects. ``get_kv_spec`` therefore
reads geometry off layer objects -- the same construction shape as
``llama3/model.py:1781`` and ``synthetic/synthetic.py:98`` -- without the
allocation that shape would otherwise imply.

WHERE THE PARAMETER NAMES COME FROM -- THEY ARE NOT CHOSEN HERE
---------------------------------------------------------------
Per the lead ruling *"-013's skeleton parameter names: the LANDED weight map's
param-name side is the authority"*
(``approvals/lead-ruling-013-param-name-authority.md``), every parameter
attribute path below is **derived from**
:func:`~vllm_neuron.model.glm5_next.weight_loaders_fp8.build_weight_mappings`
as landed by ``inc-glm53f-011`` / ``inc-glm53f-012``, measured off the bytes at
HEAD -- never invented and never "improved". The derivation is asserted
mechanically in ``test/vllm_neuron/model/glm5_next/test_kv_spec.py``, so the
map-versus-skeleton equality is checkable by reading code rather than by
loading a real checkpoint on hardware. The flat ``<leaf>_weight`` shape (a
parameter on the parent module, not a ``.weight`` on a child ``Linear``) is the
map's own convention, which follows the fork's MoE precedent
``gpt_oss/model_mxfp4.py:2336-2358``.

WHAT IS DELIBERATELY ABSENT
---------------------------
The quant-method dispatcher is ``inc-glm53f-023``'s (D14, M2) and is **not**
here: ``quantization.py:33-39`` states that sequencing fact, not an assignment
to this increment. Sharded scale loading is later work
(``weight_loaders_fp8.py:1086-1093`` states the same kind of fact).
``vllm_neuron/model/kv_cache.py`` is **untouched**: widening ``LayerSpec`` with
KDA recurrent-state fields is ``inc-glm53f-015``'s declared surface at M1, and
its acceptance asserts that the pin's 6-field construction still works with
zero signature breaks. This skeleton builds pin-shaped ``LayerSpec`` values on
the six fields that exist.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

# ---------------------------------------------------------------------------
# Parameter declaration -- the mechanism that makes a name exist without
# allocating the tensor behind it.
# ---------------------------------------------------------------------------


def _declare_parameters(module: nn.Module, *names: str) -> None:
    """Reserve parameter attribute paths on ``module`` without allocating.

    ``register_parameter(name, None)`` is torch's own declaration form: the
    name enters the module's parameter registry, ``getattr`` resolves it to
    ``None``, and no storage is created. ``named_parameters()`` and
    ``state_dict()`` both skip ``None`` entries by design, so the declared set
    is also recorded on ``declared_param_names`` -- that tuple, not
    ``named_parameters()``, is what :meth:`declared_parameter_names` walks.
    """
    for name in names:
        module.register_parameter(name, None)
    module.declared_param_names = (
        *getattr(module, "declared_param_names", ()),
        *names,
    )


# ---------------------------------------------------------------------------
# Geometry resolution for ``get_kv_spec``
# ---------------------------------------------------------------------------


def _resolve_world_size() -> int:
    """The rank count to divide head counts by, or 1 when not distributed.

    Same shape as ``synthetic/synthetic.py:99-103``, which is this fork's only
    config-derived (rather than module-derived) KV-spec precedent.
    ``NeuronConfig`` carries no tensor-parallel degree field, so the process
    group is the only available source and its absence is not an error.
    """
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def _per_rank(count: int, world_size: int) -> int:
    """Per-rank head count, floored at 1 (``synthetic.py:105-107``)."""
    return max(1, count // max(world_size, 1))


def _linear_attn_field(text_config: Glm5NextTextConfig, key: str) -> int:
    """Read one required ``linear_attn_config`` entry, loudly.

    A missing key is raised rather than defaulted: silently substituting a
    head count would make the KDA half of the KV spec wrong in a way no shape
    assertion downstream could attribute back to here.
    """
    config = text_config.linear_attn_config or {}
    if key not in config:
        raise ValueError(
            f"text_config.linear_attn_config is missing {key!r}; the KDA cache "
            f"geometry cannot be resolved (present keys: {sorted(config)})"
        )
    return int(config[key])


def _resolve_model_dtype(text_config: Glm5NextTextConfig) -> torch.dtype:
    """The stack's compute dtype, which is also the MLA latent cache dtype.

    The MLA latent is **not** a quantised cache: ``quantization.py:69-74``
    records that this checkpoint's ``quantization_config`` declares no
    ``kv_cache_quant_algo`` at all and that the blockwise weight scheme is
    rejected for KV caches. So the cache dtype is the model's own dtype.
    """
    dtype = text_config.torch_dtype
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"text_config.torch_dtype must be a torch.dtype, got {dtype!r}; "
            "the KV cache dtype cannot be resolved"
        )
    return dtype


def _resolve_kda_state_dtype(text_config: Glm5NextTextConfig) -> torch.dtype:
    """dtype for the KDA recurrent-state buffers.

    ``NeuronConfig.kda_state_dtype`` (``neuron_config.py:181-184``) is declared
    as *"a torch dtype NAME (e.g. "bfloat16") so it survives a JSON
    additional_config round-trip. None = follow the model's own dtype."* This
    resolver implements that declared rule, which is why it has a real
    ``None`` branch: an unresolved override falls back to the model dtype and
    the result is **never** ``None``. An override that does not name a torch
    dtype is raised rather than passed through as a string, because a ``str``
    reaching ``LayerSpec.dtype`` would be a silently wrong cache allocation.
    """
    neuron_config = text_config.neuron_config
    override = getattr(neuron_config, "kda_state_dtype", None)
    if override is None:
        return _resolve_model_dtype(text_config)
    if isinstance(override, torch.dtype):
        return override
    resolved = getattr(torch, str(override), None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(
            f"NeuronConfig.kda_state_dtype={override!r} does not name a torch "
            "dtype; it is declared as a dtype NAME such as 'bfloat16'"
        )
    return resolved


def _resolve_mla_head_size(text_config: Glm5NextTextConfig) -> int:
    """MLA latent cache width: ``kv_lora_rank + qk_rope_head_dim``.

    On this checkpoint ``qk_rope_head_dim == 0`` because ``mla_use_nope`` is
    set (``config.py:123-127``), so the sum **is** ``kv_lora_rank == 512``.
    The sum is used rather than the bare rank because the cached latent is the
    compressed KV vector concatenated with whatever rotary slice exists, and
    on a config that had one the bare rank would be short by that slice.
    """
    return int(text_config.kv_lora_rank) + int(text_config.qk_rope_head_dim)


# ---------------------------------------------------------------------------
# ``Glm5NextQuantConfig`` / quant-method selection -- D14 owner:
# ``inc-glm53f-023`` (M2, Lane B 1st). LANDED HERE.
#
# THE DISPATCHER GAP THIS CLOSES. ``quantization.py`` already parsed the
# checkpoint's ``quantization_config`` into a ``QuantizationSpec`` carrying
# ``weight_block_size (128, 128)``. What did not exist was anything that turned
# that spec into a METHOD -- so a spec declaring blockwise FP8 resolved to
# nothing, and a call site had no way to ask "what do I run for this module?".
# Measured at the unmodified parent ``6affd98``: the spec reported
# ``(128, 128)`` while ``Glm5NextQuantConfig()`` raised, and a well-formed but
# unsupported shape such as ``[64, 64]`` was ACCEPTED silently.
#
# WHAT THIS SECTION DOES NOT DO, deliberately:
#   * it does NOT attach itself to ``Glm5NextForConditionalGeneration`` -- that
#     tree is ``-013``'s / ``-054``'s D14 section;
#   * it does NOT reach a kernel; the MoE block-quant call site is ``-027``'s;
#   * it does NOT name or import the vendor quantisation enum. There is no
#     blockwise member in it at this pin and adding one is forbidden to this
#     campaign (plan section 11, constraint B.6), so the route is D5(b)'s direct
#     inner-kernel call and this class carries a plugin-side method object.
#
# Imports are FUNCTION-LOCAL rather than added to the module import block, for
# two reasons: the import block is ``-013``'s D14 section and this increment
# does not widen its surface into it; and it is the file family's own idiom --
# ``llama3/quantization.py`` imports its modeling module inside
# ``resolve_attention_mlp_classes`` for the same reason.
# ---------------------------------------------------------------------------


class Glm5NextQuantConfig:
    """Per-module blockwise-FP8 quantisation policy for this arch.

    Holds the parsed :class:`~vllm_neuron.model.glm5_next.quantization.QuantizationSpec`
    and the method resolved from it, and answers the one question a modeling
    call site has: *what do I run for this module?*

    Attributes:
        spec: The parsed spec, or ``None`` for an unquantized checkpoint.
        method: The method resolved for the model-wide scheme, or ``None`` when
            nothing is quantized. Per-module resolution goes through
            :meth:`get_quant_method`, which is the form that survives mixed
            precision; this attribute is the model-wide answer today because
            :meth:`QuantizationSpec.get_scheme` returns one scheme for every
            module.

    Construction raises rather than degrading when the checkpoint declares a
    block shape this build has no authored path for -- see
    :data:`~vllm_neuron.model.glm5_next.quantization.SUPPORTED_WEIGHT_BLOCK_SIZES`.
    """

    def __init__(self, spec: object | None) -> None:
        from vllm_neuron.model.glm5_next.quantization import resolve_quant_method

        self.spec = spec
        self.method = resolve_quant_method(spec)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_model_config(cls, config: Glm5NextConfig) -> Glm5NextQuantConfig:
        """Build from the config the model itself is built from.

        This is the recognition path end to end: the checkpoint's
        ``quantization_config`` was lifted onto :class:`Glm5NextConfig` by
        ``config.py``, is parsed into a spec by
        :meth:`QuantizationSpec.from_model_config`, and is resolved to a method
        here. Nothing in the chain is hand-fed.
        """
        from vllm_neuron.model.glm5_next.quantization import QuantizationSpec

        return cls(QuantizationSpec.from_model_config(config))

    # ------------------------------------------------------------------
    # Per-module query
    # ------------------------------------------------------------------
    def get_quant_method(
        self,
        layer_index: int | None = None,
        prefix: str = "",
    ) -> object | None:
        """Return the method for the module at ``(layer_index, prefix)``.

        Returns ``None`` when that module is not quantized, which means "run the
        unquantized path". The arguments are forwarded to
        :meth:`QuantizationSpec.get_scheme`, so when per-layer dispatch becomes
        real it lands there and every call site here is already passing what it
        needs.
        """
        from vllm_neuron.model.glm5_next.quantization import resolve_quant_method

        return resolve_quant_method(self.spec, layer_index, prefix)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    @property
    def is_block_quantized(self) -> bool:
        """True when a blockwise-FP8 method was resolved."""
        return self.method is not None

    @property
    def block_shape(self) -> tuple[int, int] | None:
        """``(block_h, block_w)`` of the resolved method, or ``None``.

        Delegates rather than storing a copy: the method is the authority for
        the shape, and a second copy is a second thing that can go stale.
        """
        return None if self.method is None else self.method.block_shape


# ---------------------------------------------------------------------------
# ``Glm5NextHyperConnection`` -- mHC wiring. D14 owner: ``inc-glm53f-030``
# (M2, Lane B 6th).
#
# A NAME ONLY, and deliberately NOT instantiated in the tree: the landed
# weight map declares no mHC parameter, so wiring this module in would invent
# a parameter path the map's param-name side does not carry. ``-030`` owns the
# wiring.
# ---------------------------------------------------------------------------


class Glm5NextHyperConnection(nn.Module):
    """Multi-hyper-connection (mHC) residual mixing.

    Section reserved for ``inc-glm53f-030``. ``text_config`` already carries
    the checkpoint's dials (``mhc``, ``hc_mult``, ``hc_sinkhorn_iters``,
    ``hc_eps`` -- ``config.py:139-143``) and ``NeuronConfig`` the overrides
    (``mhc_sinkhorn_iters``, ``mhc_eps``).
    """

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextHyperConnection.forward is a stub created by "
            "inc-glm53f-013; mHC wiring lands with inc-glm53f-030"
        )


# ---------------------------------------------------------------------------
# ``Glm5NextMoEBlock`` and its two expert containers. D14 owners:
# ``inc-glm53f-031`` (expert partitioning), ``-032`` (router call site),
# ``-027`` (block-quant kernel call site), ``-033`` (shared-expert path) --
# all M2 Lane B, serialized because they share one class.
# ---------------------------------------------------------------------------


class Glm5NextRoutedExperts(nn.Module):
    """The routed-expert bank at ``mlp.experts``.

    One parameter per projection covers **all** ``n_routed_experts`` experts:
    the landed map sends ``mlp.experts.<leaf>_weight`` to a *list* of
    ``n_routed_experts`` checkpoint keys
    (``weight_loaders_fp8.py:471-479``), because this checkpoint stores one
    tensor per expert while the fork's parameter side is per-projection. The
    router lives here too, and carries a bias because
    ``topk_method == "noaux_tc"`` (``weight_loaders_fp8.py:458-459``).
    """

    # ── expert partitioning -- D14 owner: ``inc-glm53f-031`` ─────────────
    #
    # WHAT THIS SECTION CLOSES. At the unmodified parent ``031535b`` this bank
    # reported ``num_routed_experts == 288`` and nothing else: it carried no
    # per-rank expert count, no local expert index set, and no shard plan
    # (measured -- ``num_local_experts``, ``local_expert_indices`` and
    # ``expert_partition`` were all absent). So an assertion written against
    # ``num_routed_experts`` would have passed at the parent and certified
    # nothing; the partition below is what this increment actually authors.
    #
    # WHY THE ARITHMETIC IS IMPORTED RATHER THAN WRITTEN HERE. It lives in
    # ``factory.py``, which nothing on the arch-lookup path pulls the modeling
    # module into. The import is FUNCTION-LOCAL, following the landed
    # ``inc-glm53f-023`` precedent in this file and for the same two reasons:
    # this file's module import block is ``-013``'s D14 section, and the
    # file family's own idiom is a local import at the consuming member.
    #
    # THE 288/64 CONSEQUENCE IS DELIBERATE AND VISIBLE. At the registered TP
    # degree freeze of 64, 288 experts are ragged, so building this bank RAISES
    # a named error instead of padding or flooring. That is campaign gap G4
    # surfaced where the model is built, and it is the lead's to dispose.

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        from vllm_neuron.model.glm5_next.factory import (
            require_uniform_expert_partition,
        )

        self.num_routed_experts = int(text_config.n_routed_experts)
        self.num_experts_per_tok = int(text_config.num_experts_per_tok)
        # ``_resolve_world_size()`` is ``-013``'s helper, called rather than
        # edited: an explicit ``world_size`` is what a caller with a degree
        # supplies, and ``None`` means "read the process group", which is 1 when
        # this stack is built undistributed.
        self.tp_degree = (
            _resolve_world_size() if world_size is None else int(world_size)
        )
        self.expert_partition = require_uniform_expert_partition(
            self.num_routed_experts, self.tp_degree
        )
        # Uniform by the gate above, so rank 0's count is every rank's count.
        self.num_local_experts = self.expert_partition.counts[0]
        _declare_parameters(
            self,
            "router_weight",
            "router_bias",
            "gate_proj_weight",
            "up_proj_weight",
            "down_proj_weight",
        )

    def local_expert_indices(self, rank: int) -> tuple[int, ...]:
        """The global expert indices ``rank`` owns, ascending.

        Delegates to the plan rather than recomputing the arithmetic, so there
        is one partition and not two that can disagree.
        """
        return self.expert_partition.local_expert_indices(rank)

    # ── router call site -- D14 owner: ``inc-glm53f-032`` ────────────────
    #
    # SCOPE. This section adds ONE method and edits no line above it.
    # ``__init__`` and ``local_expert_indices`` are ``inc-glm53f-031``'s landed
    # code and ``forward`` is ``inc-glm53f-013``'s stub; all three stay
    # byte-identical, so nothing ``-031``'s recorded acceptance asserts can move.
    #
    # WHY THE CONFIG IS AN ARGUMENT RATHER THAN STATE. The routing
    # hyperparameters live on ``Glm5NextTextConfig``, and ``-031``'s ``__init__``
    # does not retain it. Adding a field would edit that landed ``__init__``, so
    # the config is threaded in at the call instead -- the smaller change, and
    # the one D14's section-ownership rule permits.
    #
    # WHAT THIS SITE DOES NOT DO. It returns GLOBAL expert indices over all
    # ``n_routed_experts``, exactly as the checkpoint's router does. Mapping
    # those onto this rank's ``expert_partition`` slice is the DISPATCH step,
    # and it belongs to the block-quant call site (``-027``) and the
    # shared-expert path (``-033``). Doing it here would put two owners on one
    # behaviour.
    #
    # ``rms_norm_eps`` IS ABSENT, MEASURED NOT ASSUMED. Neither
    # ``Glm5NextTextConfig`` nor the campaign's pinned
    # ``fixtures/config.json`` carries an ``rms_norm_eps``, so the substrate's
    # own default (``rmsnorm_router_topk_tkg``'s ``eps=1e-6``) is the only value
    # available and is surfaced as a parameter rather than buried.

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        gamma: torch.Tensor,
        text_config: Glm5NextTextConfig,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route ``[B, S, H]`` tokens to the top-``k`` experts with ``noaux_tc``.

        Args:
            hidden_states: ``[B, S, H]`` pre-norm decoder activations.
            gamma: ``[H]`` or ``[1, H]`` router RMSNorm weights.
            text_config: the decoder config the routing hyperparameters live on.
            eps: RMSNorm epsilon; see the section note above.

        Returns:
            ``(router_logits [T, E], expert_index [T, k] int32,
            expert_affinities [T, E] float32)``. ``expert_affinities`` is the
            scattered form the downstream MoE consumes: the gate weight at each
            selected expert's column, zero elsewhere.

        The seam is entered with ``correction_bias=self.router_bias``, and that
        parameter is ``mlp.gate.e_score_correction_bias``
        (``weight_loaders_fp8.py:463-467``) -- the ``noaux_tc`` correction bias,
        NOT a router projection bias. The seam's own signature keeps the two
        apart by name, because adding this tensor to the logits instead of to
        the sigmoid scores would compute a different router that no shape check
        could catch.
        """
        # Function-local import, following the landed ``inc-glm53f-023`` and
        # ``-031`` precedent in this file: this file's module import block is
        # ``-013``'s D14 section.
        from vllm_neuron.functional.moe.router import (
            noaux_tc_rmsnorm_router_topk,
        )

        logits, expert_index, expert_affinities, _substrate_index = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=self.router_weight,
                correction_bias=self.router_bias,
                top_k=int(text_config.num_experts_per_tok),
                eps=eps,
                norm_topk_prob=bool(text_config.norm_topk_prob),
                routed_scaling_factor=float(text_config.routed_scaling_factor),
            )
        )
        return logits, expert_index, expert_affinities

    # ── block-quant kernel call site -- D14 owner: ``inc-glm53f-027`` ─────
    #
    # SCOPE. This section adds ONE method and edits no line above or below it.
    # ``__init__`` and ``local_expert_indices`` are ``inc-glm53f-031``'s landed
    # code, ``route_tokens`` is ``inc-glm53f-032``'s, and ``forward`` is
    # ``inc-glm53f-013``'s stub; all four stay byte-identical, so nothing any
    # of those three recorded acceptances asserts can move. The method sits on
    # ``Glm5NextRoutedExperts`` rather than on ``Glm5NextMoEBlock`` under D14's
    # sub-class rule -- the same reading ``-032`` landed for the sibling
    # router-call-site row, and for the same reason: the routed bank is where
    # the expert weights and the partition live.
    #
    # D5(b): THE INNER KERNEL IS CALLED DIRECTLY. The public ``moe_cte``
    # dispatcher will not forward block scales, so this site enters
    # ``inc-glm53f-025``'s ``blockwise_fp8_moe`` seam instead of the dispatcher.
    #
    # NO QUANTISATION ENUM MEMBER IS NAMED OR ADDED (plan section 11 constraint
    # B.6, and the ``-023`` section above already declares the same negative).
    # The route is selected by WHICH FUNCTION IS CALLED, and the not-block-quant
    # branch RAISES BY NAME rather than falling through to the substrate's
    # ``QuantizationType.NONE`` default (``functional/mlp.py:81``, ``:249`` --
    # a default reached by OMISSION, so a call site that forgot to route gets it
    # with no error at all). The acceptance's dispatch-counter clause exists to
    # exclude exactly that silent fallback; the raise makes it IMPOSSIBLE rather
    # than merely detectable.
    #
    # WHY THE WEIGHTS, SCALES AND CONFIG ARE ARGUMENTS. ``-031``'s ``__init__``
    # declares ``gate_proj_weight`` / ``up_proj_weight`` / ``down_proj_weight``
    # by ``register_parameter(name, None)`` and NO scale parameter, and it
    # retains no config. So there is no landed layout to read: the fused
    # ``[E, H, 2, I_TP]`` gate/up tensor the kernel needs is not the
    # checkpoint's per-projection storage, and turning one into the other is the
    # weight loader's step, not this section's. Adding fields to that landed
    # ``__init__`` would edit ``-031``'s code, which D14's section-ownership
    # rule does not permit here, so everything this site consumes is threaded in
    # at the call -- the smaller change, and ``-032``'s own precedent for the
    # config argument. What IS read off ``self`` is the partition
    # (``num_local_experts``, ``num_experts_per_tok``), which is what ``-031``
    # landed this bank to carry.
    #
    # WHAT THIS SITE DOES NOT DO, deliberately: it does not touch the shared
    # expert (``-033``), it authors no numerics of its own (``-025``'s kernel and
    # ``-024``'s retile producer own those), it does not fuse or transpose
    # checkpoint tensors, and it does not assemble the layer forward
    # (``-013`` / ``-054``). It reuses ``build_blockwise_mapping`` (F9) for the
    # token-block mapping rather than re-deriving one.
    #
    # Imports are FUNCTION-LOCAL, following the landed ``-023``, ``-031`` and
    # ``-032`` precedent in this file: the module import block is ``-013``'s D14
    # section. The two ``to_kernel_scale_layout`` helpers in this campaign have
    # THE SAME NAME AND DIFFERENT SIGNATURES
    # (``functional/blockwise_fp8_mm.py:309`` takes ``(weight_scale, rows,
    # cols)``; ``functional/moe/moe_blockwise_fp8.py:170`` takes
    # ``(consumer_scales, num_experts, rows, cols, projection)``), so the one
    # this site needs is imported from its own module UNDER AN ALIAS that names
    # the module -- an unqualified import of both would be an arity failure that
    # reads as a shape bug.

    def block_quant_expert_mm(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        gate_up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_up_consumer_scales: torch.Tensor,
        down_consumer_scales: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
        *,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
    ) -> torch.Tensor:
        """Run this rank's routed experts through the block-quant NKI kernel.

        Args:
            hidden_states: ``[T, H]`` real tokens only -- the kernel's
                padding-token slot is appended here, not by the caller
                (``bwmm_shard_on_I.py:157``).
            expert_affinities: ``[T, E_local]`` scattered router scores, the
                form :meth:`route_tokens` returns: the gate weight at each
                selected expert's column and zero elsewhere.
            gate_up_proj_weight: ``[E_local, H, 2, I_TP]`` fp8-e4m3, retiled
                onto ``BLOCK_QUANT_SIZE`` granularity.
            down_proj_weight: ``[E_local, I_TP, H]`` fp8-e4m3, retiled.
            gate_up_consumer_scales: the retile producer's FLAT emission for the
                fused gate/up bank, shape
                :func:`~vllm_neuron.functional.moe.blockwise_fp8_retile.consumer_scale_shape`
                at ``projection=GATE_UP``. Both fusion halves must be present:
                the producer writes one half per call and leaves the other
                ``NaN``, and merging them is the loader's step.
            down_consumer_scales: the same, at ``projection=DOWN``.
            quant_config: the resolved per-model quantisation policy. This is
                the route selector; see the section note above.
            block_size: tokens per block, a multiple of ``BLOCK_QUANT_SIZE``.
                Defaults to ``BLOCK_QUANT_SIZE`` itself.
            moe_group: the MoE ``GroupCoordinator``, forwarded verbatim to
                ``build_blockwise_mapping``. It is UNREAD on both of that
                function's flows when ``tp_degree == 1``
                (``moe_blockwise.py:354`` and ``:398`` gate every use behind
                ``tp_degree > 1``), which is why an undistributed call site may
                leave it ``None``; a sharded one supplies the real group rather
                than having this site invent one.
            tp_degree: ranks sharding each expert's intermediate dimension.

        Returns:
            ``[T, H]`` -- the padding-token row is sliced off. The dtype is the
            seam's own (``bfloat16`` on the NKI route) and is not re-cast here:
            the layer forward decides the residual dtype and that is
            ``inc-glm53f-054``'s section.

        Raises:
            Glm5NextBlockQuantRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one the
                retile bridges, or when an operand extent contradicts another.
                Named rather than coerced, so a mis-wired call site fails where
                it is wrong instead of computing a different function.
        """
        from vllm_neuron.functional import build_blockwise_mapping
        from vllm_neuron.functional.moe.blockwise_fp8_retile import (
            BLOCK_QUANT_SIZE,
            DOWN,
            GATE_UP,
            TILE_SIZE,
        )
        from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
            blockwise_fp8_moe,
        )
        from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
            to_kernel_scale_layout as moe_to_kernel_scale_layout,
        )

        # ---- ROUTE SELECTION. The certifying component (D1.4). ----------- #
        if not quant_config.is_block_quantized:
            raise Glm5NextBlockQuantRouteError(
                "block_quant_expert_mm is the D5(b) block-quant route and "
                f"quant_config resolved method={quant_config.method!r}. "
                "Refusing to run: there is no unquantised expert path at this "
                "site, and silently continuing is what would reach the "
                "substrate's QuantizationType.NONE default by omission."
            )
        block_shape = quant_config.block_shape
        if block_shape is None or tuple(block_shape) != (TILE_SIZE, TILE_SIZE):
            raise Glm5NextBlockQuantRouteError(
                f"quant_config declares weight_block_size={block_shape!r}; this "
                f"route consumes scales retiled from "
                f"({TILE_SIZE}, {TILE_SIZE}) checkpoint blocks onto "
                f"{BLOCK_QUANT_SIZE} granularity and has no path for any other "
                f"checkpoint block shape."
            )

        # ---- Extents, read off the operands rather than off the config. -- #
        if hidden_states.dim() != 2:
            raise Glm5NextBlockQuantRouteError(
                f"hidden_states must be [T, H], got shape "
                f"{tuple(hidden_states.shape)}"
            )
        tokens, hidden = (int(extent) for extent in hidden_states.shape)
        if gate_up_proj_weight.dim() != 4 or gate_up_proj_weight.shape[2] != 2:
            raise Glm5NextBlockQuantRouteError(
                f"gate_up_proj_weight must be [E, H, 2, I_TP], got shape "
                f"{tuple(gate_up_proj_weight.shape)}"
            )
        num_experts = int(gate_up_proj_weight.shape[0])
        intermediate = int(gate_up_proj_weight.shape[-1])
        if num_experts != int(self.num_local_experts):
            raise Glm5NextBlockQuantRouteError(
                f"gate_up_proj_weight carries {num_experts} experts but this "
                f"rank owns {self.num_local_experts}; the bank and the "
                f"partition must agree"
            )
        if int(gate_up_proj_weight.shape[1]) != hidden:
            raise Glm5NextBlockQuantRouteError(
                f"gate_up_proj_weight has H={int(gate_up_proj_weight.shape[1])} "
                f"but hidden_states has H={hidden}"
            )
        if tuple(down_proj_weight.shape) != (num_experts, intermediate, hidden):
            raise Glm5NextBlockQuantRouteError(
                f"down_proj_weight must be [E={num_experts}, "
                f"I_TP={intermediate}, H={hidden}], got shape "
                f"{tuple(down_proj_weight.shape)}"
            )
        if tuple(expert_affinities.shape) != (tokens, num_experts):
            raise Glm5NextBlockQuantRouteError(
                f"expert_affinities must be [T={tokens}, E={num_experts}], got "
                f"shape {tuple(expert_affinities.shape)}"
            )
        block = BLOCK_QUANT_SIZE if block_size is None else int(block_size)
        if block <= 0 or block % BLOCK_QUANT_SIZE:
            raise Glm5NextBlockQuantRouteError(
                f"block_size={block} is not a positive multiple of "
                f"BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE} "
                f"(bwmm_shard_on_I.py:667)"
            )

        # ---- The padding-token slot. Appended here, once. ---------------- #
        # The kernel reads a ``-1`` token position as the LAST row of
        # ``hidden_states`` (``bwmm_shard_on_I.py:157``), so both the hidden
        # tensor and the affinity tensor grow one zero row and the mapping is
        # built over ``T + 1`` tokens. Slicing it back off is the last step.
        pad_hidden = torch.zeros(
            1, hidden, dtype=hidden_states.dtype, device=hidden_states.device
        )
        padded_hidden = torch.cat([hidden_states, pad_hidden], dim=0)
        pad_affinities = torch.zeros(
            1,
            num_experts,
            dtype=expert_affinities.dtype,
            device=expert_affinities.device,
        )
        padded_affinities = torch.cat([expert_affinities, pad_affinities], dim=0)

        # ---- The token-block mapping. REUSED (F9), not authored. --------- #
        # ``conditions`` is deliberately unconsumed: it feeds the
        # ``*_hybrid`` dynamic-while variant of the vendor kernel, and D5(b)
        # calls the non-hybrid inner member, whose signature has no such
        # parameter. Named with a leading underscore rather than dropped so the
        # unused fourth return is visible instead of implied.
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            _conditions,
        ) = build_blockwise_mapping(
            expert_affinities=padded_affinities,
            num_local_experts=num_experts,
            num_experts_per_token=int(self.num_experts_per_tok),
            block_size=block,
            moe_group=moe_group,
            tp_degree=tp_degree,
        )

        # ---- The scale bridge. ``-025``'s helper, at its own arity. ------ #
        # The mapping is built from the fp32 affinities so the nonzero mask is
        # exact, and the masked tensor is cast to the activation dtype only
        # afterwards -- the kernel multiplies it into the hidden states.
        gate_up_proj_scale = moe_to_kernel_scale_layout(
            gate_up_consumer_scales,
            num_experts,
            hidden,
            intermediate,
            projection=GATE_UP,
        )
        down_proj_scale = moe_to_kernel_scale_layout(
            down_consumer_scales,
            num_experts,
            hidden,
            intermediate,
            projection=DOWN,
        )

        output = blockwise_fp8_moe(
            hidden_states=padded_hidden,
            expert_affinities_masked=expert_affinities_masked.to(
                hidden_states.dtype
            ),
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block,
            token_position_to_id=token_position_to_id,
            # ``[N]`` from the mapping, ``[N, 1]`` at the seam
            # (``moe_blockwise_fp8.py:323``).
            block_to_expert=block_to_expert.reshape(-1, 1),
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
        )
        return output[:tokens]

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextRoutedExperts.forward is a stub created by "
            "inc-glm53f-013; expert partitioning lands with inc-glm53f-031, "
            "the router call site with inc-glm53f-032, and the block-quant "
            "kernel call site with inc-glm53f-027"
        )


# ``inc-glm53f-027``'s named refusal, at module level because an exception a
# caller catches belongs in the module namespace and not nested in the class
# that raises it. It is the second and last hunk of this increment in this file,
# and it is a pure insertion: no line of ``Glm5NextRoutedExperts`` above it or
# ``Glm5NextSharedExperts`` below it moves.
class Glm5NextBlockQuantRouteError(ValueError):
    """A block-quant expert call this route refuses, named rather than coerced.

    Raised in preference to continuing, because the failure this closes is a
    call site that reaches the substrate's ``QuantizationType.NONE`` default by
    OMISSION and computes a different function while every shape check passes.
    """


class Glm5NextSharedExperts(nn.Module):
    """The always-on shared expert at ``mlp.shared_experts``.

    Declared only when ``n_shared_experts`` is nonzero, mirroring the map's
    own condition (``weight_loaders_fp8.py:481``).
    """

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.num_shared_experts = int(text_config.n_shared_experts)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    # ── shared-expert path -- D14 owner: ``inc-glm53f-033`` ───────────────
    #
    # SCOPE. This section adds ONE method and edits no line above or below it.
    # ``__init__`` above is ``inc-glm53f-013``'s landed code and ``forward``
    # below is ``-013``'s stub; both stay byte-identical, so nothing any landed
    # acceptance asserts can move. The residual add lives on
    # ``Glm5NextMoEBlock`` -- this increment's second section -- because adding
    # the routed and shared halves needs both children and neither child owns
    # the other.
    #
    # THE COUNT IS 3, AND IT IS WHY THE STRUCTURE BELOW IS THREE CALLS. The
    # route predicate (plan L934-935, revision 33) reads ``-026``'s dispatch
    # counter as EXACTLY 3 -- one per projection site -- per shared-expert call.
    # Three sequential entries into ``blockwise_fp8_mm`` is therefore the
    # criterion and not an implementation convenience: a reading of 3 proves all
    # three projections crossed the block-quant seam, so no projection can
    # silently reach the substrate's non-blockwise MLP path and ignore the block
    # scales, and a bypassed projection reads 2 and fails. ``-033`` attempt 1
    # measured why the earlier declared ``1`` was structurally unreachable
    # (``increments/evidence-033.md``): ``silu`` is non-linear, so ``down``
    # cannot fold into either predecessor, and ``blockwise_fp8_mm`` takes a 2-D
    # weight and adds exactly ``+1`` per invocation at a single unlooped site.
    #
    # NO FUSION IS AUTHORED (plan L938 and the Substrate bullet's own words).
    # ``-013`` landed ``gate_proj_weight``, ``up_proj_weight`` and
    # ``down_proj_weight`` as THREE UNFUSED parameters, matching the weight map
    # (``weight_loaders_fp8.py:481-489``) and the fork's own dense precedent
    # (``Glm5NextDenseMLP`` below: "Gate and up stay **separate** parameters").
    # Concatenating gate and up into one ``[H, 2I]`` operand would author a
    # scale-grid concatenation this increment may not author, and it would read
    # 2 rather than the declared 3.
    #
    # WHY THE WEIGHTS, SCALES AND CONFIG ARE ARGUMENTS -- ``-027``'s measured
    # shape contract (``increments/evidence-027.md`` §2.3), inherited rather
    # than re-litigated. ``-013``'s ``__init__`` declares the three projections
    # by ``register_parameter(name, None)`` and NO scale parameter, and retains
    # no config. Producing the block scales is the weight loader's step, not
    # this section's, so everything this site consumes is threaded in at the
    # call -- ``-032``'s and ``-027``'s precedent for the config argument.
    #
    # THE SCALE OPERAND IS THE PUBLIC GRID, SO NEITHER ``to_kernel_scale_layout``
    # IS IMPORTED HERE. The campaign carries two helpers of that name at
    # different arities (``functional/blockwise_fp8_mm.py:309`` takes
    # ``(weight_scale, rows, cols)``; ``functional/moe/moe_blockwise_fp8.py:170``
    # takes ``(consumer_scales, num_experts, rows, cols, projection)``).
    # ``blockwise_fp8_mm`` applies the dense one ITSELF at ``:436``, so this site
    # passes the public ``[K//256, N//256]`` grid and imports neither -- the
    # disambiguation hazard is removed rather than navigated.
    #
    # Imports are FUNCTION-LOCAL, following the landed ``-023``, ``-031``,
    # ``-032`` and ``-027`` precedent in this file: the module import block is
    # ``-013``'s D14 section.

    def shared_expert_mm(
        self,
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
    ) -> torch.Tensor:
        """Run the always-on shared expert through ``-026``'s dense block GEMM.

        The SwiGLU the checkpoint stores: ``down(silu(gate(x)) * up(x))``, with
        each of the three projections a separate entry into
        :func:`~vllm_neuron.functional.blockwise_fp8_mm.blockwise_fp8_mm`.

        Args:
            hidden_states: ``[T, H]`` activations, ``bfloat16``. ``T`` must be a
                whole number of ``TILE_SIZE`` rows -- the seam tiles ``M`` over
                the PSUM partition axis and does not pad
                (``blockwise_fp8_mm.py:239-245``), so padding is the caller's.
            gate_proj_weight: ``[H, I]`` fp8-e4m3, expressed against
                ``gate_proj_scale``.
            up_proj_weight: ``[H, I]`` fp8-e4m3.
            down_proj_weight: ``[I, H]`` fp8-e4m3.
            gate_proj_scale: ``[H//256, I//256]`` fp32, the PUBLIC block-scale
                grid :func:`~vllm_neuron.functional.blockwise_fp8_mm.scale_grid_shape`
                declares -- one scale per ``256 x 256`` weight block.
            up_proj_scale: the same, for ``up_proj_weight``.
            down_proj_scale: ``[I//256, H//256]`` fp32, for ``down_proj_weight``.
            quant_config: the resolved per-model quantisation policy, and the
                route selector; see the section note above.

        Returns:
            ``[T, H]`` **fp32** -- the seam's own return dtype, not re-cast here.
            The layer forward decides the residual dtype and that is
            ``inc-glm53f-054``'s section, exactly as ``-027`` left it.

        Raises:
            Glm5NextSharedExpertRouteError: when ``quant_config`` resolved no
                block-quant method, when its block shape is not the one the
                retile bridges, or when two operand extents contradict each
                other. Named rather than coerced, so a mis-wired call site fails
                where it is wrong instead of computing a different function.
        """
        from torch.nn.functional import silu

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm
        from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE

        # ---- ROUTE SELECTION. The certifying component (D1.4). ----------- #
        # Same shape as ``-027``'s: the route is selected by WHICH FUNCTION IS
        # CALLED and the not-block-quant branch RAISES BY NAME, so it cannot
        # fall through to the substrate's ``QuantizationType.NONE`` default
        # (``functional/mlp.py:81``, ``:249`` -- a default reached by OMISSION).
        # No quantisation enum member is named or added (plan section 11
        # constraint B.6). The counter clause DETECTS that silent fallback; this
        # raise makes it impossible.
        if not quant_config.is_block_quantized:
            raise Glm5NextSharedExpertRouteError(
                "shared_expert_mm is the block-quant shared-expert route and "
                f"quant_config resolved method={quant_config.method!r}. "
                "Refusing to run: there is no unquantised shared-expert path at "
                "this site, and silently continuing is what would reach the "
                "substrate's QuantizationType.NONE default by omission."
            )
        block_shape = quant_config.block_shape
        if block_shape is None or tuple(block_shape) != (TILE_SIZE, TILE_SIZE):
            raise Glm5NextSharedExpertRouteError(
                f"quant_config declares weight_block_size={block_shape!r}; this "
                f"route consumes scales retiled from ({TILE_SIZE}, {TILE_SIZE}) "
                f"checkpoint blocks onto BLOCK_QUANT_SIZE granularity and has no "
                f"path for any other checkpoint block shape."
            )

        # ---- Extents, read off the operands rather than off the config. -- #
        # Only the cross-operand agreements the seam cannot see are checked
        # here. ``blockwise_fp8_mm`` already refuses a K mismatch (``:420``), a
        # mis-sized or non-fp32 scale grid (``:328``, ``:335``) and every
        # blocking condition (``_require_blocked``), and repeating those would
        # create a second authority that can drift from the first.
        if hidden_states.dim() != 2:
            raise Glm5NextSharedExpertRouteError(
                f"hidden_states must be [T, H], got shape "
                f"{tuple(hidden_states.shape)}"
            )
        hidden = int(hidden_states.shape[1])
        if tuple(gate_proj_weight.shape) != tuple(up_proj_weight.shape):
            raise Glm5NextSharedExpertRouteError(
                f"gate_proj_weight {tuple(gate_proj_weight.shape)} and "
                f"up_proj_weight {tuple(up_proj_weight.shape)} must have the "
                f"same [H, I] extents; they are multiplied elementwise after "
                f"the activation"
            )
        if gate_proj_weight.dim() != 2 or int(gate_proj_weight.shape[0]) != hidden:
            raise Glm5NextSharedExpertRouteError(
                f"gate_proj_weight must be [H={hidden}, I], got shape "
                f"{tuple(gate_proj_weight.shape)}"
            )
        intermediate = int(gate_proj_weight.shape[1])
        if tuple(down_proj_weight.shape) != (intermediate, hidden):
            raise Glm5NextSharedExpertRouteError(
                f"down_proj_weight must be [I={intermediate}, H={hidden}], got "
                f"shape {tuple(down_proj_weight.shape)}"
            )

        # ---- The three projection sites. The counted seam entries. ------- #
        # ENTRY 1 of 3 -- gate.
        gate = blockwise_fp8_mm(hidden_states, gate_proj_weight, gate_proj_scale)
        # ENTRY 2 of 3 -- up.
        up = blockwise_fp8_mm(hidden_states, up_proj_weight, up_proj_scale)

        # The SwiGLU wiring. This is call-site plumbing, not authored numerics:
        # ``silu`` is torch's own, the product is elementwise, and both run in
        # the seam's fp32 return dtype so no precision is thrown away between
        # the projections. This is also the exact reason the declared count
        # cannot be 1 -- ``silu`` is non-linear, so ``down`` cannot fold into
        # either predecessor and the product must be materialised here.
        activated = silu(gate) * up

        # The down projection re-enters the seam, whose declared input dtype is
        # ``bfloat16`` (``blockwise_fp8_mm.py:408``), so the fp32 intermediate is
        # cast back to the activation dtype. The cast is named rather than
        # implicit because it is a real precision step and the acceptance's torch
        # reference mirrors it at the same point.
        # ENTRY 3 of 3 -- down.
        return blockwise_fp8_mm(
            activated.to(hidden_states.dtype), down_proj_weight, down_proj_scale
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextSharedExperts.forward is a stub created by "
            "inc-glm53f-013; the shared-expert path lands with inc-glm53f-033"
        )


# ``inc-glm53f-033``'s named refusal, at module level for the same reason
# ``-027``'s is: an exception a caller catches belongs in the module namespace
# and not nested in the class that raises it. It is a pure insertion between two
# classes -- no line of ``Glm5NextSharedExperts`` above it or
# ``Glm5NextMoEBlock`` below it moves.
class Glm5NextSharedExpertRouteError(ValueError):
    """A shared-expert call this route refuses, named rather than coerced.

    Raised in preference to continuing, because the failure this closes is a
    call site that reaches the substrate's ``QuantizationType.NONE`` default by
    OMISSION and computes a different function while every shape check passes.
    """


class Glm5NextMoEBlock(nn.Module):
    """The sparse MLP at ``mlp`` on layers at or past ``first_k_dense_replace``.

    Holds no parameter of its own: the map places every sparse-MLP parameter
    under ``mlp.experts`` or ``mlp.shared_experts``.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        # ``world_size`` is a trailing optional addition by ``inc-glm53f-031``,
        # threaded to the routed bank only. ``_build_mlp``'s call site is
        # unchanged and stays outside this increment's surface.
        self.experts = Glm5NextRoutedExperts(text_config, world_size=world_size)
        if text_config.n_shared_experts:
            self.shared_experts = Glm5NextSharedExperts(text_config)

    # ── the residual add -- D14 owner: ``inc-glm53f-033`` ─────────────────
    #
    # SCOPE. This section adds ONE method and edits no line above or below it.
    # ``__init__`` above and ``forward`` below are ``inc-glm53f-013``'s landed
    # code and stay byte-identical.
    #
    # WHY THE ADD LIVES HERE AND NOT ON EITHER CHILD. It needs the routed half
    # and the shared half, and neither ``Glm5NextRoutedExperts`` nor
    # ``Glm5NextSharedExperts`` owns the other. ``Glm5NextMoEBlock`` holds both
    # (``__init__`` above), so it is the one place the two halves meet. D14's
    # sub-class rule sends the compute to the child that owns the weights --
    # which is why ``-027`` put the routed matmul on the routed bank and this
    # increment put the shared matmul on the shared bank -- and it sends the
    # combination to their parent.
    #
    # "ADDED EXACTLY ONCE" IS STRUCTURAL HERE, NOT ASSERTED. The plan's second
    # acceptance conjunct (L933) proves the shared contribution enters the layer
    # output once rather than twice. That property is made true by construction
    # below: this method contains EXACTLY ONE call to ``shared_expert_mm`` and
    # EXACTLY ONE ``+``, and it is the only place in this file that adds a shared
    # contribution to a routed one. The acceptance measures the property
    # numerically; this structure is what makes the measurement reproducible
    # rather than incidental.
    #
    # WHAT THIS SECTION DOES NOT DO, deliberately: it does not assemble the layer
    # forward (``-013`` / ``-054``), it does not call the routed path (``-027``'s
    # ``block_quant_expert_mm``, landed and separately accepted), and it authors
    # no numerics. ``routed_output`` arrives as an argument precisely so this
    # method composes the two halves without owning either.

    def combine_routed_and_shared(
        self,
        routed_output: torch.Tensor,
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
        quant_config: Glm5NextQuantConfig,
    ) -> torch.Tensor:
        """``routed_output + shared_expert(hidden_states)`` -- the layer output.

        Args:
            routed_output: ``[T, H]`` the routed experts' contribution, as
                ``inc-glm53f-027``'s ``block_quant_expert_mm`` returns it. Passed
                in rather than computed here; see the section note above.
            hidden_states: ``[T, H]`` the shared expert's own input -- the same
                pre-norm activations the routed half consumed.
            gate_proj_weight: forwarded verbatim to
                :meth:`Glm5NextSharedExperts.shared_expert_mm`, which documents
                every operand's layout and is the single authority for it.
            up_proj_weight: as above.
            down_proj_weight: as above.
            gate_proj_scale: as above.
            up_proj_scale: as above.
            down_proj_scale: as above.
            quant_config: as above.

        Returns:
            ``[T, H]`` fp32 -- the sum, in the shared half's fp32 seam dtype.
            The residual dtype for the decoder layer is ``inc-glm53f-054``'s.

        Raises:
            Glm5NextSharedExpertRouteError: when this block declares no shared
                expert, or when ``routed_output`` and the shared contribution
                disagree in extent. A silent broadcast is the failure this
                refuses: ``[T, H] + [1, H]`` and ``[T, H] + [T, 1]`` both
                broadcast without error and both compute a different layer.
        """
        # ``n_shared_experts == 0`` leaves the attribute undeclared -- ``-013``'s
        # ``__init__`` mirrors the weight map's own condition
        # (``weight_loaders_fp8.py:481``). Checked rather than assumed, because
        # ``getattr`` on a missing module would raise ``AttributeError`` from
        # inside the shared path and read as a wiring bug rather than a config.
        shared_experts = getattr(self, "shared_experts", None)
        if shared_experts is None:
            raise Glm5NextSharedExpertRouteError(
                "combine_routed_and_shared was called on a block that declares "
                "no shared expert (n_shared_experts == 0), so there is no shared "
                "contribution to add. The routed output is already the layer "
                "output on such a block and calling this method is the bug."
            )

        # THE ONE call to the shared path.
        shared_output = shared_experts.shared_expert_mm(
            hidden_states,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            gate_proj_scale,
            up_proj_scale,
            down_proj_scale,
            quant_config,
        )

        # Extents compared EXACTLY, before the add. ``torch`` would broadcast a
        # disagreeing extent silently and return a plausible tensor of the wrong
        # shape, which no downstream shape check would catch.
        if tuple(routed_output.shape) != tuple(shared_output.shape):
            raise Glm5NextSharedExpertRouteError(
                f"routed_output has shape {tuple(routed_output.shape)} and the "
                f"shared contribution has shape {tuple(shared_output.shape)}; "
                f"they must agree exactly. Refusing to add: torch would "
                f"broadcast these and compute a different layer without error."
            )

        # THE ONE add. The shared contribution enters the layer output here and
        # nowhere else in this file.
        return routed_output + shared_output

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextMoEBlock.forward is a stub created by inc-glm53f-013; "
            "its sections land with inc-glm53f-031, -032, -027 and -033"
        )


class Glm5NextDenseMLP(nn.Module):
    """The dense MLP on the first ``first_k_dense_replace`` layers.

    Gate and up stay **separate** parameters, matching the map, which follows
    the fork's own dense precedent rather than fusing them
    (``weight_loaders_fp8.py:429-431``).
    """

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.intermediate_size = int(text_config.intermediate_size)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextDenseMLP.forward is a stub created by inc-glm53f-013; "
            "the dense-MLP compute path lands with inc-glm53f-054's forward"
        )


def _build_mlp(text_config: Glm5NextTextConfig, layer_idx: int) -> nn.Module:
    """Dense below ``first_k_dense_replace``, sparse at and above it.

    The same predicate the map uses (``weight_loaders_fp8.py:341-344``), so
    the two sides cannot disagree about which layers carry experts.
    """
    if layer_idx < int(text_config.first_k_dense_replace):
        return Glm5NextDenseMLP(text_config)
    return Glm5NextMoEBlock(text_config)


# ---------------------------------------------------------------------------
# The KDA (``linear_attention``) half. D14 owner: ``inc-glm53f-038`` (M3),
# whose acceptance runs "a 3-layer KDA stack" -- so ``Glm5NextKDALayer`` is
# the decoder layer, and the gated-delta module it holds sits at the map's
# ``linear_attn`` path.
# ---------------------------------------------------------------------------


class Glm5NextKDAAttention(nn.Module):
    """Gated-delta linear attention at ``linear_attn``.

    Every parameter name here is the landed map's
    (``weight_loaders_fp8.py:406-418``), including the three unprojected
    entries that have no ``.weight`` leaf: ``conv1d_bias``, ``dt_bias`` and
    ``A_log``.

    THE CACHE GEOMETRY THIS CARRIES, STATED EXACTLY. A linear-attention layer
    holds a **recurrent state**, not a key/value history. The pin's
    ``LayerSpec`` has no vocabulary for that state -- adding it is
    ``inc-glm53f-015``'s declared surface at M1 -- so the spec this layer
    reports describes its per-head state extent in the pin's existing fields
    and asserts nothing about recurrent-state layout. That limit is recorded
    rather than papered over.
    """

    #: The map's module path for this family, and therefore the suffix of the
    #: ``LayerSpec.name`` this layer reports.
    CACHE_NAME_SUFFIX = "linear_attn"

    def __init__(self, text_config: Glm5NextTextConfig, world_size: int) -> None:
        super().__init__()
        self.num_heads = _linear_attn_field(text_config, "num_heads")
        self.head_dim = _linear_attn_field(text_config, "head_dim")
        self.short_conv_kernel_size = _linear_attn_field(
            text_config, "short_conv_kernel_size"
        )

        # Head-count sharding follows the fork's KV-spec convention (per-rank
        # counts); the state width per head does not shard.
        self.num_kv_heads_per_rank = _per_rank(self.num_heads, world_size)
        self.head_size = self.head_dim
        self.cache_dtype = _resolve_kda_state_dtype(text_config)
        # ``NeuronConfig.kda_state_chunk_size`` is declared a tuning dial with
        # "None = let the layer choose" (``neuron_config.py:185-188``), and
        # ``LayerSpec.chunk_size`` already exists at the pin, so the dial is
        # passed through rather than dropped. None by default.
        self.cache_chunk_size = getattr(
            text_config.neuron_config, "kda_state_chunk_size", None
        )

        _declare_parameters(
            self,
            "in_proj_qkvz_weight",
            "in_proj_ba_weight",
            "out_proj_weight",
            "conv1d_weight",
            "norm_weight",
            "conv1d_bias",
            "dt_bias",
            "A_log",
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextKDAAttention.forward is a stub created by "
            "inc-glm53f-013; the KDA path lands with inc-glm53f-034..-038"
        )


class Glm5NextKDALayer(nn.Module):
    """Decoder layer on the ``linear_attention`` half of the hybrid stack.

    Section reserved for ``inc-glm53f-038`` (KDA layer + runner state
    plumbing).
    """

    #: Attribute the family's attention module is bound to -- the map's own
    #: module path (``weight_loaders_fp8.py:406``).
    ATTENTION_ATTR = "linear_attn"

    def __init__(
        self, text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = KDA_LAYER_TYPE
        _declare_parameters(
            self, "input_layernorm_weight", "post_attention_layernorm_weight"
        )
        self.linear_attn = Glm5NextKDAAttention(text_config, world_size)
        self.mlp = _build_mlp(text_config, layer_idx)

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextKDALayer.forward is a stub created by inc-glm53f-013; "
            "the KDA layer lands with inc-glm53f-038 and the full 45-layer "
            "forward with inc-glm53f-054"
        )


# ---------------------------------------------------------------------------
# The DSA (``deepseek_sparse_attention``) half. D14 owners:
# ``inc-glm53f-039`` (MLA projections) and ``-042`` (MLA decode path) inside
# ``Glm5NextMLAAttention``; ``-051`` for ``Glm5NextDSALayer`` (layer +
# sequence tiling), whose acceptance runs "a 3-layer DSA stack" -- so that
# class is the decoder layer and the MLA module sits at ``self_attn``.
# ---------------------------------------------------------------------------


class Glm5NextDSAIndexer(nn.Module):
    """The DSA sparse indexer at ``self_attn.indexer``.

    PROVISIONAL leaf names, flagged as such on the landed side too
    (``weight_loaders_fp8.py:376``): ``index_n_heads`` / ``index_head_dim``
    are in the checkpoint config but the indexer's leaf names are unconfirmed
    against the shard index. Carried here exactly as the map declares them so
    the two sides move together if they move at all.
    """

    def __init__(self) -> None:
        super().__init__()
        _declare_parameters(
            self, "wq_weight", "wk_weight", "k_norm_weight", "weights_proj_weight"
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextDSAIndexer.forward is a stub created by "
            "inc-glm53f-013; the DSA indexer lands with the DSA path"
        )


class Glm5NextMLAAttention(nn.Module):
    """Multi-head latent attention at ``self_attn``, NoPE on this checkpoint.

    Parameter names are the landed map's
    (``weight_loaders_fp8.py:366-389``). No ``*_rope_*`` projection exists:
    ``mla_use_nope`` with ``qk_rope_head_dim == 0`` means there is no rotary
    head slice at all.

    THE CACHE GEOMETRY. MLA caches **one compressed latent vector per token
    per layer**, not one entry per attention head. The map's own projection
    name says so -- ``kv_a_proj_with_mqa``, multi-query, a single KV head --
    so ``num_kv_heads`` is 1 and is **not** tensor-parallel sharded: a single
    latent is replicated across ranks rather than split. The width is
    ``kv_lora_rank + qk_rope_head_dim``, which is 512 on this checkpoint.
    """

    #: The map's module path for this family (``weight_loaders_fp8.py:366``).
    CACHE_NAME_SUFFIX = "self_attn"

    #: MLA compresses KV to one latent per token; the latent is replicated
    #: across tensor-parallel ranks, so this is 1 at every world size.
    NUM_LATENT_KV_HEADS = 1

    def __init__(self, text_config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.num_attention_heads = int(text_config.num_attention_heads)
        self.kv_lora_rank = int(text_config.kv_lora_rank)
        self.q_lora_rank = int(text_config.q_lora_rank)
        self.qk_nope_head_dim = int(text_config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(text_config.qk_rope_head_dim)
        self.v_head_dim = int(text_config.v_head_dim)
        self.mla_use_nope = bool(text_config.mla_use_nope)

        self.num_kv_heads_per_rank = self.NUM_LATENT_KV_HEADS
        self.head_size = _resolve_mla_head_size(text_config)
        self.cache_dtype = _resolve_model_dtype(text_config)
        self.cache_chunk_size = None

        _declare_parameters(
            self,
            "q_a_proj_weight",
            "q_b_proj_weight",
            "kv_a_proj_with_mqa_weight",
            "kv_b_proj_weight",
            "o_proj_weight",
            "q_a_layernorm_weight",
            "kv_a_layernorm_weight",
        )
        self.indexer = Glm5NextDSAIndexer()

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextMLAAttention.forward is a stub created by "
            "inc-glm53f-013; the projections land with inc-glm53f-039 and "
            "the decode path with inc-glm53f-042"
        )


class Glm5NextDSALayer(nn.Module):
    """Decoder layer on the ``deepseek_sparse_attention`` half.

    Section reserved for ``inc-glm53f-051`` (DSA runner integration and
    sequence tiling).
    """

    #: Attribute the family's attention module is bound to -- the map's own
    #: module path (``weight_loaders_fp8.py:366``).
    ATTENTION_ATTR = "self_attn"

    def __init__(
        self, text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = DSA_LAYER_TYPE
        _declare_parameters(
            self, "input_layernorm_weight", "post_attention_layernorm_weight"
        )
        self.self_attn = Glm5NextMLAAttention(text_config)
        self.mlp = _build_mlp(text_config, layer_idx)

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextDSALayer.forward is a stub created by inc-glm53f-013; "
            "the DSA layer lands with inc-glm53f-051 and the full 45-layer "
            "forward with inc-glm53f-054"
        )


def _build_layer(
    text_config: Glm5NextTextConfig, layer_idx: int, layer_type: str, world_size: int
) -> nn.Module:
    """One decoder layer, family chosen by EQUALITY on ``layer_types``.

    Never by substring: ``"attention"`` is a substring of both family names
    (``config.py:33-37``), so a substring test would silently mis-partition
    the stack -- and mis-partitioning it is exactly what the 34/11 split
    would then fail to detect.
    """
    if layer_type == DSA_LAYER_TYPE:
        return Glm5NextDSALayer(text_config, layer_idx, world_size)
    if layer_type == KDA_LAYER_TYPE:
        return Glm5NextKDALayer(text_config, layer_idx, world_size)
    raise ValueError(
        f"layer {layer_idx} declares unrecognised attention family "
        f"{layer_type!r}; expected {KDA_LAYER_TYPE!r} or {DSA_LAYER_TYPE!r}"
    )


# ---------------------------------------------------------------------------
# The tree. D14 owner for this section: ``inc-glm53f-013`` (creator), then
# ``inc-glm53f-054`` at M4, which replaces the stubs with the full 45-layer
# forward.
# ---------------------------------------------------------------------------


class Glm5NextModel(nn.Module):
    """The decoder stack, named ``model`` because the map's paths say so.

    Every mapped parameter outside the layer stack hangs here or on the root
    (``weight_loaders_fp8.py:315-319``).
    """

    def __init__(self, config: Glm5NextConfig, world_size: int) -> None:
        super().__init__()
        text_config = config.text_config
        self.config = config
        self.text_config = text_config
        self.vocab_size = int(text_config.vocab_size)
        self.hidden_size = int(text_config.hidden_size)

        _declare_parameters(self, "embed_tokens_weight", "norm_weight")

        layer_types = list(text_config.layer_types or ())
        self.layers = nn.ModuleList(
            [
                _build_layer(text_config, layer_idx, layer_type, world_size)
                for layer_idx, layer_type in enumerate(layer_types)
            ]
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextModel.forward is a stub created by inc-glm53f-013; the "
            "full 45-layer forward lands with inc-glm53f-054"
        )


class Glm5NextForConditionalGeneration(nn.Module):
    """The blockwise-FP8 GLM-5.3-Flash implementation.

    The module path, this class name and the ``from_configs`` signature are
    **pinned by landed code**: ``factory.py:68`` already reads ``from
    .model_fp8 import Glm5NextForConditionalGeneration as Model`` and
    ``factory.py:70-74`` calls
    ``Model.from_configs(hf_config, text_neuron_config=..., vision_neuron_config=...)``.
    The name is duplicated with ``factory.py:21`` on purpose -- that is the
    plugin's selector-to-implementation convention, the same pair
    ``llama3/factory.py:42`` uses -- so neither side is renamed here.
    """

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        if config.text_config is None:
            raise ValueError(
                "Glm5NextConfig.text_config is required to build the decoder "
                "stack; got None"
            )
        self.config = config
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        self.world_size = _resolve_world_size()

        self.model = Glm5NextModel(config, self.world_size)
        # Mirrors the map's own condition: a tied head has no separate
        # checkpoint key and therefore no separate parameter
        # (``weight_loaders_fp8.py:318-319``).
        if not self.text_config.tie_word_embeddings:
            _declare_parameters(self, "lm_head_weight")

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> Glm5NextForConditionalGeneration:
        """Build from an HF config, the signature ``factory.py:70-74`` calls."""
        config = Glm5NextConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    # ── declared parameter names ─────────────────────────────────────────

    def declared_parameter_names(self) -> tuple[str, ...]:
        """Every parameter attribute path this tree declares, in tree order.

        This is the enumerable form of the derivation the lead ruling
        requires: the set returned here is compared against
        ``build_weight_mappings(text_config)``'s keys in
        ``test_kv_spec.py``. ``named_parameters()`` cannot be used, because
        declared-but-unmaterialised parameters are ``None`` and torch skips
        ``None`` entries in both ``named_parameters()`` and ``state_dict()``.
        """
        names: list[str] = []
        for module_path, module in self.named_modules():
            for leaf in getattr(module, "declared_param_names", ()):
                names.append(f"{module_path}.{leaf}" if module_path else leaf)
        return tuple(names)

    # ── KV cache management ──────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

    def get_kv_spec(self) -> KVSpec:
        """One ``LayerSpec`` per layer of the hybrid stack, in layer order.

        Field mapping and naming follow ``llama3/model.py:1781-1795``; the
        construction path does **not**, because that precedent reads
        instantiated submodules and this stack is never instantiated with
        weights. Geometry is read off the declared layer objects instead.

        The two families report different geometry from one uniform read:
        each layer exposes its attention module and that module declares
        ``num_kv_heads_per_rank``, ``head_size``, ``cache_dtype`` and
        ``cache_chunk_size``. ``sliding_window_size`` is ``None`` on every
        entry -- this arch declares no sliding window on either half.
        """
        layers: list[LayerSpec] = []
        for layer_idx, layer in enumerate(self.model.layers):
            attention = layer.attention
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_idx}.{attention.CACHE_NAME_SUFFIX}",
                    num_kv_heads=attention.num_kv_heads_per_rank,
                    head_size=attention.head_size,
                    dtype=attention.cache_dtype,
                    sliding_window_size=None,
                    chunk_size=attention.cache_chunk_size,
                )
            )
        return KVSpec(layers=layers)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration.forward is a stub created by "
            "inc-glm53f-013; the full 45-layer forward lands with "
            "inc-glm53f-054"
        )
