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
genuinely exists on the module (``module.o_proj_weight`` returns ``None``),
the name is enumerable through :meth:`declared_parameter_names`, and the tree
allocates **zero** ``torch.nn.Parameter`` objects. ``get_kv_spec`` therefore
reads geometry off layer objects -- the same construction shape as
``llama3/model.py:1781`` and ``synthetic/synthetic.py:98`` -- without the
allocation that shape would otherwise imply.

WHERE THE PARAMETER NAMES COME FROM -- THEY ARE NOT CHOSEN HERE
---------------------------------------------------------------
Per the lead ruling *"-013's skeleton parameter names: the LANDED weight map's
param-name side is the authority"* -- recorded at
``artifacts/campaigns/glm-5.3-flash-port/increments/evidence-013.md`` L212,
since the original ``approvals/lead-ruling-013-param-name-authority.md`` was
deleted in the 2026-08-31 residue purge -- every parameter
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
(``weight_loaders_fp8.py:1104-1111`` states the same kind of fact).
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
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FP8_SCALE_SUFFIX,
    dequantise_blockwise,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

# ---------------------------------------------------------------------------
# Parameter declaration -- the mechanism that makes a name exist without
# allocating the tensor behind it.
# ---------------------------------------------------------------------------


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    """True for any fp8 dtype the installed ``torch`` has.

    CLASSIFIED, NEVER NAME-MATCHED, and the rule is the fork's own rather than
    this increment's: a one-byte floating dtype whose ``str()`` names ``float8``
    (``vllm_neuron/accuracy/testing.py:86``, and its test-side twin
    ``test/vllm_neuron/accuracy/test_glm5next_tolerance_registry.py:142``).
    Classified rather than hard-coded because which fp8 dtypes exist is a
    property of the installed ``torch``, not of this file.
    """
    return dtype.itemsize == 1 and "float8" in str(dtype)


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
# (M2, Lane B 6th, and Lane B's last).
#
# WHAT LANDS HERE, AND WHAT DELIBERATELY DOES NOT
# -----------------------------------------------
# The whole mHC layer: the pre block, ONE entry into ``inc-glm53f-028``'s
# Sinkhorn seam, ONE entry into ``inc-glm53f-029``'s combine seam, and the
# residual plumbing between them. **Calling this layer from the decoder is NOT
# here.** ``Glm5NextKDALayer`` and ``Glm5NextDSALayer`` are other increments'
# D14 sections (``-038``, ``-051``), and D14's rule is to raise rather than
# widen, so this increment stops at its own section boundary and the decoder
# call site is left to those owners. :meth:`Glm5NextHyperConnection.forward` is
# shaped as the seam they will call.
#
# NO WEIGHT-MAP FAMILY IS ADDED, and that is a lead ruling rather than an
# omission. ``weight_loaders_fp8.py:88-89`` declared ``multi_hyper_connections``
# absent here -- no settled leaf names -- until ``inc-glm53f-078`` read the real
# shard index and grounded it. This increment's acceptance is a synthetic
# tiny case that builds these parameters in-test, so the map is not on its
# route, and the checkpoint-leaf question stays a lead-owned open question for a
# later design revision. ``weight_loaders_fp8.py`` is untouched.
#
# NO TOKEN TILING IS AUTHORED HERE, also a lead ruling. ``-028``'s kernel
# refuses its ``M`` above ``PARTITION_MAX`` by raising ``SinkhornError`` and has
# no torch path (``sinkhorn.py:321-330``); the refusal is allowed to propagate
# unchanged. A host-side tiling loop and a large-``M`` torch fallback are both
# out of scope, the second by P13 outright. The measured consequence -- see the
# token-ceiling note on :meth:`mhc_pre` -- is recorded for the design revision
# that settles the policy.
# ---------------------------------------------------------------------------


class Glm5NextHyperConnectionError(ValueError):
    """A rank, extent or configuration this layer refuses, named not coerced.

    Only the cross-argument agreements the two seams cannot see are checked
    here. Each seam already refuses its own extents by name
    (``sinkhorn.py:312-342``, ``hyper_connection.py:238-308``), and restating
    those bounds would create a second authority that can drift from the first
    -- the same reason ``-033``'s route error checks only its own call site.
    """


class Glm5NextHyperConnection(nn.Module):
    """Multi-hyper-connection (mHC) residual mixing, one layer call.

    The operation is the pinned base's own pair of mHC blocks, ``MHCPreOp`` and
    ``MHCPostOp`` (``vllm/model_executor/layers/mhc.py`` at ``vllm==0.24.0``),
    with the two device-side pieces routed to this fork's NKI kernels:

    1. **pre** -- project the ``hc_mult`` residual streams through ``fn``, RMS
       scale, and split the result into three heads: ``pre_mix`` (which folds
       the streams into the sub-block's single input), ``post_mix``, and the
       ``[S, S]`` per-token stream-mixing matrix. The mixing matrix is then
       Sinkhorn-normalised by **``inc-glm53f-028``'s kernel**.
    2. the caller's sub-block runs on the folded input.
    3. **post** -- mix the streams back out by **``inc-glm53f-029``'s kernel**.

    WHERE THE ARITHMETIC COMES FROM -- IT IS NOT CHOSEN HERE
    -------------------------------------------------------
    Every line below is the base's, read at tag ``v0.24.0`` because the
    campaign's target base is the 0.24 line, and read in **two independent
    spellings that agree**: ``mhc_pre_torch`` / ``mhc_post_torch`` in
    ``vllm/model_executor/kernels/mhc/torch.py`` (the plain-torch backend), and
    ``mhc_pre_ref`` / ``mhc_post_ref`` in ``tests/kernels/test_mhc_kernels.py``
    (the TileLang-repo reference). The acceptance carries both and asserts they
    agree, so the transcription rests on two statements rather than on one
    reading of one file.

    THE ONE COMPOSITION QUESTION THIS SECTION HAD TO ANSWER
    ------------------------------------------------------
    ``-028``'s seam normalises **one** ``[M, N]`` matrix; the target needs
    **``T`` independent** ``[S, S]`` ones, and this increment's route predicate
    declares the Sinkhorn seam is entered **exactly once per layer call**. The
    two are reconciled by a **block-diagonal embedding**: the ``T`` little
    matrices are scattered onto the diagonal of one ``[T*S, T*S]`` matrix, so
    ``-028``'s column target ``M / N`` is exactly ``1`` -- the target's own
    column target -- and the off-diagonal zeros stay zero under multiplicative
    rescaling, which makes every row sum and every column sum range over
    exactly one token's block. **The alternative was measured and rejected:** a
    flat ``[T*S, S]`` reshape lets ``-028``'s column pass sum ACROSS tokens, and
    ``probe-030-composition-algebra.out`` reads ``max_abs`` up to ``4.68e-01``
    against the target for it while the block-diagonal embedding reads
    ``8.99e-07``, with the off-block maximum exactly ``0.0``. This is layout,
    not authored numerics: all of the Sinkhorn arithmetic stays inside ``-028``'s
    kernel.

    :func:`torch.block_diag` is torch's own member, reused rather than written.

    TWO DIVERGENCES FROM THE BASE THAT THIS LAYER CANNOT REMOVE
    ----------------------------------------------------------
    Both live inside ``-028``'s landed kernel, so both are recorded rather than
    repaired: the base adds ``hc_sinkhorn_eps`` to every Sinkhorn denominator
    while ``-028`` adds an inert ``1e-30`` (``sinkhorn.py:148``), and the base's
    schedule is ``softmax`` then one column pass then ``(R-1)`` row/column pairs
    while ``-028``'s is ``R`` pairs starting with a row pass. Sinkhorn-Knopp has
    one fixed point, so at the target's ``20`` iterations the gap is small --
    ``probe-030-layer-delta.out`` measures it using at most **3.2 %** of the
    declared tolerance budget, in exact double precision so the reading is the
    schedule-and-eps gap alone. The acceptance measures it again through the
    real kernels.

    PRECISION: fp32 IN AND OUT, following ``-028`` and ``-029``
    ---------------------------------------------------------
    The base's mHC takes bf16 and its own kernel test therefore compares at
    ``atol=5e-2``; this increment is declared at ``atol=1e-5``, three orders
    tighter, and bf16's ~3 decimal digits cannot express that difference. Both
    seams are fp32 in and fp32 out for exactly this reason, so this layer keeps
    fp32 across them and returns the seam's own dtype. Casting the result to the
    decoder's residual dtype is the decoder's business, as ``-027`` and ``-033``
    both left it.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        neuron_config: NeuronConfig | None = None,
        post_mult_value: float = 1.0,
    ) -> None:
        """Size the layer from the checkpoint's own dials.

        Args:
            text_config: carries ``hc_mult``, ``hc_sinkhorn_iters`` and
                ``hc_eps`` (``config.py:225-227``).
            neuron_config: framework overrides. ``mhc_sinkhorn_iters`` and
                ``mhc_eps`` (``neuron_config.py:194,197``) win when not
                ``None``, which is the override contract ``-013``'s section note
                for this class already stated.
            post_mult_value: the base's ``hc_post_mult_value``. **Its default is
                the base's own test value**, ``hc_post_alpha = 1.0``
                (``tests/kernels/test_mhc_kernels.py:126``); no fork config
                field carries it, so it is a constructor argument rather than an
                invented config default.

        TWO EPSILONS FOR THE BASE'S THREE, GROUNDED ON THE CHECKPOINT. The
        base's signature takes three (``rms_eps``, ``hc_pre_eps``,
        ``hc_sinkhorn_eps``) and the fork's config carries two fields. The
        split follows what the checkpoint sets, not what the base's test sets:

        * ``hc_pre_eps`` and ``hc_sinkhorn_eps`` are mHC-native, and the
          checkpoint's own ``text_config.hc_eps`` is ``1e-06``, so both keep
          ``hc_eps`` and the base's collapse onto one value is faithful for
          them. That is the value ``inc-glm53f-030`` measured its tiny case on,
          and nothing it recorded moves.
        * ``rms_eps`` is an RMSNorm epsilon, and the checkpoint's RMSNorm
          epsilon is ``1e-05`` -- a different number. It lives on
          ``Glm5NextTextConfig.rms_norm_eps`` (``inc-glm53f-080``) and reaches
          the router seam through :meth:`Glm5NextRoutedExperts.route_tokens`.
          It reaches no mHC line: ``self.hc_eps`` and the three sites that
          consume it below are unchanged.

        WHAT THIS CORRECTS. The earlier wording argued the single field was
        faithful for all three uses, and grounded that on the base's own kernel
        test setting ``hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6``
        (``tests/kernels/test_mhc_kernels.py:121``). That is the base's number,
        not the target's. Two thirds of the claim stand on the checkpoint's own
        ``hc_eps``; the RMSNorm third is settled against the checkpoint
        instead, which is where it always belonged.

        Raises:
            Glm5NextHyperConnectionError: on a non-positive ``hc_mult``,
                ``hidden_size`` or iteration count.
        """
        super().__init__()
        hc_mult = int(text_config.hc_mult)
        hidden = int(text_config.hidden_size)
        iters = int(text_config.hc_sinkhorn_iters)
        eps = float(text_config.hc_eps)
        if neuron_config is not None:
            if neuron_config.mhc_sinkhorn_iters is not None:
                iters = int(neuron_config.mhc_sinkhorn_iters)
            if neuron_config.mhc_eps is not None:
                eps = float(neuron_config.mhc_eps)

        problems: list[str] = []
        if hc_mult <= 0:
            problems.append(f"hc_mult={hc_mult} must be positive")
        if hidden <= 0:
            problems.append(f"hidden_size={hidden} must be positive")
        if iters <= 0:
            problems.append(
                f"sinkhorn iterations={iters} must be positive; the checkpoint "
                f"declares hc_sinkhorn_iters={text_config.hc_sinkhorn_iters}"
            )
        if problems:
            raise Glm5NextHyperConnectionError(
                "mHC layer refuses this configuration: " + "; ".join(problems)
            )

        self.hc_mult = hc_mult
        self.hidden_size = hidden
        self.sinkhorn_iters = iters
        self.hc_eps = eps
        self.post_mult_value = float(post_mult_value)

        # ``hc_mult3`` is the base's own name for the projection's output width:
        # ``hc_mult`` pre weights + ``hc_mult`` post weights + ``hc_mult ** 2``
        # mixing weights, in that order, which is the order the three heads are
        # sliced out of ``mixes`` below.
        self.hc_mult3 = 2 * hc_mult + hc_mult * hc_mult

        # ORDINARY PARAMETERS, not ``_declare_parameters``' reservations, and
        # that is the lead's ruling for this section: the declared acceptance is
        # a synthetic case whose test SETS these tensors. The three names are
        # the base's own ``mhc_pre`` argument names, taken rather than chosen,
        # exactly as ``-029`` took its seam's signature -- they are THIS CLASS'S
        # OWN SIGNATURE AND NOT MAP NAMES.
        #
        # RE-GROUNDED BY ``inc-glm53f-082``. This note used to add "the weight
        # map declares this family absent, so there is no map name to reserve",
        # which ``inc-glm53f-078`` falsified: the map now emits six mHC names
        # per layer (``MHC_LEAVES``). Those six are reserved FLAT ON THE LAYER
        # by ``Glm5NextKDALayer`` and ``Glm5NextDSALayer``, not here, because
        # this class is not bound into a layer anywhere in this tree. The three
        # ``nn.Parameter`` names below are unchanged.
        self.fn = nn.Parameter(
            torch.zeros(self.hc_mult3, hc_mult * hidden, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.zeros(self.hc_mult3, dtype=torch.float32))

    # ── mHC pre -- the folded input, and ONE Sinkhorn dispatch ────────────
    def mhc_pre(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The base's ``mhc_pre``, with its Sinkhorn on ``-028``'s kernel.

        Args:
            residual: ``[T, S, H]`` -- the ``S = hc_mult`` residual streams.

        Returns:
            ``(post_mix, comb_mix, layer_input)`` -- ``[T, S, 1]``,
            ``[T, S, S]`` and ``[T, H]``, all fp32. ``comb_mix[t, i, j]``
            weights input stream ``i`` into output stream ``j``, the base's
            convention and the one ``-029``'s kernel reads.

        THE TOKEN CEILING, MEASURED AND DELIBERATELY NOT WORKED AROUND. The
        block-diagonal embedding puts ``T * S`` on the Sinkhorn's ``M``, and
        ``-028`` refuses ``M > PARTITION_MAX``. So this layer serves
        ``T <= PARTITION_MAX // S`` -- **32** at the checkpoint's ``hc_mult 4``
        -- where ``-029``'s combine kernel on its own would serve ``T <= 128``.
        The Sinkhorn therefore binds first. Nothing here pads, tiles or falls
        back: the seam's ``SinkhornError`` propagates to the caller unchanged,
        which is what the lead ruled and what P13 requires. The reading is in
        ``probe-030-m129.out``.

        Raises:
            Glm5NextHyperConnectionError: on a non-3-D ``residual`` or a stream
                or hidden extent that contradicts this layer's configuration.
            SinkhornError: from the seam, on a token count this layer's
                embedding puts above ``-028``'s ``M`` bound. Propagated, never
                caught.
        """
        from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_normalise

        tokens, streams, hidden = self._require_streams(residual)

        flat = residual.reshape(tokens, streams * hidden).to(torch.float32)
        mixes = flat @ self.fn.to(torch.float32).t()
        # The RMS scale. Both upstream spellings divide the squared sum by the
        # projection's own input width -- ``hc_mult * hidden_size`` in
        # ``mhc_pre_torch``, ``fn.shape[-1]`` in ``mhc_pre_ref`` -- and those are
        # the same number.
        sqrsum = flat.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / float(streams * hidden) + self.hc_eps)

        scale = self.hc_scale.to(torch.float32)
        base = self.hc_base.to(torch.float32)
        pre_mix = (
            torch.sigmoid(mixes[:, :streams] * scale[0] + base[:streams])
            + self.hc_eps
        )
        post_mix = (
            torch.sigmoid(
                mixes[:, streams : 2 * streams] * scale[1]
                + base[streams : 2 * streams]
            )
            * self.post_mult_value
        )
        comb_logits = mixes[:, 2 * streams :].reshape(
            tokens, streams, streams
        ) * scale[2] + base[2 * streams :].reshape(1, streams, streams)
        # ``softmax`` and the ``+ eps`` are the base's, and they sit OUTSIDE the
        # seam because ``-028``'s kernel starts from an affinity matrix. This is
        # elementwise glue, which P13 leaves to torch.
        comb_start = torch.softmax(comb_logits, dim=-1) + self.hc_eps

        # ---- ENTRY 1 of 1 into ``-028``'s Sinkhorn seam. ----------------- #
        # The counted dispatch. One call for all ``T`` tokens, which is what the
        # block-diagonal embedding buys and what the route predicate declares.
        normalised = sinkhorn_normalise(
            torch.block_diag(*comb_start.unbind(0)), iters=self.sinkhorn_iters
        )
        comb_mix = self._diagonal_blocks(normalised, tokens, streams)

        layer_input = (pre_mix.unsqueeze(-1) * residual.to(torch.float32)).sum(dim=1)
        return post_mix.reshape(tokens, streams, 1), comb_mix, layer_input

    # ── mHC post -- ONE combine dispatch ──────────────────────────────────
    def mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """The base's ``mhc_post``, on ``-029``'s kernel.

        Args:
            x: ``[T, H]`` -- the sub-block's single-stream output.
            residual: ``[T, S, H]`` -- the streams, unchanged from
                :meth:`mhc_pre`'s input.
            post_layer_mix: ``[T, S, 1]`` from :meth:`mhc_pre`.
            comb_res_mix: ``[T, S, S]`` from :meth:`mhc_pre`.

        Returns:
            ``[T, S, H]`` fp32 -- the seam's own return dtype, not re-cast.

        Raises:
            HyperConnectionError: from the seam, on any inadmissible rank or
                extent. Propagated, never caught: a geometry the kernel cannot
                serve must not quietly reach a torch path (P13, D6).
        """
        from vllm_neuron.functional.mhc.hyper_connection import (
            hyper_connection_combine,
        )

        # ---- ENTRY 1 of 1 into ``-029``'s combine seam. ------------------ #
        # Argument names and order are the seam's, which are the base's, so this
        # is a call rather than a translation -- ``hyper_connection.py:375-376``
        # asks for exactly that.
        return hyper_connection_combine(
            x=x.to(torch.float32),
            residual=residual.to(torch.float32),
            post_layer_mix=post_layer_mix.to(torch.float32),
            comb_res_mix=comb_res_mix.to(torch.float32),
        )

    # ── one layer call ────────────────────────────────────────────────────
    def forward(self, residual: torch.Tensor, sublayer: object) -> torch.Tensor:
        """One mHC layer call: pre, then the sub-block, then post.

        This is the shape the decoder sections call, and it is why "per layer
        call" is a well-defined read window for the two counters: one entry here
        is exactly one Sinkhorn dispatch and one combine dispatch.

        Args:
            residual: ``[T, S, H]`` residual streams.
            sublayer: the wrapped sub-block, a callable ``[T, H] -> [T, H]``.
                Annotated ``object`` rather than ``Callable`` on purpose -- the
                module-level import block is ``-013``'s D14 section, and ``-023``
                set the precedent that a later section adds nothing to it, so
                every import in this section is function-local.

        Returns:
            ``[T, S, H]`` fp32 -- the re-mixed streams.

        Raises:
            Glm5NextHyperConnectionError: if ``sublayer`` is not callable, or if
                its output does not have the ``[T, H]`` shape the combine needs.
        """
        if not callable(sublayer):
            raise Glm5NextHyperConnectionError(
                f"sublayer must be a callable [T, H] -> [T, H], got "
                f"{type(sublayer).__name__}"
            )
        post_mix, comb_mix, layer_input = self.mhc_pre(residual)
        x = sublayer(layer_input)
        if not isinstance(x, torch.Tensor) or tuple(x.shape) != tuple(
            layer_input.shape
        ):
            got = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
            raise Glm5NextHyperConnectionError(
                f"sublayer returned {got}, expected the [T, H] shape it was "
                f"given, {tuple(layer_input.shape)} -- the combine reads x and "
                f"the streams on the same token and hidden extents"
            )
        return self.mhc_post(x, residual, post_mix, comb_mix)

    # ── layout helpers -- no arithmetic, and that is the point ────────────
    def _require_streams(self, residual: torch.Tensor) -> tuple[int, int, int]:
        """``(T, S, H)``, once the cross-argument agreements hold.

        Only what the seams cannot see: they read their extents off the tensors
        they are handed, so neither can tell that a stream or hidden extent
        disagrees with the CONFIG this layer was built from -- and a wrong
        stream count would silently mis-slice ``mixes`` instead of failing.
        """
        if residual.dim() != 3:
            raise Glm5NextHyperConnectionError(
                f"residual must be 3-D [T, S, H], got shape "
                f"{tuple(residual.shape)}"
            )
        tokens, streams, hidden = (int(v) for v in residual.shape)
        if streams != self.hc_mult:
            raise Glm5NextHyperConnectionError(
                f"residual carries S={streams} streams and this layer was built "
                f"for hc_mult={self.hc_mult}; the projection's output width "
                f"{self.hc_mult3} is sliced into three heads by that number, so "
                f"a mismatch would mis-slice rather than fail"
            )
        if hidden != self.hidden_size:
            raise Glm5NextHyperConnectionError(
                f"residual carries H={hidden} and this layer was built for "
                f"hidden_size={self.hidden_size}; fn is "
                f"[{self.hc_mult3}, {self.hc_mult * self.hidden_size}]"
            )
        return tokens, streams, hidden

    @staticmethod
    def _diagonal_blocks(
        normalised: torch.Tensor, tokens: int, streams: int
    ) -> torch.Tensor:
        """Read the ``T`` per-token blocks back off a ``[T*S, T*S]`` diagonal.

        The inverse of :func:`torch.block_diag` for equal-sized blocks. Pure
        indexing: it selects, and computes nothing.
        """
        index = torch.arange(tokens, device=normalised.device)
        return normalised.reshape(tokens, streams, tokens, streams)[
            index, :, index, :
        ]


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
    (``weight_loaders_fp8.py:670-679``), because this checkpoint stores one
    tensor per expert while the fork's parameter side is per-projection. The
    router lives here too, and carries a bias because
    ``topk_method == "noaux_tc"`` (``weight_loaders_fp8.py:654-655``).
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
    # THE 288/64 REFUSAL IS GONE, AND IT WAS NEVER THE FORK'S RULE --
    # ``inc-glm53f-087``. This paragraph used to say the refusal at the registered
    # tensor-parallel degree freeze of 64 was deliberate and visible, and that it
    # was campaign gap G4 surfaced where the model is built. That was wrong on the
    # only point that mattered: an expert bank divides by the EXPERT-PARALLEL
    # degree, never by the tensor-parallel one. The fork's one landed EP-aware
    # bank divides ``num_local_experts // self.ep_degree``
    # (``gpt_oss/model_bf16.py:1072``) and shards the INTERMEDIATE dimension by TP
    # instead (``:986-988``). So with expert parallelism off -- this campaign's
    # route -- the degree is 1, all 288 experts are local on every rank and
    # NOTHING RAISES at TP = 64. The raggedness gate is kept, and its subject is
    # now the expert-parallel degree, which is the one thing the old code got
    # right: a named error rather than ``gpt_oss``'s silent floor division.

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        world_size: int | None = None,
        ep_degree: int | None = None,
    ) -> None:
        super().__init__()
        from vllm_neuron.model.glm5_next.factory import (
            _resolve_ep_degree,
            require_uniform_expert_partition,
        )

        self.num_routed_experts = int(text_config.n_routed_experts)
        self.num_experts_per_tok = int(text_config.num_experts_per_tok)
        # ``_resolve_world_size()`` is ``-013``'s helper, called rather than
        # edited: an explicit ``world_size`` is what a caller with a degree
        # supplies, and ``None`` means "read the process group", which is 1 when
        # this stack is built undistributed.
        #
        # ``tp_degree`` KEEPS ITS NAME AND ITS MEANING and simply stopped being
        # the divisor (``inc-glm53f-087``): it is the tensor-parallel world size,
        # which is what shards the intermediate dimension.
        self.tp_degree = (
            _resolve_world_size() if world_size is None else int(world_size)
        )
        # THE DIVISOR. ``ep_degree`` is a trailing optional addition in the same
        # shape ``world_size`` already used, so no landed call site moves.
        self.ep_degree = _resolve_ep_degree(ep_degree)
        self.expert_partition = require_uniform_expert_partition(
            self.num_routed_experts, self.ep_degree
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
    # ``rms_norm_eps`` IS THE CONFIG'S, MEASURED NOT ASSUMED. The earlier note
    # here read the key as absent. Both halves of that reading were true when
    # taken -- neither ``Glm5NextTextConfig`` nor the pinned
    # ``fixtures/config.json`` carried an ``rms_norm_eps`` -- but the conclusion
    # drawn from them, that the substrate's own ``eps=1e-6`` was the only value
    # available, is refuted by the checkpoint: its ``text_config`` carries
    # ``rms_norm_eps = 1e-05``. ``inc-glm53f-080`` adds the field and
    # re-transcribes the fixture, so this method resolves the epsilon from the
    # config it already receives and hands THAT number to the seam. ``eps``
    # stays a parameter so a caller can still override it; a caller that passes
    # nothing now gets the checkpoint's value rather than the kernel's.

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        gamma: torch.Tensor,
        text_config: Glm5NextTextConfig,
        eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route ``[B, S, H]`` tokens to the top-``k`` experts with ``noaux_tc``.

        Args:
            hidden_states: ``[B, S, H]`` pre-norm decoder activations.
            gamma: ``[H]`` or ``[1, H]`` router RMSNorm weights.
            text_config: the decoder config the routing hyperparameters live on.
            eps: RMSNorm epsilon. ``None``, the default, resolves it from
                ``text_config.rms_norm_eps`` -- the checkpoint's ``1e-05``.
                Pass a float to override it. See the section note above.

        Returns:
            ``(router_logits [T, E], expert_index [T, k] int32,
            expert_affinities [T, E] float32)``. ``expert_affinities`` is the
            scattered form the downstream MoE consumes: the gate weight at each
            selected expert's column, zero elsewhere.

        The seam is entered with ``correction_bias=self.router_bias``, and that
        parameter is ``mlp.gate.e_score_correction_bias``
        (``weight_loaders_fp8.py:662-665``) -- the ``noaux_tc`` correction bias,
        NOT a router projection bias. The seam's own signature keeps the two
        apart by name, because adding this tensor to the logits instead of to
        the sigmoid scores would compute a different router that no shape check
        could catch.
        """
        # The config's epsilon is the operative one unless a caller overrides
        # it. Resolved here rather than as a signature default, because a
        # signature default hard-wires one number into the code and this one
        # belongs to the checkpoint.
        if eps is None:
            eps = float(text_config.rms_norm_eps)

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
        expert_parallel_rank: int = 0,
    ) -> torch.Tensor:
        """Run this rank's routed experts through the block-quant NKI kernel.

        Args:
            hidden_states: ``[T, H]`` real tokens only -- the kernel's
                padding-token slot is appended here, not by the caller
                (``bwmm_shard_on_I.py:157``).
            expert_affinities: ``[T, E]`` scattered router scores over ALL
                ``n_routed_experts``, which is the form :meth:`route_tokens`
                returns: the gate weight at each selected expert's column and
                zero elsewhere. THIS SITE DOES THE GLOBAL-TO-LOCAL MAPPING, so
                the caller hands the router's output straight through and slices
                nothing. ``inc-glm53f-032``'s landed note above assigns the step
                here in those words. Until repair batch R5 this parameter was
                documented as ``[T, E_local]`` and the extent check enforced
                that, which no caller could satisfy from ``route_tokens`` at any
                expert-parallel degree above 1 -- finding ``B21-027``.
            expert_parallel_rank: which rank's expert slice to select, read
                through :meth:`local_expert_indices`. It is an argument rather
                than an attribute because ``__init__`` is ``inc-glm53f-031``'s
                landed code and this section edits no line above itself. The
                default 0 is the only rank that exists at degree 1.
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
        from vllm_neuron.functional import (
            build_blockwise_mapping,
            get_local_expert_affinities,
        )
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
        # ---- THE DISPATCH STEP. Global router columns -> this rank's slice.  #
        # ``inc-glm53f-032``'s landed note above says in its own words that
        # mapping global expert indices onto this rank's ``expert_partition``
        # slice "belongs to the block-quant call site (``-027``)". Until repair
        # batch R5 this site did the opposite: it REFUSED the global form, so
        # nothing could feed it from ``route_tokens`` above degree 1 and the MoE
        # path could not run on more than one rank. Finding ``B21-027``.
        #
        # The fork's own MoE call site is the precedent, including the guard:
        # ``gpt_oss/model_mxfp4.py:1506-1516`` maps only when the degree is
        # above 1 and passes the affinities through untouched at degree 1.
        routed = int(self.num_routed_experts)
        if tuple(expert_affinities.shape) != (tokens, routed):
            raise Glm5NextBlockQuantRouteError(
                f"expert_affinities must be [T={tokens}, E={routed}] -- the "
                f"GLOBAL router width, which is what route_tokens returns and "
                f"what this site maps onto this rank's {num_experts} experts -- "
                f"got shape {tuple(expert_affinities.shape)}"
            )
        if routed != num_experts:
            local_indices = torch.tensor(
                self.local_expert_indices(int(expert_parallel_rank)),
                dtype=torch.int64,
                device=expert_affinities.device,
            )
            expert_affinities = get_local_expert_affinities(
                expert_affinities, local_indices
            )
        if tuple(expert_affinities.shape) != (tokens, num_experts):
            raise Glm5NextBlockQuantRouteError(
                f"the global-to-local mapping produced "
                f"{tuple(expert_affinities.shape)} and this rank owns "
                f"{num_experts} experts over {tokens} tokens; the partition and "
                f"the gather disagree"
            )
        block = BLOCK_QUANT_SIZE if block_size is None else int(block_size)
        if block <= 0 or block % BLOCK_QUANT_SIZE:
            raise Glm5NextBlockQuantRouteError(
                f"block_size={block} is not a positive multiple of "
                f"BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE} "
                f"(bwmm_shard_on_I.py:667)"
            )

        # ---- The token-block mapping, over the REAL token count. --------- #
        # REUSED (F9), not authored.
        #
        # THE ORDER HERE IS LOAD-BEARING, and until repair batch R5 round 2 it
        # was the wrong way round. The kernel needs a padding-token slot, and
        # this site used to append it to the affinities BEFORE building the
        # mapping. That made the mapping see ``T + 1`` tokens, which is odd for
        # every even real token count, and the mapping's two NKI subkernels
        # need ``chunk_size % 128 == 0`` and ``total_tokens % f_len == 0``
        # (``moe_blockwise.py:520, 536``). Both fail on an odd count, so the
        # mapping fell to ``_build_blockwise_mapping_torch`` on EVERY call at
        # EVERY shape the plan declares -- a silent torch fallback for per-token
        # device work the fork already ships NKI subkernels for
        # (``moe_blockwise.py:10-11``). Finding ``B21-027``, P13.
        #
        # The mapping is now built over the real ``T`` and the padding slot is
        # appended afterwards. The fork's own MoE call site is the precedent:
        # ``gpt_oss/model_mxfp4.py:1518-1536`` never appends a row before the
        # mapping.
        #
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
            expert_affinities=expert_affinities,
            num_local_experts=num_experts,
            num_experts_per_token=int(self.num_experts_per_tok),
            block_size=block,
            moe_group=moe_group,
            tp_degree=tp_degree,
        )

        # ---- The padding-token slot, appended AFTER the mapping. --------- #
        # The kernel reads a ``-1`` token position as the LAST row of
        # ``hidden_states`` (``bwmm_shard_on_I.py:157``), so the hidden tensor
        # grows one zero row and the result is sliced back at the end.
        #
        # The affinities grow the same slot in their FLAT form, because that is
        # what the mapping returns: ``expert_affinities_masked`` is
        # ``[T * E_local, 1]`` and the layout is token-major
        # (``moe_blockwise.py:103-105``, a ``view(-1, 1)`` of ``[T, E]``).
        # Appending ``E_local`` zero entries to the flat tensor is therefore the
        # same tensor as appending one zero ROW before the view -- measured
        # byte-identical, not assumed (``probe-R5r2-pad-order-equivalence.py``,
        # reading ``Q2``). Only this order keeps ``total_tokens`` even.
        pad_hidden = torch.zeros(
            1, hidden, dtype=hidden_states.dtype, device=hidden_states.device
        )
        padded_hidden = torch.cat([hidden_states, pad_hidden], dim=0)
        pad_masked = torch.zeros(
            num_experts,
            1,
            dtype=expert_affinities_masked.dtype,
            device=expert_affinities_masked.device,
        )
        expert_affinities_masked = torch.cat(
            [expert_affinities_masked, pad_masked], dim=0
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
        # The checkpoint's SwiGLU bound, resolved HERE and not at the call.
        # ``inc-glm53f-033`` repair round 2. The reference clamps both MLP
        # projections with this value before their product
        # (``modeling_glm5_next.py:102-104``), so the bound is a model value like
        # any other and belongs on the object the config builds. This is the same
        # construction-time read ``inc-glm53f-080`` made for the decoder's
        # epsilon in ``Glm5NextKDAAttention.__init__``, on the same kind of
        # scalar, one line below the read this method already made for
        # ``n_shared_experts``.
        self.swiglu_limit = float(text_config.swiglu_limit)
        _declare_parameters(
            self, "gate_proj_weight", "up_proj_weight", "down_proj_weight"
        )

    # ── shared-expert path -- D14 owner: ``inc-glm53f-033`` ───────────────
    #
    # SCOPE. This section adds ONE method, and ``forward`` below is ``-013``'s
    # stub and stays byte-identical.
    #
    # ``__init__`` ABOVE IS NO LONGER BYTE-IDENTICAL, and the reason is recorded
    # rather than left to a diff. Round 1 of this increment's repair made the
    # SwiGLU bound a required argument of the two methods below, because
    # ``Glm5NextTextConfig`` did not model ``swiglu_limit`` at all and there was
    # no config value to read. Round 2 lifted the field
    # (``config.py``'s ``swiglu_limit``), so the bound now has a home on the
    # config, and the lead's round-2 bound directs it to be read where the object
    # is built. One line was added to ``__init__`` and nothing in it was changed,
    # so no landed reading of the three declared parameters or of
    # ``num_shared_experts`` can move.
    #
    # The residual add lives on ``Glm5NextMoEBlock`` -- this increment's second
    # section -- because adding the routed and shared halves needs both children
    # and neither child owns the other.
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
    # WHY THE WEIGHTS AND SCALES ARE ARGUMENTS -- ``-027``'s measured shape
    # contract (``increments/evidence-027.md`` §2.3), inherited rather than
    # re-litigated. ``-013``'s ``__init__`` declares the three projections by
    # ``register_parameter(name, None)`` and NO scale parameter. Producing the
    # block scales is the weight loader's step, not this section's, so every
    # WEIGHT-LOADER PRODUCT this site consumes is threaded in at the call --
    # ``-032``'s and ``-027``'s precedent for the quant-config argument.
    #
    # THE SWIGLU BOUND IS NOT ONE OF THOSE, and the distinction is the whole
    # reason it moved in round 2. A block scale is produced per checkpoint load
    # by the weight loader; ``swiglu_limit`` is a scalar the checkpoint declares
    # and the config models, exactly like ``n_shared_experts`` one line up and
    # like ``rms_norm_eps`` on the KDA attention. Model scalars resolve at
    # construction, weight-loader products arrive at the call, and the two rules
    # do not compete.
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

        The SwiGLU the checkpoint stores, **and it clamps both projections before
        the product**: ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))``, where
        ``L`` is ``self.swiglu_limit``, the checkpoint's own bound resolved from
        the config when this object was built. Each of the three projections
        is a separate entry into
        :func:`~vllm_neuron.functional.blockwise_fp8_mm.blockwise_fp8_mm`.

        The clamp is the checkpoint's, not a guard this code invented; the repair
        note at the activation step below cites the reference line for line and
        says what the omission cost.

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

        The SwiGLU bound is NOT an argument. It is ``self.swiglu_limit``,
        resolved from ``text_config.swiglu_limit`` in ``__init__``, so this method
        cannot be called with a bound that is not the one the checkpoint declared
        for this model. Round 1 of the ``B22-M1`` repair did pass it in, because
        the config did not model the key yet; round 2 lifted the field and moved
        the read to construction. **No literal bound appears on this path**, which
        is the half of ``B22-M1-shared-expert-swiglu-clamp-omitted`` that asked
        for the value to come from the checkpoint.

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

        # ---- THE CHECKPOINT'S TWO CLAMPS. `B22-M1`, rounds 1 and 2. -------- #
        # WHAT WAS WRONG. This step read `activated = silu(gate) * up`, with no
        # clamp, so it computed a DIFFERENT FUNCTION from the one the checkpoint
        # stores. The correctness reference this campaign declares -- transformers
        # v5.16.1, `Glm5NextTextMLP.forward`
        # (`models/glm5_next/modeling_glm5_next.py:98-104`, and `:196-197` builds
        # `Glm5NextTextMoE.shared_experts` as that same MLP) -- bounds the gate
        # ABOVE and the up operand on BOTH sides before multiplying:
        #
        #     gate = gate.clamp(min=None, max=self.swiglu_limit)   # `:102`
        #     up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)  # `:103`
        #
        # The two lines below are that transliteration, `min=None` on the gate
        # included: the gate's bound is ONE-SIDED in the reference and copying it
        # as a two-sided clamp would be a second wrong function, not a tidier
        # one.
        #
        # WHAT THE OMISSION COST, measured rather than argued. At this section's
        # own declared fixture the pre-activations are gate `[67.94, 184.89]` and
        # up `[84.30, 166.43]`, so every element of both operands sits outside
        # the checkpoint's `[-10, 10]` box, and the clamped and unclamped results
        # differ by `max_rel_error=1.437580e+02` against a declared `rtol=3e-2`
        # (`increments/probe-R7-clamp-and-config-lift.out`). On the real
        # checkpoint that error entered the residual stream on each of the 42 MoE
        # layers, silently: no shape moves, nothing raises, and the route counter
        # still reads `nki_dispatch=3, torch_fallback=0`.
        #
        # WHERE THE BOUND COMES FROM. `self.swiglu_limit`, resolved from
        # `text_config.swiglu_limit` in `__init__`. Round 1 passed it in as a
        # required argument because `Glm5NextTextConfig` did not model the key at
        # all; round 2 lifted the field (`config.py`'s `swiglu_limit`, defaulting
        # to the checkpoint's own `10.0`) and moved the read to construction, so a
        # production caller resolves nothing and `inc-glm53f-054` will find this
        # method needing only its operands. Reading it off `self` also removes the
        # last way a caller could hand this path a bound the checkpoint never
        # declared. See `increments/evidence-033-r2.md`.
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # The SwiGLU wiring. This is call-site plumbing, not authored numerics:
        # ``silu`` is torch's own, the product is elementwise, and both run in
        # the seam's fp32 return dtype so no precision is thrown away between
        # the projections. This is also the exact reason the declared count
        # cannot be 1 -- ``silu`` is non-linear, so ``down`` cannot fold into
        # either predecessor and the product must be materialised here.
        activated = silu(gate) * up

        # The down projection re-enters the seam, whose declared input dtype is
        # ``bfloat16`` (``blockwise_fp8_mm.py:410``), so the fp32 intermediate is
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
        ep_degree: int | None = None,
    ) -> None:
        super().__init__()
        # ``world_size`` is a trailing optional addition by ``inc-glm53f-031``,
        # threaded to the routed bank only. ``_build_mlp``'s call site is
        # unchanged and stays outside this increment's surface.
        #
        # ``ep_degree`` is the same shape, added by ``inc-glm53f-087``: also
        # trailing, also optional, also threaded to the routed bank only. It is
        # the EXPERT-PARALLEL degree the bank divides by; ``world_size`` stays the
        # tensor-parallel one. ``_build_mlp``'s signature still does not move.
        self.experts = Glm5NextRoutedExperts(
            text_config, world_size=world_size, ep_degree=ep_degree
        )
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

        The SwiGLU bound is not an argument here either, and it is not forwarded.
        The shared expert this block built holds it
        (:meth:`Glm5NextSharedExperts.__init__`), so the bound reaches the clamp
        without passing through this method at all. ``inc-glm53f-033`` repair
        round 2; round 1 forwarded a required argument because the config did not
        model the key yet.

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
# ``self_attn`` path. RE-GROUNDED BY ``inc-glm53f-082``: this header used to
# call ``linear_attn`` the map's path, which it stopped being when
# ``inc-glm53f-078`` measured the family off the published checkpoint index.
# ``linear_attn`` survives below only as ``CACHE_NAME_SUFFIX``, which names
# the KV-cache entry and is not a module path.
# ---------------------------------------------------------------------------


class Glm5NextKDAAttention(nn.Module):
    """Gated-delta linear attention at ``self_attn``.

    Every parameter name here is the landed map's ``KDA_PROJECTIONS`` plus
    ``KDA_BARE_LEAVES``, as ``weight_loaders_fp8.py``'s ``_add_kda_attention``
    emits them: thirteen ``<leaf>_weight`` names and the two bare state tensors
    ``A_log`` and ``dt_bias``, **fifteen in all and not one scale companion**.
    RE-GROUNDED BY ``inc-glm53f-082``: ``inc-glm53f-078`` measured that set off
    the published checkpoint index and retired the six fused ``linear_attn.*``
    names this class used to declare. ``conv1d_bias`` is gone with them because
    the index carries no conv1d bias of any spelling. Cites here name symbols
    rather than line numbers, because the line numbers are what went stale.

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
        # The gate's lower bound is the checkpoint's, read through the same
        # checked accessor as the rest of ``linear_attn_config`` so a missing key
        # raises here instead of reaching the gate seam as a default. The seam
        # requires it and holds no copy of it (``gate_clamp.py:253-259``).
        self.gate_lower_bound = float(
            _linear_attn_field(text_config, "gate_lower_bound")
        )
        # The decoder's RMSNorm epsilon, from the config rather than from a local
        # default, on ``inc-glm53f-080``'s precedent at line 889 of this file.
        self.rms_norm_eps = float(text_config.rms_norm_eps)
        # ``NeuronConfig.kda_state_chunk_size`` is declared a tuning dial with
        # "None = let the layer choose" (``neuron_config.py:185-188``), and
        # ``LayerSpec.chunk_size`` already exists at the pin, so the dial is
        # passed through rather than dropped. None by default.
        self.cache_chunk_size = getattr(
            text_config.neuron_config, "kda_state_chunk_size", None
        )

        # The four recurrent-state values ``get_kv_spec`` reports for this
        # family. ``inc-glm53f-015`` declared the four field names and
        # ``inc-glm53f-016`` certified the runner's translation of them; D14
        # gives the VALUES to ``inc-glm53f-038``, because they come from this
        # layer's own state geometry.
        #
        # BOTH PAIRS ARE DERIVED FROM vLLM's OWN CALCULATORS AND NEITHER IS
        # WRITTEN AS A LITERAL. The conv state's extent ORDER is chosen by the
        # environment (``VLLM_SSM_CONV_STATE_LAYOUT``), so a hand-written pair
        # would be right under one layout and silently transposed under the
        # other -- and a transposition survives a byte reconciliation, which is
        # why ``test_kv_cache_spec.py:31-40`` records the orientation term as
        # load-bearing. The import is function-local because this module holds
        # no vLLM import at module level.
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
            get_conv_state_layout,
            is_conv_state_dim_first,
        )

        conv_state_shape, recurrent_state_shape = (
            MambaStateShapeCalculator.kda_state_shape(
                tp_world_size=world_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                conv_kernel_size=self.short_conv_kernel_size,
            )
        )
        self.kda_conv_state_shape = tuple(conv_state_shape)
        self.kda_recurrent_state_shape = tuple(recurrent_state_shape)
        # ``kda_state_dtype(model_dtype, "auto")`` returns
        # ``(state_dtype, torch.float32)`` and resolves ``"auto"`` to the dtype
        # passed in (``vllm/utils/torch_utils.py:291-295``), so passing
        # ``self.cache_dtype`` keeps ``NeuronConfig.kda_state_dtype``'s override
        # honoured on the conv carrier while the recurrent carrier's float32
        # still comes from the vendor rather than from a local constant.
        (
            self.kda_conv_state_dtype,
            self.kda_recurrent_state_dtype,
        ) = MambaStateDtypeCalculator.kda_state_dtype(self.cache_dtype, "auto")

        # Which axis of ``kda_conv_state_shape`` is the channel axis, read from
        # the same authority that ordered it. The forward slices the conv
        # carrier, so it must not infer the order from the extents.
        # ``is_conv_state_dim_first`` IS the predicate ``_orient_conv_shape``
        # branched on (``vllm/model_executor/layers/mamba/mamba_utils.py:46-48``
        # -- path-qualified because this tree carries a second ``mamba_utils.py``
        # at ``vllm/v1/worker/``), so this is the same reading and
        # not a second inference from the extents. The layout string is kept
        # beside it because a transcript naming ``"SD"`` is checkable and a bare
        # bool is not.
        self.kda_conv_state_layout = get_conv_state_layout()
        self.kda_conv_state_dim_first = is_conv_state_dim_first()

        # The map's fifteen, in the map's own order: ``KDA_PROJECTIONS`` as
        # ``<leaf>_weight``, then ``KDA_BARE_LEAVES`` with no suffix at all.
        _declare_parameters(
            self,
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "b_proj_weight",
            "f_a_proj_weight",
            "f_b_proj_weight",
            "g_a_proj_weight",
            "g_b_proj_weight",
            "q_conv1d_weight",
            "k_conv1d_weight",
            "v_conv1d_weight",
            "o_norm_weight",
            "o_proj_weight",
            "A_log",
            "dt_bias",
        )

    # ----------------------------------------------------------------- #
    # The conv carrier's extent order, handled in one place.             #
    # ----------------------------------------------------------------- #
    def _conv_history(self, conv_state: torch.Tensor) -> torch.Tensor:
        """The carried conv rows as ``[kernel_size - 1, channels]``, float32.

        The stored order is the environment's, so it is converted here once
        rather than assumed at the call site.
        """
        rows = conv_state.t() if self.kda_conv_state_dim_first else conv_state
        return rows.to(torch.float32)

    def _store_conv_history(
        self, conv_state: torch.Tensor, rows: torch.Tensor
    ) -> None:
        """Write ``rows`` (``[kernel_size - 1, channels]``) back in stored order."""
        value = rows.t() if self.kda_conv_state_dim_first else rows
        conv_state.copy_(value.to(conv_state.dtype))

    def _resolve_chunk_size(self, chunk_size: int | None) -> int:
        """The chunk width the chunked seams are entered with.

        ``NeuronConfig.kda_state_chunk_size`` is declared a dial whose ``None``
        means "let the layer choose", so this is where the layer chooses. The
        choice is DERIVED from the two declared bounds it has to satisfy rather
        than picked: the intra-chunk seam needs a power of two in ``[2, 128]``,
        and both chunked seams refuse a chunk-local cumulative gate above
        ``GATE_CUMSUM_ABS_LIMIT``. The gate this layer produces lies in
        ``[gate_lower_bound, 0]``, so one chunk's cumulative gate cannot exceed
        ``chunk * |gate_lower_bound|`` -- and the largest power of two that keeps
        that product inside the limit is the widest chunk that cannot be refused
        for gate range at any input.
        """
        from vllm_neuron.functional.kda.chunked_recurrence import (
            GATE_CUMSUM_ABS_LIMIT,
            MAX_TILE,
        )

        requested = chunk_size if chunk_size is not None else self.cache_chunk_size
        if requested is not None:
            return int(requested)

        bound = abs(self.gate_lower_bound)
        widest = 2
        candidate = 2
        while candidate <= MAX_TILE:
            if bound * candidate <= GATE_CUMSUM_ABS_LIMIT:
                widest = candidate
            candidate *= 2
        return widest

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        is_prefill: bool,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """One KDA layer's gated-delta linear attention over ``[T, hidden]``.

        The five landed seams are entered in this order, and each entry is one
        counted dispatch: the short convolution (``inc-glm53f-034``), the gate
        clamp (``inc-glm53f-084``'s re-authored form), then either the two
        chunked seams (``inc-glm53f-035a`` and ``-035b``) or the single-token
        decode seam (``inc-glm53f-036``). This function composes them; it
        implements none of their arithmetic, which is why its substrate class is
        non-kernel-class.

        Args:
            hidden_states: ``[T, hidden]``. One rank's slice of one step's
                tokens.
            conv_state: this layer's conv carrier from the runner bank, shaped
                as ``get_kv_spec`` reports it. READ for the left context of the
                convolution and WRITTEN IN PLACE with the new tail.
            recurrent_state: this layer's recurrent carrier from the runner
                bank, ``[heads, V, K]``. Index 1 is the VALUE extent and index 2
                the KEY extent -- the orientation ``-035b``'s ``final_state``
                and ``-036``'s ``state`` both use. ``V == K == head_dim`` here,
                so that choice cannot be read off the shape and is stated rather
                than implied. Written in place.
            is_prefill: whether this call is a prefill. It selects the
                recurrence route and nothing else.
            chunk_size: overrides the resolved chunk width, for a test that
                needs to name it.

        Returns:
            ``[T, hidden]`` at the input dtype.

        THE PREFILL ARM ENTERS THE CHUNKED SEAMS WITH A ZERO STATE, BY
        CONSTRUCTION. Neither chunked seam accepts an entering state, so a
        prefill starts the recurrence at zero and ``recurrent_state`` is
        OVERWRITTEN by this call rather than read by it. That is this block's
        declared NON-GOAL held as code, not an oversight: carrying a state into
        a chunked prefill is what chunked prefill, prefix caching or
        multi-token prediction would need, and none of the three exists at the
        pin.

        WHOLE CHUNKS GO THROUGH THE CHUNKED PAIR AND THE REMAINDER WALKS. A
        prefill of ``T = n * chunk + r`` tokens takes one intra-chunk and one
        inter-chunk dispatch for the ``n`` whole chunks together -- their inputs
        carry the chunk axis, so ``n`` chunks cost one dispatch each -- and then
        ``r`` single-token decode dispatches for the remainder. A decode call
        takes no chunked dispatch at any token count.

        The imports are function-local because this module's import block is
        another increment's section (D14).
        """
        from vllm_neuron.functional.kda.chunked_recurrence import (
            kda_inter_chunk,
            kda_intra_chunk,
        )
        from vllm_neuron.functional.kda.decode_state import kda_decode_step
        from vllm_neuron.functional.kda.depthwise_conv1d import depthwise_conv1d
        from vllm_neuron.functional.kda.gate_clamp import (
            MAX_TILE as GATE_MAX_TILE,
            kda_gate_clamp,
        )

        if hidden_states.dim() != 2:
            raise ValueError(
                f"hidden_states must be [tokens, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(hidden_states.shape[0])
        heads = int(self.num_kv_heads_per_rank)
        kdim = int(self.head_dim)
        width = heads * kdim
        state_rows = int(self.short_conv_kernel_size) - 1

        expected_recurrent = (heads, kdim, kdim)
        if tuple(recurrent_state.shape) != expected_recurrent:
            raise ValueError(
                f"recurrent_state {tuple(recurrent_state.shape)} must be "
                f"{expected_recurrent} for this rank's geometry"
            )
        if tuple(conv_state.shape) != tuple(self.kda_conv_state_shape):
            raise ValueError(
                f"conv_state {tuple(conv_state.shape)} must be the shape "
                f"get_kv_spec reports, {tuple(self.kda_conv_state_shape)}"
            )

        x = hidden_states.to(torch.float32)

        def project(weight: torch.Tensor) -> torch.Tensor:
            return x @ weight.to(torch.float32).t()

        q_in = project(self.q_proj_weight)
        k_in = project(self.k_proj_weight)
        v_in = project(self.v_proj_weight)
        raw_beta = project(self.b_proj_weight)
        # Both gates are low-rank: a bottleneck projection, then an expansion
        # back to the head width. The bottleneck width is read from the weights
        # rather than from the config, because the config declares no such field.
        raw_gate = (x @ self.f_a_proj_weight.to(torch.float32).t()) @ (
            self.f_b_proj_weight.to(torch.float32).t()
        )
        out_gate = (x @ self.g_a_proj_weight.to(torch.float32).t()) @ (
            self.g_b_proj_weight.to(torch.float32).t()
        )

        # --- seam 1: the short convolution, ONE dispatch for q, k and v ------
        # The three streams are convolved together as one channel block, which
        # is the same channel extent the state calculator reports
        # (``conv_dim = proj + 2 * proj_k``). Padding is carried by the state
        # rather than by the seam, which refuses non-zero width padding.
        conv_in = torch.cat((q_in, k_in, v_in), dim=-1)
        padded = torch.cat((self._conv_history(conv_state), conv_in), dim=0)
        channels = 3 * width
        img = (
            padded.t().contiguous().reshape(1, channels, 1, state_rows + tokens)
        )
        filt = torch.cat(
            (
                self.q_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
                self.k_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
                self.v_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
            ),
            dim=0,
        ).contiguous()
        conv_out = depthwise_conv1d(img, filt)
        conv_out = conv_out.reshape(channels, tokens).t()
        conv_out = torch.nn.functional.silu(conv_out)
        q_conv, k_conv, v_conv = conv_out.split(width, dim=-1)
        self._store_conv_history(conv_state, padded[padded.shape[0] - state_rows :])

        # --- seam 2: the gate clamp, one dispatch per head per token tile -----
        # The seam takes ONE head per call: it refuses an ``a_log`` holding more
        # than one value, because the decay rate is per head while the bias and
        # the gate are per key channel.
        #
        # It also refuses more than ``GATE_MAX_TILE`` tokens in one call, because
        # both of its axes pass through a transpose that serves that width. So a
        # prompt longer than one tile is handed over in tiles and reassembled
        # here, in the caller. The seam keeps its refusal rather than growing a
        # quiet fallback: tiling is a caller's concern, and a kernel that stretched
        # its own bound would be the torch-level fallback the substrate rule
        # forbids.
        #
        # Tiling is exact here, not approximate. The gate applies a per-channel
        # bias, one scalar decay rate and a sigmoid, with NO reduction along the
        # token axis, so tile boundaries cannot move a value: a tiled call and a
        # whole call agree bit for bit.
        a_log = self.A_log.to(torch.float32).reshape(-1)
        dt_bias = self.dt_bias.to(torch.float32).reshape(-1)

        def clamp_one_head(h: int) -> torch.Tensor:
            """One head's gate over the whole prompt, in tiles the seam accepts."""
            span = slice(h * kdim, (h + 1) * kdim)
            head_bias = dt_bias[span]
            head_decay = a_log[h]
            tiles = [
                kda_gate_clamp(
                    raw_gate[start : start + GATE_MAX_TILE, span].contiguous(),
                    head_decay,
                    bias=head_bias,
                    lower=self.gate_lower_bound,
                )
                for start in range(0, tokens, GATE_MAX_TILE)
            ]
            # A prompt that already fits is returned as the single tile it is, so
            # it still costs exactly one dispatch and no concatenation.
            return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)

        gate_parts = [clamp_one_head(h) for h in range(heads)]
        beta = torch.sigmoid(raw_beta)

        # --- seams 3 to 5: the recurrence, per head --------------------------
        chunk = self._resolve_chunk_size(chunk_size)
        n_chunks = tokens // chunk if is_prefill else 0
        chunked = n_chunks * chunk
        core = torch.empty(tokens, width, dtype=torch.float32)
        for h in range(heads):
            span = slice(h * kdim, (h + 1) * kdim)
            q_h = q_conv[:, span].contiguous()
            k_h = k_conv[:, span].contiguous()
            v_h = v_conv[:, span].contiguous()
            gk_h = gate_parts[h]
            beta_h = beta[:, h].contiguous()

            if is_prefill:
                state = torch.zeros(kdim, kdim, dtype=torch.float32)
            else:
                state = recurrent_state[h].to(torch.float32)

            if chunked:
                shape = (n_chunks, chunk, kdim)
                intra = kda_intra_chunk(
                    q_h[:chunked].reshape(shape).contiguous(),
                    k_h[:chunked].reshape(shape).contiguous(),
                    v_h[:chunked].reshape(shape).contiguous(),
                    beta_h[:chunked].reshape(n_chunks, chunk).contiguous(),
                    gk_h[:chunked].reshape(shape).contiguous(),
                )
                inter = kda_inter_chunk(
                    intra.kg,
                    intra.w,
                    intra.u,
                    gk_h[:chunked].reshape(shape).contiguous(),
                    q_h[:chunked].reshape(shape).contiguous(),
                    intra.aqk,
                )
                core[:chunked, span] = inter.o.reshape(chunked, kdim)
                state = inter.final_state

            for t in range(chunked, tokens):
                step = kda_decode_step(
                    state,
                    q_h[t : t + 1],
                    k_h[t : t + 1],
                    v_h[t : t + 1],
                    beta_h[t].reshape(1, 1),
                    gk_h[t : t + 1],
                )
                core[t, span] = step.o.reshape(-1)
                state = step.state

            recurrent_state[h] = state.to(recurrent_state.dtype)

        # --- gated output norm, then the output projection -------------------
        # ``rmsnorm(core) * sigmoid(out_gate)``, normalised over the head extent
        # because the norm gain is one value per key channel. The reference
        # builds this half as a gated RMSNorm whose activation is sigmoid
        # (``kimi_gdn_linear_attn.py:219``), which is why the raw gate is passed
        # through a sigmoid here and not through a silu.
        shaped = core.reshape(tokens, heads, kdim)
        variance = shaped.pow(2).mean(dim=-1, keepdim=True)
        shaped = shaped * torch.rsqrt(variance + self.rms_norm_eps)
        shaped = shaped * self.o_norm_weight.to(torch.float32).reshape(1, 1, kdim)
        shaped = shaped * torch.sigmoid(
            out_gate.reshape(tokens, heads, kdim)
        )
        attn_out = shaped.reshape(tokens, width) @ (
            self.o_proj_weight.to(torch.float32).t()
        )
        return attn_out.to(hidden_states.dtype)


class Glm5NextKDALayer(nn.Module):
    """Decoder layer on the ``linear_attention`` half of the hybrid stack.

    Section reserved for ``inc-glm53f-038`` (KDA layer + runner state
    plumbing).
    """

    #: Attribute the family's attention module is bound to -- the map's own
    #: module path, which ``weight_loaders_fp8.py``'s ``_add_kda_attention``
    #: builds as ``f"{param_prefix}.self_attn"``. RE-GROUNDED BY
    #: ``inc-glm53f-082``, which moved this off ``linear_attn`` because
    #: ``declared_parameter_names`` builds its paths from ``named_modules()``.
    #: **NOT the same string as ``Glm5NextKDAAttention.CACHE_NAME_SUFFIX``**,
    #: which stays ``linear_attn`` and names a KV-cache entry, not a module.
    ATTENTION_ATTR = "self_attn"

    def __init__(
        self, text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = KDA_LAYER_TYPE
        _declare_parameters(
            self,
            "input_layernorm_weight",
            "post_attention_layernorm_weight",
            # The six mHC weights sit FLAT ON THE LAYER because that is where
            # the map puts them: ``MHC_LEAVES``, emitted for every layer by an
            # unconditional ``_add_mhc`` as ``f"{param_prefix}.{leaf}"`` -- no
            # ``.weight`` leaf, no scale companion, no submodule. A later
            # increment that binds a ``Glm5NextHyperConnection`` instance keeps
            # them here at layer level; moving them under a submodule attribute
            # reddens the map equality, and re-opening the map is the lead's
            # call rather than that increment's.
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
        )
        self.self_attn = Glm5NextKDAAttention(text_config, world_size)
        self.mlp = _build_mlp(text_config, layer_idx)
        self.rms_norm_eps = float(text_config.rms_norm_eps)

    @property
    def attention(self) -> nn.Module:
        return getattr(self, self.ATTENTION_ATTR)

    def _input_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-attention RMSNorm: ``x / sqrt(mean(x**2) + eps) * gain``.

        A method rather than a module-level helper, because this file's
        module-level region is another increment's D14 section. The epsilon is
        the checkpoint's ``rms_norm_eps``, resolved at construction on the same
        ground ``inc-glm53f-080`` states at line 889.
        """
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.rms_norm_eps)
        normed = normed * self.input_layernorm_weight.to(torch.float32)
        return normed.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        is_prefill: bool,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Pre-norm, then the linear-attention half, then the residual add.

        WHAT THIS FORWARD DELIBERATELY DOES NOT DO, AND WHY IT IS NOT A GAP.
        A finished decoder layer also runs its feed-forward half and its
        hyper-connection mixing. Neither is reachable at this milestone and
        neither is this increment's to write:

        * ``self.mlp`` raises whichever branch ``_build_mlp`` chose --
          ``Glm5NextDenseMLP.forward`` and ``Glm5NextMoEBlock.forward`` are both
          still stubs -- and those sections belong to ``inc-glm53f-031`` through
          ``inc-glm53f-033``.
        * the six mHC weights sit flat on this layer but no
          ``Glm5NextHyperConnection`` instance is bound to it yet, and that
          wiring is ``inc-glm53f-030``'s section.

        D14 tells an implementer whose increment would have to touch a class
        outside its own section to raise that rather than widen its surface, so
        this forward stops at the attention half and ``inc-glm53f-054`` joins the
        halves when it writes the 45-layer forward.

        Args and returns are the attention module's, passed through unchanged;
        see :meth:`Glm5NextKDAAttention.forward` for what the two carriers mean.
        """
        residual = hidden_states
        normed = self._input_norm(hidden_states)
        attn_out = self.attention(
            normed,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=is_prefill,
            chunk_size=chunk_size,
        )
        return residual + attn_out


# ---------------------------------------------------------------------------
# The DSA (``deepseek_sparse_attention``) half. D14 owners:
# ``inc-glm53f-039`` (MLA projections) and ``-042`` (MLA decode path) inside
# ``Glm5NextMLAAttention``; ``-051`` for ``Glm5NextDSALayer`` (layer +
# sequence tiling), whose acceptance runs "a 3-layer DSA stack" -- so that
# class is the decoder layer and the MLA module sits at ``self_attn``.
# ---------------------------------------------------------------------------


class Glm5NextDSAIndexer(nn.Module):
    """The DSA sparse indexer at ``self_attn.indexer``.

    LEAF NAMES PROVISIONAL AT THIS INCREMENT, and not any more: the map now
    records ``dsa_indexer`` as GROUNDED (``weight_loaders_fp8.py:84-86``) because
    ``inc-glm53f-078`` read the real shard index. The two dials it was sized from,
    ``index_n_heads`` / ``index_head_dim``, live in ``fixtures/config.json:195-196`` and in
    ``fixtures/hf-config.json:25``/``:21``, never in fork Python -- as the map declares them.
    """

    def __init__(self) -> None:
        super().__init__()
        # The map's seven, in the map's own order: the four scaled projections,
        # then ``k_norm_bias``, then the two bare compress tensors.
        # ``inc-glm53f-082`` replaced the provisional ``wq_weight`` with the
        # ``wq_b_weight`` the checkpoint actually carries.
        _declare_parameters(
            self,
            "wq_b_weight",
            "wk_weight",
            "k_norm_weight",
            "weights_proj_weight",
            "k_norm_bias",
            "index_kpool_compress_ape",
            "index_kpool_compress_gate",
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
        # ``inc-glm53f-039b`` adds these two scalars, because the projections
        # section below needs them and neither existed on the skeleton. The
        # hidden size is the input width of two of the five sites and the output
        # width of a third; the epsilon is the checkpoint's, read through the
        # config on the same ground stated at line 889 rather than defaulted
        # locally.
        self.hidden_size = int(text_config.hidden_size)
        self.rms_norm_eps = float(text_config.rms_norm_eps)

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
        # -- inc-glm53f-085 (WP5 repair) declares the four blockwise-FP8 scale
        #    parameters, in its OWN call so the seven names above stay exactly as
        #    ``inc-glm53f-039b`` and ``inc-glm53f-013`` landed them.
        #
        #    DERIVED FROM THE MAP'S OWN LIST, never a second literal: the four
        #    leaves come from ``DSA_SCALED_PROJECTIONS`` and the suffix from
        #    ``FP8_SCALE_SUFFIX``, both in ``weight_loaders_fp8.py``. That is what
        #    makes the declared-name set and the weight map's parameter set unable
        #    to drift apart -- the bijection ``test_kv_spec.py`` asserts.
        #
        #    ``kv_b_proj`` is absent BECAUSE it is absent from that list: this
        #    checkpoint keeps it in BF16 (``inc-glm53f-078``), so it carries no
        #    scale companion and takes no dequant.
        _declare_parameters(
            self,
            *(f"{leaf}_{FP8_SCALE_SUFFIX}" for leaf in DSA_SCALED_PROJECTIONS),
        )
        self.indexer = Glm5NextDSAIndexer()

    # -- the PROJECTIONS section -- D14 owner: ``inc-glm53f-039b`` (M3) -------
    #
    # WHAT THIS SECTION IS FOR. The five low-rank projections had no substrate
    # member left to call: ``inc-glm53f-072`` measured both candidates and both
    # REFUSE this checkpoint's widths. ``inc-glm53f-039a`` therefore wrote the
    # projection as a NKI kernel, and this section is the call site that reaches
    # it. Every number below is computed from the config; not one is read from a
    # weight's shape, so a mis-shaped checkpoint is caught rather than adopted.
    #
    # WHY THERE IS NO TORCH MATMUL ANYWHERE BELOW. The sibling linear-attention
    # class projects with ``x @ w.t()`` and is right to: its widths are small and
    # no kernel refuses them. Here a torch matmul would be a fallback for work a
    # kernel now does, so this section's acceptance counts occurrences of the
    # torch matmul forms in this class and requires ZERO. That is also why the
    # weights are transposed once, below, rather than per call.
    #
    # WHAT THIS SECTION DOES NOT DO, deliberately:
    #   * it does NOT implement ``forward`` -- D14 gives the forward stubs to
    #     ``inc-glm53f-013`` and then to ``inc-glm53f-054``, and the decode path
    #     that would call these methods to ``inc-glm53f-042``. Both methods below
    #     are entry points those increments call; neither runs on its own.
    #   * it allocates NO rotary parameter and computes no rotary slice, because
    #     ``qk_rope_head_dim`` is 0 on this checkpoint and that 0 is a value
    #     rather than a placeholder. The absence is counted by the acceptance.

    #: Attribute the transposed projection weights are cached on. A plain
    #: attribute and NOT a buffer on purpose: a buffer enters ``state_dict()``,
    #: which would double every projection weight in a saved checkpoint.
    PREPARED_WEIGHTS_ATTR = "_prepared_projection_weights"

    def projection_widths(self) -> tuple[tuple[str, int, int], ...]:
        """The five sites as ``(name, in_features, out_features)``, closed form.

        Each width is derived from the config and named where it comes from, so
        the expectation a test compares against is not a transcription of the
        same literal the code used.

        On this checkpoint the rotary head width is 0, so the query head width
        is the nope width alone and the latent width is the rank alone. Both
        sums are written out anyway: on a config that had a rotary slice the
        bare value would be short by exactly that slice, which is the same
        reason ``_resolve_mla_head_size`` sums rather than takes the rank.
        """
        heads = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        return (
            ("q_a_proj", self.hidden_size, self.q_lora_rank),
            ("q_b_proj", self.q_lora_rank, heads * qk_head_dim),
            (
                "kv_a_proj_with_mqa",
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            (
                "kv_b_proj",
                self.kv_lora_rank,
                heads * (self.qk_nope_head_dim + self.v_head_dim),
            ),
            ("o_proj", heads * self.v_head_dim, self.hidden_size),
        )

    def prepare_projection_weights(self) -> int:
        """Transpose the five projection weights ONCE. Returns how many.

        THE ONE-TIME TRANSPOSE IS THIS INCREMENT'S OWED WORK, and it is owed
        because of a hardware fact on the other side of the seam. A matmul
        contracts the partition axis, so the kernel needs each weight
        contraction-major, ``[in_features, out_features]``. A checkpoint stores
        the torch orientation, ``[out_features, in_features]``. Transposing per
        call would copy up to 64 MB every time; a projection weight never
        changes, so it is transposed here, once, when the weights are loaded.
        The kernel's own record states the same division of labour
        (``../../../artifacts/campaigns/glm-5.3-flash-port/increments/evidence-039a.md``).

        Each weight is checked against the closed-form widths before it is
        transposed. A checkpoint whose projection is the wrong shape fails here,
        with the site named, instead of reaching the kernel as a geometry it
        would accept and quietly compute the wrong thing with.
        """
        prepared: dict[str, torch.Tensor] = {}
        for name, idim, odim in self.projection_widths():
            weight = getattr(self, f"{name}_weight", None)
            if weight is None:
                raise ValueError(
                    f"{name}_weight is declared but not materialised; load the "
                    f"checkpoint before preparing the projection weights"
                )
            if tuple(weight.shape) != (odim, idim):
                raise ValueError(
                    f"{name}_weight is {tuple(weight.shape)}; this config's "
                    f"closed form is [out_features, in_features] = "
                    f"{(odim, idim)}"
                )
            # -- inc-glm53f-085 owns THIS ONE STEP and nothing else here. The
            #    dequant runs in the checkpoint's own [out_features, in_features]
            #    orientation, where the scale grid matches by construction, and
            #    only THEN is the weight transposed by the line below -- so
            #    ``inc-glm53f-039b``'s one-time transpose keeps both its position
            #    and its ground.
            weight = self._dequantised_projection_weight(name, weight)
            # ``.t()`` alone is a view, and the kernel loads from memory, so the
            # copy is forced here -- once -- rather than left for the seam.
            prepared[name] = weight.to(torch.float32).t().contiguous()
        setattr(self, self.PREPARED_WEIGHTS_ATTR, prepared)
        return len(prepared)

    def _dequantised_projection_weight(
        self, name: str, weight: torch.Tensor
    ) -> torch.Tensor:
        """One projection weight in a real dtype, dequantised if it arrived as fp8.

        WHY THIS EXISTS. The published checkpoint stores four of the five MLA
        projections as blockwise-FP8 bytes with a ``weight_scale_inv`` companion
        per 128x128 tile. Before this method the bytes were transposed and used
        as if they were real numbers, so those four projections computed the
        wrong function at exactly the right shapes -- a defect no shape check
        can see. ``inc-glm53f-085`` repairs it.

        THE TEST IS THE DTYPE, NEVER A CONFIG FLAG. A quantisation flag can be
        wrong, absent, or stale, and a weight that is fp8 bytes is fp8 bytes
        whatever a flag says. So the three cases are decided by what actually
        arrived:

        * a real dtype is returned UNCHANGED, exactly as ``inc-glm53f-039b``
          landed it -- this method adds nothing to that path;
        * fp8 bytes with their scale materialised are dequantised;
        * fp8 bytes with NO scale materialised RAISE, naming the site and the
          parameter that is missing.

        THE THIRD CASE IS THE POINT. Silently treating unscaled fp8 bytes as
        numbers is the defect being repaired, so the one thing this method must
        never do is continue quietly when the scale is absent. A refusal that
        names the site is also distinguishable from a skipped test, which is
        what makes it checkable.

        The dequant itself is the fork's own ``dequantise_blockwise``; this
        method chooses no arithmetic of its own. It returns fp32, so the
        caller's following ``.to(torch.float32)`` is already satisfied and
        stays a no-op rather than a second conversion.
        """
        if not _is_fp8_dtype(weight.dtype):
            return weight
        scale_name = f"{name}_{FP8_SCALE_SUFFIX}"
        scale = getattr(self, scale_name, None)
        if scale is None:
            raise ValueError(
                f"{name}_weight arrived as {weight.dtype} blockwise-FP8 bytes "
                f"but {scale_name} is not materialised, so the bytes cannot be "
                f"dequantised; load the checkpoint's {FP8_SCALE_SUFFIX} "
                f"companion for {name} before preparing the projection weights"
            )
        return dequantise_blockwise(weight, scale)

    def _prepared_weight(self, name: str) -> torch.Tensor:
        """One prepared weight, or a refusal naming what was not done.

        Refusing is what makes "never per call" checkable. If this fell back to
        transposing on demand, the per-call copy the section exists to avoid
        would come back silently and nothing would report it.
        """
        prepared = getattr(self, self.PREPARED_WEIGHTS_ATTR, None)
        if not prepared:
            raise ValueError(
                "prepare_projection_weights() has not run; the projection "
                "weights are transposed once at load time, never per call"
            )
        return prepared[name]

    def _latent_norm(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """RMSNorm on a low-rank latent: ``x / sqrt(mean(x**2) + eps) * gain``.

        A method rather than a module-level helper, because this file's
        module-level region is another increment's D14 section -- the same
        reason the sibling layer's own norm gives.

        The two latent norms sit BETWEEN the projections, so they belong to this
        section and to no other. They are applied here rather than left for a
        later increment: a projection chain that emitted un-normalised latents
        would be numerically wrong, and correcting it later would put a second
        writer inside this section.
        """
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.rms_norm_eps)
        return normed * gain.to(torch.float32)

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query, no-rotary key and value from hidden states. FOUR dispatches.

        ``hidden_states`` is ``[tokens, hidden_size]``. The three returns are
        ``[tokens, heads, qk_nope_head_dim]``, ``[tokens, heads,
        qk_nope_head_dim]`` and ``[tokens, heads, v_head_dim]``.

        The chain is the low-rank one this checkpoint declares: compress to a
        rank, normalise the latent, expand to the head width. The key and value
        come out of ONE expansion and are split, which is why this is four
        dispatches and not five.
        """
        from vllm_neuron.functional.attention.mla_projections import mla_projection

        widths = {name: (idim, odim) for name, idim, odim in self.projection_widths()}
        x = hidden_states.to(torch.float32)
        if x.ndim != 2 or int(x.shape[1]) != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [tokens, {self.hidden_size}]; got "
                f"{tuple(hidden_states.shape)}"
            )
        tokens = int(x.shape[0])
        heads = self.num_attention_heads

        q_latent = mla_projection(x, self._prepared_weight("q_a_proj"))
        q_latent = self._latent_norm(q_latent, self.q_a_layernorm_weight)
        query = mla_projection(q_latent, self._prepared_weight("q_b_proj"))
        query = query.reshape(tokens, heads, widths["q_b_proj"][1] // heads)

        kv_latent = mla_projection(x, self._prepared_weight("kv_a_proj_with_mqa"))
        kv_latent = self._latent_norm(kv_latent, self.kv_a_layernorm_weight)
        key_value = mla_projection(kv_latent, self._prepared_weight("kv_b_proj"))
        key_value = key_value.reshape(
            tokens, heads, self.qk_nope_head_dim + self.v_head_dim
        )
        key_nope = key_value[..., : self.qk_nope_head_dim]
        value = key_value[..., self.qk_nope_head_dim :]

        out_dtype = hidden_states.dtype
        return (
            query.to(out_dtype),
            key_nope.contiguous().to(out_dtype),
            value.contiguous().to(out_dtype),
        )

    def project_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        """The output projection. ONE dispatch.

        ``attn_out`` is ``[tokens, heads, v_head_dim]`` or the same flattened to
        ``[tokens, heads * v_head_dim]``; the return is ``[tokens,
        hidden_size]``. Both input forms are accepted because the decode path
        that calls this is another increment's and its layout is its own choice,
        not something this section should dictate.
        """
        from vllm_neuron.functional.attention.mla_projections import mla_projection

        expected_width = self.num_attention_heads * self.v_head_dim
        x = attn_out.to(torch.float32)
        if x.ndim == 3:
            x = x.reshape(int(x.shape[0]), -1)
        if x.ndim != 2 or int(x.shape[1]) != expected_width:
            raise ValueError(
                f"attn_out must be [tokens, {self.num_attention_heads}, "
                f"{self.v_head_dim}] or [tokens, {expected_width}]; got "
                f"{tuple(attn_out.shape)}"
            )
        projected = mla_projection(x.contiguous(), self._prepared_weight("o_proj"))
        return projected.to(attn_out.dtype)

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
    #: module path, which ``weight_loaders_fp8.py``'s ``_add_dsa_attention``
    #: builds as ``f"{param_prefix}.self_attn"``. RE-GROUNDED BY
    #: ``inc-glm53f-082``: the old cite pointed at a line that is now the
    #: post-attention-layernorm mapping. The value itself does not move.
    ATTENTION_ATTR = "self_attn"

    def __init__(
        self, text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = DSA_LAYER_TYPE
        _declare_parameters(
            self,
            "input_layernorm_weight",
            "post_attention_layernorm_weight",
            # The same six mHC weights, flat on the layer for the same reason as
            # ``Glm5NextKDALayer``: ``_add_mhc`` runs for EVERY layer of the
            # stack, not only the linear-attention half.
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
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
    (``config.py:36-40``), so a substring test would silently mis-partition
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
    **pinned by landed code**: ``factory.py:340`` already reads ``from
    .model_fp8 import Glm5NextForConditionalGeneration as Model`` and
    ``factory.py:342`` calls
    ``Model.from_configs(hf_config, text_neuron_config=..., vision_neuron_config=...)``.
    The name is duplicated with ``factory.py:268`` on purpose -- that is the
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
        # (``weight_loaders_fp8.py:382-383``).
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
        """Build from an HF config, the signature ``factory.py:342`` calls."""
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

        THE FOUR RECURRENT-STATE FIELDS ARE FILLED BY THE SAME UNIFORM READ.
        A linear-attention layer holds a short-convolution state and a
        recurrent state instead of a key/value history, and reports the two on
        ``LayerSpec``'s four ``kda_*`` fields (``inc-glm53f-015``'s declared
        interface). ``Glm5NextKDAAttention`` derives all four from vLLM's own
        state calculators at its own rank geometry; ``Glm5NextMLAAttention``
        declares none of them. So the read is a DEFAULTING one -- a family that
        carries no recurrent state reports ``None`` on all four by carrying no
        attribute, rather than by this method testing which family it is
        looking at. That keeps the one loop family-blind, which is the property
        ``inc-glm53f-013`` built it for, and it is why the runner recognises the
        two halves BY THE FIELDS THEY CARRY (``neuron_model_runner.py``
        ``:8720-8726``) rather than by a layer name.

        All four move together or not at all. The runner refuses a layer that
        declares part of the geometry (``neuron_model_runner.py:8727-8733``),
        because the conv and recurrent carriers are paired positionally, so a
        partial set would shorten the reported page. One ``getattr`` per field
        against one attribute-carrying class satisfies that by construction.
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
                    kda_conv_state_shape=getattr(
                        attention, "kda_conv_state_shape", None
                    ),
                    kda_recurrent_state_shape=getattr(
                        attention, "kda_recurrent_state_shape", None
                    ),
                    kda_conv_state_dtype=getattr(
                        attention, "kda_conv_state_dtype", None
                    ),
                    kda_recurrent_state_dtype=getattr(
                        attention, "kda_recurrent_state_dtype", None
                    ),
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
