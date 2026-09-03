# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash (Glm5Next) Configuration
=====================================

Nested multimodal config: the top-level ``Glm5NextConfig`` composes a
``Glm5NextTextConfig`` and a ``Glm5NextVisionConfig``, mirroring the
HuggingFace ``config.json`` which carries ``text_config`` and
``vision_config`` sub-objects plus a top-level ``quantization_config``.

<-- MODEL-SPECIFIC: the field set below is GLM-5.3-Flash's. Two things are
unusual enough to call out, because both drive downstream shape decisions:

1. **The attention stack is HYBRID, interleaved 3:1.** ``text_config``
   carries a 45-entry ``layer_types`` schedule over two families:
   ``linear_attention`` (KDA, gated-delta style) and
   ``deepseek_sparse_attention`` (DSA, which is where MLA lives). The split
   is read off the schedule, never assumed -- see ``attention_layer_split``.
2. **Quantization is BLOCKWISE FP8**, not per-tensor: ``weight_block_size``
   is a 2-D block shape, and ``activation_scheme`` is dynamic. Only the two
   fields this module needs are lifted here; the quantization *spec* (scheme
   resolution, per-module policy) is a separate concern and a separate file.
"""

import json
import logging
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

logger = logging.getLogger(__name__)

# Layer-family names exactly as they appear in text_config.layer_types.
# Compared by EQUALITY, never by substring: "attention" is a substring of both,
# so a substring test would silently mis-partition the stack.
KDA_LAYER_TYPE = "linear_attention"
DSA_LAYER_TYPE = "deepseek_sparse_attention"

# The checkpoint interleaves 3 KDA layers then 1 DSA layer, repeating, so a
# layer is DSA iff its index mod 4 == 3. Used only to supply a default when
# a config omits layer_types; a present schedule always wins.
_DSA_PERIOD = 4
_DSA_PHASE = 3


class Glm5NextExpertConfigError(ValueError):
    """An expert-count field the routed-MoE stack cannot be built from.

    ``inc-glm53f-031``. Named rather than a bare ``ValueError`` for the same
    reason ``RaggedExpertPartitionError`` is: the expert-count path has two
    distinct failure classes -- a count that is out of range (here) and a count
    that is in range but does not shard uniformly (``factory.py``) -- and a
    caller that must tell them apart cannot do so from ``ValueError``.
    """


def default_layer_types(num_hidden_layers: int) -> list[str]:
    """The 3:1 KDA/DSA schedule, generated from the interleave rule."""
    return [
        DSA_LAYER_TYPE if i % _DSA_PERIOD == _DSA_PHASE else KDA_LAYER_TYPE
        for i in range(num_hidden_layers)
    ]


def _from_hf_sub_config(cls, hf_sub_config, neuron_config=None):
    """Build a sub-config from an HF config sub-object.

    Filters the HF dict to fields declared on the target dataclass, coerces
    the dtype string, and attaches the neuron_config. Same shape as the
    sibling arch packages use, so the two read alike.

    EVERY DROPPED KEY IS NAMED AT ``WARNING`` (``inc-glm53f-080``). The filter
    below keeps only declared fields, and it used to drop the rest without a
    word: the real checkpoint's ``text_config`` carries 58 keys, so every key
    this dataclass family does not declare reached nothing and a reader had no
    way to learn which. One of them was the model's own ``rms_norm_eps``. The
    log is the repair, so the next missing field is found by reading a warning
    rather than by counting fields by hand.

    THE COUNT IS NOT WRITTEN HERE ANY MORE, and that is deliberate. This
    docstring used to say the family "models 30 of them, so 26 real keys reached
    nothing", which cannot be right about both halves at once: 58 minus 30 is
    28. The measured decomposition at ``inc-glm53f-033`` repair round 2 was 33
    declared fields, 31 of them keys the vendor config also carries, plus the one
    key the ``dtype`` remap consumes, so the log read 32 modelled and 26 dropped.
    That round then added ``swiglu_limit`` and the log reads 33 modelled and 25
    dropped. A count in this prose is a second place to keep the same number, so
    the number now lives only where it is measured -- the log itself, and
    ``test_config.py``'s conjunct (c), which derives it from the vendor config
    and this dataclass rather than restating it.
    """
    if isinstance(hf_sub_config, PretrainedConfig):
        config_dict = hf_sub_config.to_dict()
    elif isinstance(hf_sub_config, dict):
        config_dict = hf_sub_config
    else:
        raise TypeError(f"Unsupported config type: {type(hf_sub_config)}")

    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in field_names}

    # HF config.json uses "dtype"; the dataclass uses "torch_dtype".
    # ``remapped`` records the HF key this branch CONSUMES, so the drop log
    # below cannot report a key the adapter actually reads. It is filled only
    # when the branch fires: an HF dict carrying both names leaves "dtype"
    # genuinely unused, and then it is a dropped key like any other.
    remapped: set[str] = set()
    if (
        "torch_dtype" not in filtered
        and "dtype" in config_dict
        and "torch_dtype" in field_names
    ):
        filtered["torch_dtype"] = config_dict["dtype"]
        remapped.add("dtype")
    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

    dropped = sorted(set(config_dict) - field_names - remapped)
    if dropped:
        # One record, every name in it: a per-key record would put one line per
        # dropped key in the log for one config -- 25 of them for this
        # checkpoint's text config -- and get filtered out as noise.
        logger.warning(
            "%s models %d of the %d keys in this HF config and DROPS the "
            "other %d: %s",
            cls.__name__,
            len(config_dict) - len(dropped),
            len(config_dict),
            len(dropped),
            ", ".join(dropped),
        )

    if neuron_config is not None:
        filtered["neuron_config"] = neuron_config

    return cls(**filtered)


@dataclass
class Glm5NextTextConfig:
    """Text decoder config extracted from ``hf_config.text_config``.

    Defaults are the GLM-5.3-Flash checkpoint's.
    """

    # -- Core shape ---------------------------------------------------------
    num_hidden_layers: int = 45
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    vocab_size: int = 154880
    max_position_embeddings: int = 1048576
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    first_k_dense_replace: int = 3

    # -- Hybrid attention schedule (MODEL-SPECIFIC) -------------------------
    # 45 entries, one per layer, each KDA_LAYER_TYPE or DSA_LAYER_TYPE.
    layer_types: list[str] | None = None
    linear_attn_config: dict = field(
        default_factory=lambda: {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        }
    )

    # -- MLA, on the DSA half ----------------------------------------------
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 256
    # 0 is a VALUE here, not a placeholder: this checkpoint is NoPE on the
    # MLA half (mla_use_nope), so there is no rotary head slice at all.
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    mla_use_nope: bool = True

    # -- MoE ---------------------------------------------------------------
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5

    # -- The SwiGLU bound the checkpoint clamps with ------------------------
    # ``swiglu_limit`` is the bound the reference applies to BOTH MLP
    # projections before their product: it clamps ``gate`` from above and
    # ``up`` on both sides, and only then multiplies
    # (``modeling_glm5_next.py:102-104``). The checkpoint declares it in
    # ``text_config`` AND in ``vision_config``, both at ``10.0``, and
    # ``Glm5NextVisionConfig`` below has carried the field since
    # ``inc-glm53f-032``; the text config did not, so the counting pass dropped
    # the key and the shared expert had no checkpoint value to clamp with. That
    # is the second surface of ``B22-M1-shared-expert-swiglu-clamp-omitted``,
    # lifted by ``inc-glm53f-033`` repair round 2.
    #
    # THE DEFAULT IS THE CHECKPOINT'S OWN, for the same reason
    # ``rms_norm_eps``'s is: a config that omits the key then resolves to the
    # target's number instead of to whatever a compute path happens to pick.
    # The literal lives here, in the one place that models the checkpoint's
    # declared values, and NOT on the compute path -- the shared expert reads
    # this field and carries no bound of its own.
    swiglu_limit: float = 10.0

    # -- Normalisation epsilons, TWO of them and not one -------------------
    # ``rms_norm_eps`` is the decoder's RMSNorm epsilon and ``hc_eps`` is the
    # mHC epsilon. The checkpoint sets them to DIFFERENT values (1e-05 and
    # 1e-06), so collapsing them onto one field would change what every
    # RMSNorm computes. The default here is the checkpoint's own 1e-05, so a
    # config that omits the key resolves to the target's number rather than to
    # whatever a kernel happens to default to (``inc-glm53f-080``).
    rms_norm_eps: float = 1e-05

    # -- Multi-hyper-connections (mHC) -------------------------------------
    mhc: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-06

    # -- Framework config (not model-specific) ------------------------------
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = default_layer_types(self.num_hidden_layers)
        self._validate_layer_types()
        self._validate_expert_counts()

    def _validate_expert_counts(self) -> None:
        """The expert-count fields the MoE-288 stack is built from, PER FIELD.

        ``inc-glm53f-031``. Loud rather than defaulted, for the same reason
        :meth:`_validate_layer_types` is: an out-of-range count produces a
        routed-expert bank whose shard arithmetic is wrong in a way no
        downstream shape assertion could attribute back to here.

        **Measured at the unmodified parent ``031535b`` before this was
        authored: every one of ``n_routed_experts=0``, ``n_routed_experts=-8``,
        ``num_experts_per_tok=0``, ``num_experts_per_tok=999`` and
        ``n_shared_experts=-1`` was ACCEPTED silently**, so this validator is
        the whole per-field path.

        WHERE THE BOUNDARY IS, AND WHY IT IS HERE. This method validates
        **well-formedness** -- each field inside its own range -- which is
        exactly what :meth:`_validate_layer_types` already does for the
        schedule (a length, and family membership). Two **cross-field**
        questions are deliberately NOT asked here, and each has a named home on
        the sharding path instead:

        * ``num_experts_per_tok <= n_routed_experts`` -- a **router**
          precondition, enforced by ``factory.py``'s
          :func:`~vllm_neuron.model.glm5_next.factory.require_routable_expert_counts`.
        * the uniform-partition question -- depends on a tensor-parallel degree
          this dataclass does not carry, enforced by ``factory.py``'s
          :func:`~vllm_neuron.model.glm5_next.factory.require_uniform_expert_partition`.

        **The boundary is not a convenience, and it was measured.** Asking the
        router question at construction time rejected ``inc-glm53f-011``'s
        landed ``mini_config`` fixture, which declares a 4-expert bank while
        inheriting the checkpoint's top-8 default -- a structural key-mapping
        fixture that never routes a token. That fixture's latent incoherence is
        recorded and routed to the lead; it is NOT repaired here, because
        ``test_weight_loaders.py`` is outside this increment's declared surface.
        A config dataclass that refuses a structural fixture has moved a router
        precondition to the wrong layer, and the refusal is kept -- just at the
        layer that routes.
        """
        if int(self.n_routed_experts) < 1:
            raise Glm5NextExpertConfigError(
                f"n_routed_experts must be >= 1, got {self.n_routed_experts}; "
                "a sparse MLP with no routed expert has nothing to route to"
            )
        if int(self.num_experts_per_tok) < 1:
            raise Glm5NextExpertConfigError(
                f"num_experts_per_tok must be >= 1, got "
                f"{self.num_experts_per_tok}; a top-k router with k < 1 selects "
                "no expert for any token"
            )
        if int(self.n_shared_experts) < 0:
            raise Glm5NextExpertConfigError(
                f"n_shared_experts must be >= 0, got {self.n_shared_experts}; "
                "0 is the declared no-shared-expert value, negative is not a value"
            )

    def _validate_layer_types(self) -> None:
        """The schedule must be a length-correct, EXHAUSTIVE two-family partition.

        Both halves are load-bearing. A wrong length would make the split
        disagree with num_hidden_layers; an unrecognised family name would be
        dropped by a counting pass and inflate the other family's count
        silently, which is exactly the failure the split must not have.
        """
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )
        unknown = sorted(
            {t for t in self.layer_types if t not in (KDA_LAYER_TYPE, DSA_LAYER_TYPE)}
        )
        if unknown:
            raise ValueError(
                f"layer_types carries unrecognised attention families {unknown}; "
                f"expected only {KDA_LAYER_TYPE!r} and {DSA_LAYER_TYPE!r}"
            )

    @property
    def kda_layer_indices(self) -> list[int]:
        """Indices of the linear-attention (KDA) layers."""
        return [i for i, t in enumerate(self.layer_types) if t == KDA_LAYER_TYPE]

    @property
    def dsa_layer_indices(self) -> list[int]:
        """Indices of the sparse-attention (DSA/MLA) layers."""
        return [i for i, t in enumerate(self.layer_types) if t == DSA_LAYER_TYPE]

    @property
    def attention_layer_split(self) -> tuple[int, int]:
        """``(num_kda_layers, num_dsa_layers)``, counted off ``layer_types``.

        Returned as ONE pair rather than two independent counts because the
        two are a single fact about the stack: the partition is exhaustive
        (``_validate_layer_types``), so the pair always sums to
        ``num_hidden_layers`` and neither half is meaningful alone.
        """
        return (len(self.kda_layer_indices), len(self.dsa_layer_indices))

    @classmethod
    def from_hf_config(cls, hf_text_config, neuron_config: NeuronConfig = None):
        """Build the text config from the ``text_config`` sub-object."""
        return _from_hf_sub_config(cls, hf_text_config, neuron_config)


@dataclass
class Glm5NextVisionConfig:
    """Vision encoder config extracted from ``hf_config.vision_config``."""

    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    image_size: int = 448
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    projection_intermediate_size: int = 10240
    attention_bias: bool = True
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0

    neuron_config: VisionNeuronConfig | None = None

    @classmethod
    def from_hf_config(cls, hf_vision_config, neuron_config: VisionNeuronConfig = None):
        """Build the vision config from the ``vision_config`` sub-object."""
        return _from_hf_sub_config(cls, hf_vision_config, neuron_config)


@dataclass
class Glm5NextConfig:
    """Top-level config composing the text and vision sub-configs.

    Each sub-config carries its own NeuronConfig because the text decoder and
    the vision encoder may need different parallelism / compilation settings.
    """

    text_config: Glm5NextTextConfig | None = None
    vision_config: Glm5NextVisionConfig | None = None

    # -- Top-level fields from HF config -----------------------------------
    image_token_id: int = 154854
    tie_word_embeddings: bool = False

    # -- Blockwise-FP8 fields lifted from quantization_config ---------------
    # ALL FIVE of them since ``inc-glm53f-079``. The quantization SPEC (scheme
    # resolution, per-module policy) is still deliberately NOT modelled here;
    # this dataclass only carries what the checkpoint declares.
    quant_method: str | None = "fp8"
    activation_scheme: str | None = "dynamic"
    weight_block_size: list[int] | None = field(default_factory=lambda: [128, 128])

    #: Substring-match list of module names the checkpoint keeps in BF16, taken
    #: verbatim off ``quantization_config.modules_to_not_convert``. The real
    #: checkpoint ships 1,509 entries and this adapter read NONE of them before
    #: ``inc-glm53f-079``, so every BF16 family was treated as block-FP8. The
    #: matching rule is the fork's own and is not restated here
    #: (``neuron_config.py``'s ``modules_to_not_convert``, consumed by
    #: ``qwen3_vl/model_mxfp8.py``'s ``_keep_bf16``): a module keeps BF16 when
    #: any entry is a substring of its qualified name.
    modules_to_not_convert: list[str] | None = None

    #: The FP8 byte format the checkpoint declares (``"e4m3"``). Lifted so a
    #: consumer can check it rather than assume it; nothing in this campaign
    #: branches on a second format yet, and a surprise here is worth seeing.
    fmt: str | None = None

    @property
    def is_block_quantized(self) -> bool:
        """True when the checkpoint carries a 2-D weight block shape."""
        return bool(self.weight_block_size) and len(self.weight_block_size) == 2

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        text_neuron_config: NeuronConfig = None,
        vision_neuron_config: VisionNeuronConfig = None,
    ):
        """Decompose an HF config and build the nested vllm-neuron config.

        Args:
            hf_config: a HuggingFace PretrainedConfig, an already-loaded dict,
                or a path to a ``config.json``.
            text_neuron_config: NeuronConfig for the text decoder.
            vision_neuron_config: NeuronConfig for the vision encoder.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                top_level = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            top_level = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            top_level = hf_config
        else:
            raise TypeError(f"Unsupported hf_config type: {type(hf_config)}")

        text_config = Glm5NextTextConfig.from_hf_config(
            top_level["text_config"], text_neuron_config
        )
        vision_config = Glm5NextVisionConfig.from_hf_config(
            top_level["vision_config"], vision_neuron_config
        )

        # tie_word_embeddings lives at the top level in HF config.json;
        # propagate it so weight loading can find it on the text config.
        tie_word_embeddings = top_level.get("tie_word_embeddings", False)
        text_config.tie_word_embeddings = tie_word_embeddings

        quant_cfg = top_level.get("quantization_config") or {}

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=top_level.get("image_token_id", 154854),
            tie_word_embeddings=tie_word_embeddings,
            quant_method=quant_cfg.get("quant_method"),
            activation_scheme=quant_cfg.get("activation_scheme"),
            weight_block_size=quant_cfg.get("weight_block_size"),
            # inc-glm53f-079: the two fields this adapter used to drop on the
            # floor. The skip list is the checkpoint's own quantisation policy,
            # so leaving it unread meant the policy could not be honoured.
            modules_to_not_convert=quant_cfg.get("modules_to_not_convert"),
            fmt=quant_cfg.get("fmt"),
        )
