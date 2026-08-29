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
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

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
    if (
        "torch_dtype" not in filtered
        and "dtype" in config_dict
        and "torch_dtype" in field_names
    ):
        filtered["torch_dtype"] = config_dict["dtype"]
    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

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
    # Only the two this module needs. The quantization SPEC (scheme
    # resolution, per-module policy) is deliberately NOT modelled here.
    quant_method: str | None = "fp8"
    activation_scheme: str | None = "dynamic"
    weight_block_size: list[int] | None = field(default_factory=lambda: [128, 128])

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
        )
