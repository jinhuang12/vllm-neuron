# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 language model configuration."""

import json
import math
import os
from dataclasses import dataclass, fields

import torch
from safetensors import safe_open
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

KDA_LAYER_TYPE = "kda"
MLA_LAYER_TYPE = "mla"

# Name prefix of the language model's weights in the K3 checkpoint.
CHECKPOINT_PREFIX = "language_model."


def _as_dict(config: PretrainedConfig | dict | str | os.PathLike) -> dict:
    """Return a config as a dict, from an HF config, a dict or a JSON path."""
    if isinstance(config, (str, os.PathLike)):
        with open(config) as f:
            return json.load(f)
    if isinstance(config, PretrainedConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    raise TypeError(f"Unsupported config type: {type(config)}")


def _heads_per_rank(num_heads: int, tp_size: int, kind: str) -> int:
    """Return num_heads // tp_size, or raise when the heads do not divide."""
    if tp_size < 1 or num_heads % tp_size:
        raise ValueError(
            f"{num_heads} {kind} heads do not divide over tensor-parallel "
            f"size {tp_size}"
        )
    return num_heads // tp_size


@dataclass
class KimiK3TextConfig:
    """Language model config read from ``hf_config.text_config``.

    Defaults are those of the checkpoint's ``KimiLinearConfig``, so a key the
    checkpoint omits resolves as the reference model resolves it.
    """

    vocab_size: int = 163840
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    activation_situ_beta: float | None = None
    activation_situ_linear_beta: float | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Kimi Delta Attention; the layer lists are 1-based, as the checkpoint
    # stores them.
    linear_attn_config: dict | None = None

    # Multi-head latent attention.
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    mla_use_nope: bool = False
    mla_use_output_gate: bool = False

    # Attention residuals.
    attn_res_block_size: int | None = None

    # Mixture of experts with a latent expert width.
    num_experts: int | None = None
    num_experts_per_token: int | None = None
    num_shared_experts: int = 0
    moe_intermediate_size: int | None = None
    routed_expert_hidden_size: int | None = None
    latent_moe_use_norm: bool = False
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"

    quantization_config: dict | None = None
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self._validate_layer_lists()

    def _validate_layer_lists(self) -> None:
        """Require the KDA and MLA lists to split layers 1..N exactly once."""
        if self.linear_attn_config is None:
            raise ValueError("Kimi K3 requires text_config.linear_attn_config")
        kda = list(self.linear_attn_config["kda_layers"])
        mla = list(self.linear_attn_config["full_attn_layers"])
        if sorted(kda + mla) != list(range(1, self.num_hidden_layers + 1)):
            raise ValueError(
                "kda_layers and full_attn_layers must split layers "
                f"1..{self.num_hidden_layers} with no overlap; got "
                f"{len(kda)} KDA and {len(mla)} MLA entries"
            )

    @property
    def layer_types(self) -> list[str]:
        """Return the attention type of each layer, by 0-based index."""
        kda = set(self.linear_attn_config["kda_layers"])
        return [
            KDA_LAYER_TYPE if i + 1 in kda else MLA_LAYER_TYPE
            for i in range(self.num_hidden_layers)
        ]

    @property
    def kda_layer_indices(self) -> list[int]:
        """Return the 0-based indices of the KDA layers."""
        return [i for i, t in enumerate(self.layer_types) if t == KDA_LAYER_TYPE]

    @property
    def mla_layer_indices(self) -> list[int]:
        """Return the 0-based indices of the MLA layers."""
        return [i for i, t in enumerate(self.layer_types) if t == MLA_LAYER_TYPE]

    @property
    def kda_num_heads(self) -> int:
        """Return the KDA head count of the whole model."""
        return int(self.linear_attn_config["num_heads"])

    def kda_heads_per_rank(self, tp_size: int) -> int:
        """Return the KDA heads each tensor-parallel rank holds."""
        return _heads_per_rank(self.kda_num_heads, tp_size, "KDA")

    def mla_heads_per_rank(self, tp_size: int) -> int:
        """Return the MLA heads each tensor-parallel rank holds."""
        return _heads_per_rank(self.num_attention_heads, tp_size, "MLA")

    @classmethod
    def from_hf_config(
        cls,
        hf_text_config: PretrainedConfig | dict,
        neuron_config: NeuronConfig | None = None,
    ) -> "KimiK3TextConfig":
        """Build the text config from the HF ``text_config`` sub-object."""
        config_dict = _as_dict(hf_text_config)
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in config_dict.items() if k in names}
        # HF config.json names the dtype "dtype"; older configs "torch_dtype".
        dtype = kwargs.get("torch_dtype", config_dict.get("dtype"))
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        kwargs["neuron_config"] = neuron_config
        return cls(**kwargs)


@dataclass
class KimiK3Config:
    """Top-level Kimi K3 config; only the language model is built."""

    text_config: KimiK3TextConfig
    media_placeholder_token_id: int | None = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str | os.PathLike,
        neuron_config: NeuronConfig | None = None,
    ) -> "KimiK3Config":
        """Build the config from the HF top-level config and a NeuronConfig."""
        top = _as_dict(hf_config)
        text_config = KimiK3TextConfig.from_hf_config(
            top["text_config"], neuron_config
        )
        tie = bool(top.get("tie_word_embeddings", text_config.tie_word_embeddings))
        text_config.tie_word_embeddings = tie
        return cls(
            text_config=text_config,
            media_placeholder_token_id=top.get("media_placeholder_token_id"),
            tie_word_embeddings=tie,
        )


def stored_a_log_length(
    checkpoint_dir: str | os.PathLike, text_config: KimiK3TextConfig
) -> int:
    """Return the A_log length the checkpoint stores, one value for all KDA layers.

    The checkpoint stores more entries than there are KDA heads; the extra
    entries are unused. A stored shape of ``(1, 1, H, 1)`` counts as length H.
    """
    with open(os.path.join(checkpoint_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for layer in text_config.kda_layer_indices:
        name = f"{CHECKPOINT_PREFIX}model.layers.{layer}.self_attn.A_log"
        by_shard.setdefault(weight_map[name], []).append(name)

    lengths = set()
    for shard, names in by_shard.items():
        with safe_open(os.path.join(checkpoint_dir, shard), framework="pt") as f:
            lengths.update(math.prod(f.get_slice(n).get_shape()) for n in names)
    if len(lengths) != 1:
        raise ValueError(f"KDA layers store A_log at different lengths: {lengths}")
    length = lengths.pop()
    if length < text_config.kda_num_heads:
        raise ValueError(
            f"A_log stores {length} entries for {text_config.kda_num_heads} heads"
        )
    return length
