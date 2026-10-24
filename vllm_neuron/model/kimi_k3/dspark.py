# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: three causal MLA layers over the target's tapped states.

The drafter projects three target hidden-state taps into each layer's latent cache,
runs a block of query rows through three MLA layers with a dense SiLU MLP, and
picks its drafts in order with a Markov head. Its embedding and LM head are the
target's. Every weight is split over the tensor-parallel group: q_b, kv_b and
o_proj by head, the MLP by its width, q_a, kv_a and the context projection by
their input width (each output is then all-reduced), and the second Markov
weight by vocabulary. The first Markov weight and the norms are whole.
"""

from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .config import _as_dict
from .model import KimiK3RMSNorm, _tp_group, _Weight

# Checkpoint tensors the drafter does not load: the training-only confidence head,
# and the embedding and LM head, which come from the target.
SKIPPED_PREFIXES = ("confidence_head.", "embed_tokens.", "lm_head.")


@dataclass
class K3DSparkConfig:
    """Draft config read from the DSpark checkpoint's ``config.json``; every size is required."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    vocab_size: int
    markov_rank: int
    target_hidden_size: int
    target_layer_ids: tuple[int, ...]
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_hf_config(
        cls, hf_config: PretrainedConfig | dict, neuron_config: NeuronConfig | None = None
    ) -> "K3DSparkConfig":
        """Build the config from the draft's HF config, a dict or a JSON path."""
        config_dict = _as_dict(hf_config)
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in config_dict.items() if k in names}
        dtype = kwargs.get("torch_dtype", config_dict.get("dtype"))
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        kwargs["target_layer_ids"] = tuple(config_dict["target_layer_ids"])
        return cls(**kwargs, neuron_config=neuron_config)


def _per_rank(size: int, tp_size: int, what: str) -> int:
    """Return size // tp_size, or raise when ``size`` does not divide."""
    if size % tp_size:
        raise ValueError(f"{what} {size} does not divide over tensor-parallel size {tp_size}")
    return size // tp_size


class _ShardedWeight(_Weight):
    """A checkpoint module holding this rank's slice of its stored ``weight`` along ``dim``."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, dim: int, tp_size: int):
        super().__init__(shape, dtype)
        set_weight_loader(self.weight, sharding_weight_loader(dim, shape[dim], tp_size))


class K3DSparkMLA(nn.Module):
    """One draft layer's MLA weights for this rank's heads, and its latent cache."""

    def __init__(self, c: K3DSparkConfig, tp_size: int):
        super().__init__()
        heads = _per_rank(c.num_attention_heads, tp_size, "MLA heads")
        k = _per_rank(c.hidden_size, tp_size, "hidden size")
        dtype = c.torch_dtype
        q_head = c.qk_nope_head_dim + c.qk_rope_head_dim
        self.q_a_proj = _ShardedWeight((c.q_lora_rank, k), dtype, 1, tp_size)
        self.q_a_layernorm = KimiK3RMSNorm(c.q_lora_rank, c.rms_norm_eps, dtype)
        self.q_b_proj = _ShardedWeight((heads * q_head, c.q_lora_rank), dtype, 0, tp_size)
        latent = c.kv_lora_rank + c.qk_rope_head_dim
        self.kv_a_proj_with_mqa = _ShardedWeight((latent, k), dtype, 1, tp_size)
        self.kv_a_layernorm = KimiK3RMSNorm(c.kv_lora_rank, c.rms_norm_eps, dtype)
        kv_b_rows = heads * (c.qk_nope_head_dim + c.v_head_dim)
        self.kv_b_proj = _ShardedWeight((kv_b_rows, c.kv_lora_rank), dtype, 0, tp_size)
        self.o_proj = _ShardedWeight((c.hidden_size, heads * c.v_head_dim), dtype, 1, tp_size)
        # Paged latent cache [blocks, block_size, kv_lora_rank + rope], bound with the KV caches.
        self.kv_cache = None


class K3DSparkMLP(nn.Module):
    """Dense SiLU MLP weights for this rank's share of the MLP width."""

    def __init__(self, c: K3DSparkConfig, tp_size: int):
        super().__init__()
        width = _per_rank(c.intermediate_size, tp_size, "MLP width")
        dtype = c.torch_dtype
        self.gate_proj = _ShardedWeight((width, c.hidden_size), dtype, 0, tp_size)
        self.up_proj = _ShardedWeight((width, c.hidden_size), dtype, 0, tp_size)
        self.down_proj = _ShardedWeight((c.hidden_size, width), dtype, 1, tp_size)


class K3DSparkDecoderLayer(nn.Module):
    """One draft layer; ``layer_idx`` names its cache after the target's layers."""

    def __init__(self, c: K3DSparkConfig, layer_idx: int, tp_size: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = K3DSparkMLA(c, tp_size)
        self.mlp = K3DSparkMLP(c, tp_size)
        self.input_layernorm = KimiK3RMSNorm(c.hidden_size, c.rms_norm_eps, c.torch_dtype)
        self.post_attention_layernorm = KimiK3RMSNorm(
            c.hidden_size, c.rms_norm_eps, c.torch_dtype
        )

    @property
    def cache_name(self) -> str:
        """The name of this layer's latent cache in the KV cache config."""
        return f"layers.{self.layer_idx}.self_attn"


class K3DSparkMarkovHead(nn.Module):
    """The Markov head: the first weight whole, the second split by vocabulary."""

    def __init__(self, c: K3DSparkConfig, tp_size: int):
        super().__init__()
        vocab = _per_rank(c.vocab_size, tp_size, "vocabulary")
        self.markov_w1 = _Weight((c.vocab_size, c.markov_rank), c.torch_dtype)
        self.markov_w2 = _ShardedWeight((vocab, c.markov_rank), c.torch_dtype, 0, tp_size)


class K3DSparkForCausalLM(nn.Module):
    """The DSpark drafter with the target's embedding and LM head."""

    def __init__(self, config: K3DSparkConfig, start_layer_idx: int = 0):
        super().__init__()
        c = config
        self.config = c
        group = _tp_group()
        self.tp_size = group.world_size if group is not None else 1
        taps = len(c.target_layer_ids) * c.target_hidden_size
        self.context_proj = _ShardedWeight(
            (c.hidden_size, _per_rank(taps, self.tp_size, "tap width")),
            c.torch_dtype, 1, self.tp_size,
        )
        self.context_norm = KimiK3RMSNorm(c.hidden_size, c.rms_norm_eps, c.torch_dtype)
        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(c, start_layer_idx + i, self.tp_size)
                for i in range(c.num_hidden_layers)
            ]
        )
        self.final_norm = KimiK3RMSNorm(c.hidden_size, c.rms_norm_eps, c.torch_dtype)
        self.markov_head = K3DSparkMarkovHead(c, self.tp_size)
        # The target's modules, set by alias_target_head; none is allocated here.
        self.embed_tokens = None
        self.lm_head = None
        self.register_buffer("context_kv_weight", None, persistent=False)

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig | dict,
        start_layer_idx: int = 0,
        neuron_config: NeuronConfig | None = None,
    ) -> "K3DSparkForCausalLM":
        """Build the drafter from its HF config; its caches follow the target's layers."""
        return cls(K3DSparkConfig.from_hf_config(config, neuron_config), start_layer_idx)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load this rank's slice of every draft weight, then build the load-time products."""
        group = _tp_group()
        rank = group.rank_in_group if group is not None else 0
        checkpoint = SafetensorsCheckpoint(str(checkpoint_path), cache_dir)
        if dist.is_initialized():
            load = checkpoint.load_sharded_pipelined
        else:
            load = checkpoint.load_sharded
        result = load(rank, self.tp_size, self, {}, device, strict=False)
        unmapped = [k for k in result.unexpected_keys if not k.startswith(SKIPPED_PREFIXES)]
        if result.missing_keys or unmapped:
            raise ValueError(
                "DSpark checkpoint does not match the drafter: missing "
                f"{sorted(result.missing_keys)}, unmapped {sorted(unmapped)}"
            )
        self.load_state_dict(result.state_dict, strict=False, assign=True)
        self.prepare_after_load()

    def prepare_after_load(self) -> None:
        """Stack the layers' kv_a slices in fp32: the context KV projection's weight."""
        self.context_kv_weight = torch.cat(
            [layer.self_attn.kv_a_proj_with_mqa.weight for layer in self.layers]
        ).float()

    def alias_target_head(self, target_model: nn.Module) -> None:
        """Use the target's embedding and LM head modules; no weight is copied."""
        embed, lm_head = target_model.model.embed_tokens, target_model.lm_head
        c = self.config
        vocab_rows = c.vocab_size // self.tp_size
        if embed.weight.shape[-1] != c.hidden_size or tuple(lm_head.weight.shape) != (
            vocab_rows, c.hidden_size
        ):
            raise ValueError(
                f"target embedding {tuple(embed.weight.shape)} and LM head "
                f"{tuple(lm_head.weight.shape)} do not fit the drafter's hidden size "
                f"{c.hidden_size} and vocabulary slice of {vocab_rows} rows"
            )
        self.embed_tokens = embed
        self.lm_head = lm_head

    def get_kv_spec(self) -> KVSpec:
        """One key-only latent cache per draft layer, as the target's MLA layers hold."""
        c = self.config
        latent = c.kv_lora_rank + c.qk_rope_head_dim
        return KVSpec(
            layers=[
                LayerSpec(layer.cache_name, 1, latent, c.torch_dtype, key_only=True)
                for layer in self.layers
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Bind each draft layer's latent cache."""
        for layer in self.layers:
            if layer.cache_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer.cache_name} not initialized")
            # [blocks, 1, block_size, D]: one latent head.
            layer.self_attn.kv_cache = kv_caches[layer.cache_name][0].squeeze(1)
