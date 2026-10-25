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

import math
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._functional_collectives import all_reduce
from transformers import PretrainedConfig

from vllm_neuron.functional.argmax import argmax as distributed_argmax
from vllm_neuron.functional.embedding import embedding
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .config import _as_dict
from .model import (
    KimiK3RMSNorm,
    _optional_kernel,
    _output_projection,
    _require,
    qkv_proj_fp32acc,
    _tp_group,
    _Weight,
)

o_proj = _optional_kernel("attention.o_proj", "o_proj")

# Checkpoint tensors the drafter does not load: the training-only confidence head,
# and the embedding and LM head, which come from the target.
SKIPPED_PREFIXES = ("confidence_head.", "embed_tokens.", "lm_head.")

# The output-projection kernels' partition count: the TKG kernel's row limit per
# call, and the widest K block of its input layout.
_PARTITIONS = 128
# LM-head columns per fused norm-projection call: half of a rank's 5,120 at TP 32,
# under the kernel's 3,072-column limit.
_LOGIT_COLUMNS = 2560


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
    max_position_embeddings: int
    rope_parameters: dict
    rope_interleave: bool = True
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


def _yarn_mscale(scale: float, mscale: float) -> float:
    """YaRN's attention scale for a context extended ``scale`` times."""
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def yarn_rope_table(c: K3DSparkConfig, positions: int) -> torch.Tensor:
    """The draft's YaRN rotary for positions 0 to ``positions - 1``: ``[P, rope]`` fp32, cos | sin.

    Every entry is computed in fp64 and rounded once to fp32.
    """
    rope = c.rope_parameters
    if rope.get("rope_type") != "yarn" or not c.rope_interleave:
        raise ValueError(f"the DSpark drafter takes an interleaved YaRN rotary, got {rope}")
    dim, base, factor = c.qk_rope_head_dim, float(rope["rope_theta"]), float(rope["factor"])
    original = rope["original_max_position_embeddings"]

    def correction_dim(rotations: float) -> float:
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(rope.get("beta_fast", 32))), 0)
    high = min(math.ceil(correction_dim(rope.get("beta_slow", 1))), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    inv_freq = ramp / (factor * freqs) + (1 - ramp) / freqs
    mscale = (
        _yarn_mscale(factor, rope.get("mscale", 1))
        / _yarn_mscale(factor, rope.get("mscale_all_dim", 0))
        * rope.get("attn_factor", 1)
    )
    angles = torch.arange(positions, dtype=torch.float64)[:, None] * inv_freq
    return (torch.cat([angles.cos(), angles.sin()], -1) * mscale).float()


def _rotate_interleaved(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd) column pair of ``x`` ``[T, L, rope]`` by its row's cos | sin."""
    cos, sin = table[:, None].chunk(2, -1)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)


def _k_shard_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x`` ``[T, K]`` times ``weight`` ``[K, N]`` through the output-projection kernels."""
    rows, width = x.shape
    if rows > _PARTITIONS:
        return _require(o_proj, "output projection")(x[None], weight)[0]
    blocks = -(-width // _PARTITIONS)
    layout = x.reshape(rows, blocks, width // blocks).permute(2, 0, 1).unsqueeze(-1)
    return _output_projection(layout, weight)


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

    def __init__(
        self, config: K3DSparkConfig, start_layer_idx: int = 0, max_positions: int | None = None
    ):
        super().__init__()
        c = config
        self.config = c
        group = _tp_group()
        self.tp_size = group.world_size if group is not None else 1
        self.tp_rank = group.rank_in_group if group is not None else 0
        self.tp_device_group = group.device_group if group is not None else None
        # Rows of the rotary table: the longest sequence the drafter serves.
        self.max_positions = max_positions or c.max_position_embeddings
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
        # The target's embedding and stacked LM head, set by alias_target_head; none is
        # allocated here.
        self.embed_tokens = None
        self.head_weights = None
        self.register_buffer("context_kv_weight", None, persistent=False)
        self.register_buffer("rope_table", None, persistent=False)

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig | dict,
        start_layer_idx: int = 0,
        neuron_config: NeuronConfig | None = None,
        max_positions: int | None = None,
    ) -> "K3DSparkForCausalLM":
        """Build the drafter from its HF config; its caches follow the target's layers."""
        return cls(
            K3DSparkConfig.from_hf_config(config, neuron_config), start_layer_idx, max_positions
        )

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
        """Build the fp32 context KV weight from the kv_a slices, and the rotary table."""
        self.context_kv_weight = torch.cat(
            [layer.self_attn.kv_a_proj_with_mqa.weight for layer in self.layers]
        ).float()
        table = yarn_rope_table(self.config, self.max_positions)
        self.rope_table = table.to(self.context_kv_weight.device)

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the ranks' partials over the tensor-parallel group."""
        if self.tp_size == 1:
            return x
        return all_reduce(x, reduceOp="sum", group=self.tp_device_group)

    def _rank_slice(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's share of the last dim of ``x``."""
        width = x.shape[-1] // self.tp_size
        return x[..., self.tp_rank * width : (self.tp_rank + 1) * width]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
    ) -> None:
        """Project the target's tapped rows into every draft layer's latent cache.

        Args:
            context_states: The target's taps concatenated, ``[T, taps * H]``, whole on
                every rank.
            context_positions: Position per row ``[T]``.
            context_slot_mapping: This rank's cache slot per row ``[T]``. The runner maps
                a row another rank owns to the null block.
        """
        c = self.config
        partial = _k_shard_matmul(self._rank_slice(context_states), self.context_proj.weight.t())
        hidden = self.context_norm(self._all_reduce(partial.float()).to(c.torch_dtype))
        kv = _k_shard_matmul(self._rank_slice(hidden).float(), self.context_kv_weight.t())
        kv = self._all_reduce(kv).view(len(context_states), len(self.layers), -1)
        latent, rope = kv.split([c.kv_lora_rank, c.qk_rope_head_dim], -1)
        rope = _rotate_interleaved(rope, self.rope_table[context_positions])
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            rows = torch.cat([attn.kv_a_layernorm(latent[:, i]), rope[:, i]], -1)
            cache = attn.kv_cache.view(-1, rows.shape[-1])
            cache.index_copy_(0, context_slot_mapping, rows.to(cache.dtype))

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm and the target's LM head on ``[B, S, H]`` rows: ``[B, S, V / TP]`` bf16.

        Each stacked ``[H, V / 2TP]`` half is read in column parts of at most 2,560, in order.
        """
        norm = self.final_norm
        parts = [part for half in self.head_weights for part in half.split(_LOGIT_COLUMNS, 1)]
        project = _require(qkv_proj_fp32acc, "fused norm and projection")
        return torch.cat([project(hidden, w, norm.weight, norm.eps) for w in parts], -1)

    def _pick(self, logits: torch.Tensor) -> torch.Tensor:
        """The argmax over the whole vocabulary of this rank's ``[B, V / TP]`` logits."""
        if self.tp_size == 1:
            return torch.argmax(logits, dim=-1)
        return distributed_argmax(logits, dim=-1, gather_dim=-1, process_group=self.tp_device_group)

    def sample_drafts(self, hidden: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Pick each sequence's drafts in order, every pick biased by the one before it.

        Args:
            hidden: The block's output rows ``[B, k, H]``, before the final norm.
            anchors: Each sequence's anchor token ``[B]``, the pick before the first.

        Returns:
            The drafts ``[B, k]`` int32, the same on every rank. A tie goes to the
            lowest token id.
        """
        base = self.compute_logits(hidden)
        head = self.markov_head
        prev, picks = anchors, []
        for i in range(base.shape[1]):
            embed = embedding(prev, head.markov_w1.weight)
            bias = _k_shard_matmul(embed, head.markov_w2.weight.t())
            prev = self._pick(base[:, i].float() + bias.float()).to(torch.int32)
            picks.append(prev)
        return torch.stack(picks, dim=1)

    def alias_target_head(self, target_model: nn.Module) -> None:
        """Use the target's embedding and its two stacked LM-head halves; no weight is copied."""
        embed, halves = target_model.model.embed_tokens, target_model.head_weights
        if halves is None:
            raise ValueError(
                "the target's LM head is not stacked yet: run its prepare_after_load first"
            )
        c = self.config
        half = (c.hidden_size, c.vocab_size // self.tp_size // 2)
        shapes = [tuple(w.shape) for w in halves]
        if embed.weight.shape[-1] != c.hidden_size or shapes != [half, half]:
            raise ValueError(
                f"target embedding {tuple(embed.weight.shape)} and LM-head halves {shapes} "
                f"do not fit the drafter's hidden size {c.hidden_size} and two {half} "
                "halves of its vocabulary slice"
            )
        self.embed_tokens = embed
        self.head_weights = halves

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
