# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: three causal MLA layers over the target's tapped states.

The drafter projects three target hidden-state taps into each layer's latent cache,
runs a block of query rows through three MLA layers with a dense SiLU MLP, and
picks its drafts in order with a Markov head. Its embedding and LM head are the
target's. Every weight is split over the tensor-parallel group: q_b, kv_b and
o_proj by head, the MLP by its width, q_a, kv_a and the context projection by
their input width (each output is then all-reduced), and the second Markov
weight by vocabulary. The first Markov weight and the norms are whole. Each
layer's attention is one call of the paged MLA kernel's window mode, with the
rotary on and no output gate.
"""

import math
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._functional_collectives import all_reduce
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_dcp_group

from vllm_neuron.functional.argmax import argmax as distributed_argmax
from vllm_neuron.functional.attention.mla_paged import mla_paged_window_dcp, prepare_mla_weights
from vllm_neuron.functional.embedding import embedding
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3.dflash_model import (
    _correct_target_positions_for_async,
    _current_seq_lens,
    _paged_slots,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .config import _as_dict
from .model import (
    KimiK3RMSNorm,
    _int8_pages,
    _optional_kernel,
    _output_projection,
    _page_layout,
    _require,
    qkv_proj_fp32acc,
    _tp_group,
    _waited,
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


def _owned_rows(context_lens, positions, block_size: int, group):
    """The cached rows of each sequence this rank holds ``[B]``, and which block rows ``[B, k]``.

    Rank r of a decode context group of n ranks holds sequence block j where j % n == r.
    """
    size = group.world_size
    rank = group.rank_in_group_tensor.reshape(1) if size > 1 else 0
    turn = block_size * size
    last = (context_lens % turn - rank * block_size).clamp(0, block_size)
    held = context_lens // turn * block_size + last
    owner = positions // block_size % size == rank
    return held.to(torch.int32), owner.to(torch.int32)


def _owned_slots(positions, seqs, table, block_size: int, max_blocks: int, group):
    """Each row's cache slot on this rank ``[R]``: real where the rank holds its block, else 0.

    At one rank this is the shared paged rule. Under decode context parallelism, table column
    ``c`` names this rank's block for sequence blocks ``c * n`` to ``c * n + n - 1``, so the row's
    owner rank writes at column ``position // (block_size * n)``, and the others write the null
    block, as the runner maps the target's rows.
    """
    size = group.world_size
    if size == 1:
        return _paged_slots(positions, seqs, table, block_size, max_blocks)
    rank = group.rank_in_group_tensor.reshape(1)
    column = torch.clamp(positions // (block_size * size), max=max_blocks - 1)
    ids = torch.index_select(table.contiguous().reshape(-1), 0, seqs * max_blocks + column)
    slot = torch.where(ids < 0, 0, ids * block_size + positions % block_size)
    return torch.where(positions // block_size % size == rank, slot, 0)


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
        # Load-time layouts: the rank's K rows of q_a and kv_a in fp32 [K, Q + L], o_proj as
        # [heads * V, H], and the attention kernel's q_b and kv_b layouts.
        for name in ("latent_weight", "output_weight", "w_qb_t", "w_uk", "w_uv_t"):
            self.register_buffer(name, None, persistent=False)
        # Paged latent cache [blocks, block_size, kv_lora_rank + rope], bound with the KV caches,
        # its int8 [blocks, page bytes] page buffer and where the cache sits in a page.
        self.kv_cache = None
        self.pages = None
        self.cache_layout = None

    def prepare_after_load(self) -> None:
        """Lay out the weights the block pass reads, once; the loaded weights stay."""
        latent = torch.cat([self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight])
        self.latent_weight = latent.float().t().contiguous()
        self.output_weight = self.o_proj.weight.t().contiguous()
        self.w_qb_t, self.w_uk, self.w_uv_t = prepare_mla_weights(
            self.q_b_proj.weight, self.kv_b_proj.weight
        )


class K3DSparkMLP(nn.Module):
    """Dense SiLU MLP weights for this rank's share of the MLP width."""

    def __init__(self, c: K3DSparkConfig, tp_size: int):
        super().__init__()
        width = _per_rank(c.intermediate_size, tp_size, "MLP width")
        dtype = c.torch_dtype
        self.gate_proj = _ShardedWeight((width, c.hidden_size), dtype, 0, tp_size)
        self.up_proj = _ShardedWeight((width, c.hidden_size), dtype, 0, tp_size)
        self.down_proj = _ShardedWeight((c.hidden_size, width), dtype, 1, tp_size)
        # Load-time layouts: gate and up stacked as [H, 2 * width], down as [width, H].
        self.register_buffer("gate_up_weight", None, persistent=False)
        self.register_buffer("down_weight", None, persistent=False)

    def prepare_after_load(self) -> None:
        """Lay out the weights the block pass reads, once; the loaded weights stay."""
        gate_up = torch.cat([self.gate_proj.weight, self.up_proj.weight])
        self.gate_up_weight = gate_up.t().contiguous()
        self.down_weight = self.down_proj.weight.t().contiguous()


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
        # The unloaded tree's tensors by name, kept aside by load_weights_lite.
        self._unloaded: dict[str, torch.Tensor] | None = None

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
        self._restore_unloaded()
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

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lay out the draft's weights on the unloaded tree, for the device-free compile.

        It reads nothing from the checkpoint: each layout takes its shape and dtype from the
        meta parameters, so the tree holds the tensors a full load holds and the worker sizes
        the same KV cache. The unloaded tensors stay aside for a later load.
        """
        self._unloaded = dict(self.named_parameters(remove_duplicate=False))
        self._unloaded.update(self.named_buffers(remove_duplicate=False))
        self.prepare_after_load()

    def _restore_unloaded(self) -> None:
        """Undo a lite load: drop the layouts it added and put back the tensors it kept."""
        if self._unloaded is None:
            return
        kept, self._unloaded = self._unloaded, None

        def assign(name: str, value: torch.Tensor | None) -> None:
            path, _, leaf = name.rpartition(".")
            setattr(self.get_submodule(path), leaf, value)

        tensors = [*self.named_parameters(remove_duplicate=False),
                   *self.named_buffers(remove_duplicate=False)]
        for name in [name for name, _ in tensors if name not in kept]:
            assign(name, None)
        for name, tensor in kept.items():
            assign(name, tensor)

    def prepare_after_load(self) -> None:
        """Build the fp32 context KV weight, the rotary table and each layer's layouts."""
        self.context_kv_weight = torch.cat(
            [layer.self_attn.kv_a_proj_with_mqa.weight for layer in self.layers]
        ).float()
        table = yarn_rope_table(self.config, self.max_positions)
        self.rope_table = table.to(self.context_kv_weight.device)
        for layer in self.layers:
            layer.self_attn.prepare_after_load()
            layer.mlp.prepare_after_load()

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
            context_slot_mapping: This rank's cache slot per row ``[T]``; a row another
                rank owns maps to the null block.
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

    @torch.no_grad()
    def forward(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: dict,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor,
        prev_num_draft_tokens: torch.Tensor,
        req_indices_per_token: torch.Tensor,
        cur_num_draft_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft round after a target step: context write, block pass, then the picks.

        Each target row's taps become context rows at its position. The block is the
        anchor, the target's last accepted token, then mask tokens, at positions p to
        p + k - 1, p being the context length; every row is picked, row j predicting
        position p + j + 1. The proposer sets k, ``num_speculative_tokens``, and the
        checkpoint's ``mask_token_id``.

        Returns:
            The anchor with the drafts ``[B, 1 + k]`` int32, and the drafts ``[B, k]``.
        """
        c, k = self.config, self.num_speculative_tokens
        batch, device = last_token_indices.shape[0], target_hidden_states.device
        valid = (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < c.vocab_size)
        valid_count = valid.sum(dim=1)
        positions = _correct_target_positions_for_async(
            target_positions, prev_sampled_token_ids, prev_num_draft_tokens,
            req_indices_per_token, c.vocab_size,
        )
        context_lens = _current_seq_lens(
            positions[last_token_indices], raw_sampled_token_ids, prev_sampled_token_ids,
            prev_num_draft_tokens, cur_num_draft_tokens, c.vocab_size, k,
            valid_per_req=valid_count,
        )
        block_positions = (context_lens[:, None] + torch.arange(k, device=device)).reshape(-1)

        meta = attn_metadata[self.layers[0].cache_name]
        table = meta["block_table_tensor"][:batch].to(torch.int32)
        block_size, max_blocks = meta["block_size"], int(meta["max_blocks_per_seq"])
        # Each target row's sequence, from the sequences' last-row boundaries; rows past
        # the last one are padding and write the null block.
        rows = torch.arange(target_hidden_states.shape[0], device=device)
        row_seq = (rows[:, None] > last_token_indices[None, :]).sum(dim=1)
        is_pad = row_seq >= batch
        row_seq = row_seq.clamp(max=batch - 1)
        group = get_dcp_group()
        slots = _owned_slots(positions, row_seq, table, block_size, max_blocks, group)
        self.precompute_and_store_context_kv(
            target_hidden_states, positions, torch.where(is_pad, 0, slots)
        )

        block_seq = torch.arange(batch * k, device=device) // k
        block_meta = {
            "slot_mapping": _owned_slots(
                block_positions, block_seq, table, block_size, max_blocks, group
            ),
            "block_size": block_size,
            "block_table_tensor": table,
            "max_blocks_per_seq": max_blocks,
            "context_lens": context_lens.to(torch.int32),
            "is_causal": True,
        }
        last = (valid_count - 1).clamp(min=0).unsqueeze(1).long()
        anchors = raw_sampled_token_ids.gather(1, last).squeeze(1).to(torch.int32)
        anchors = torch.where(valid_count > 0, anchors, torch.zeros_like(anchors))
        masks = torch.full((batch, k - 1), self.mask_token_id, dtype=torch.int32, device=device)
        ids = torch.cat([anchors[:, None], masks], dim=1)
        rank = _tp_group().rank_in_group_tensor.reshape(1) if self.tp_size > 1 else None
        embedded = _waited(self.embed_tokens(ids.reshape(-1), scatter_tokens=False, rank=rank))
        hidden = self.block_pass(
            embedded, block_positions, {layer.cache_name: block_meta for layer in self.layers}
        )
        drafts = self.sample_drafts(hidden, anchors)
        return torch.cat([anchors[:, None], drafts], dim=1), drafts

    def block_pass(
        self, embedded: torch.Tensor, positions: torch.Tensor, attn_metadata: dict
    ) -> torch.Tensor:
        """The three draft layers over the block's rows: ``[B, k, H]``, before the final norm.

        Args:
            embedded: The block's embedded rows ``[B * k, H]``, each sequence's k rows in order.
            positions: Position per row ``[B * k]``.
            attn_metadata: Per draft layer's cache name: ``block_table_tensor`` ``[B, pages]``,
                ``block_size``, ``context_lens`` ``[B]`` (the rows before the block) and
                ``slot_mapping`` ``[B * k]``, this rank's slot of each row.
        """
        if any(layer.self_attn.kv_cache.dtype != torch.bfloat16 for layer in self.layers):
            raise ValueError("the block pass reads bf16 latent caches")
        batch = attn_metadata[self.layers[0].cache_name]["block_table_tensor"].shape[0]
        x = embedded
        for layer in self.layers:
            meta = attn_metadata[layer.cache_name]
            x = x + self._attention(layer, x, positions, meta).to(x.dtype)
            x = x + self._mlp(layer, x).to(x.dtype)
        return x.view(batch, -1, x.shape[-1])

    def _attention(self, layer, x, positions, meta) -> torch.Tensor:
        """One layer's MLA over the block rows: the all-reduced output projection ``[R, H]``.

        The q_a and kv_a partials of the normed rows' K slice sum in fp32. The window mode then
        attends each row to its sequence's cached rows and to the block rows up to its own,
        and writes the block rows' latent rows at their slots.
        """
        attn = layer.self_attn
        normed = self._rank_slice(layer.input_layernorm(x)).float()
        latent = _waited(self._all_reduce(_k_shard_matmul(normed, attn.latent_weight)))
        table, block_size = meta["block_table_tensor"], meta["block_size"]
        group = get_dcp_group()
        held, owner = _owned_rows(meta["context_lens"], positions.view(table.shape[0], -1),
                                  block_size, group)
        out, _ = mla_paged_window_dcp(
            latent, None, attn.q_a_layernorm.weight, attn.kv_a_layernorm.weight, attn.w_qb_t,
            attn.w_uk, attn.w_uv_t, attn.pages, attn.cache_layout[0], block_size, table, held,
            meta["slot_mapping"], owner, self.config.rms_norm_eps, group,
            rope_table=self.rope_table, positions=positions,
        )
        weight = attn.output_weight
        return _waited(self._all_reduce(_output_projection(out.to(weight.dtype), weight)))

    def _mlp(self, layer, x) -> torch.Tensor:
        """One layer's dense SiLU MLP on the block rows: the all-reduced down projection."""
        mlp, norm = layer.mlp, layer.post_attention_layernorm
        project = _require(qkv_proj_fp32acc, "fused norm and projection")
        gate, up = project(x[None], mlp.gate_up_weight, norm.weight, norm.eps)[0].chunk(2, -1)
        act = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return _waited(self._all_reduce(_k_shard_matmul(act, mlp.down_weight)))

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
        """Bind each draft layer's latent cache and the int8 page view of its buffer."""
        for layer in self.layers:
            if layer.cache_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer.cache_name} not initialized")
            # [blocks, 1, block_size, D]: one latent head.
            attn = layer.self_attn
            attn.kv_cache = kv_caches[layer.cache_name][0].squeeze(1)
            attn.pages = _int8_pages(attn.kv_cache)
            attn.cache_layout = _page_layout(attn.kv_cache, attn.pages)
