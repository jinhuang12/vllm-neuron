# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 Config
==================

<-- MODEL-SPECIFIC: every field below is DeepSeek-V4-specific and parsed
from the pinned ``config.json`` of ``deepseek-ai/DeepSeek-V4-Flash-0731``.

The config also owns the small amount of *derived* structure the rest of
the family reads instead of re-deriving per module:

* the per-layer KV-compression class (SWA-only / C4-sparse / C128-dense),
* which layers route MoE through the vocab-indexed hash table,
* the block-FP8 linear inventory with each linear's local K extent, which
  the factory uses to enforce the ``K_local % 128 == 0`` activation-group
  alignment invariant.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .quantization import QuantizationSpec

#: Config fields present in the checkpoint that this port does not consume,
#: with the reason.
#:
#: The ``dspark_*`` fields drive the checkpoint's own draft model. The
#: checkpoint repo ships DeepSeek's reference implementation, and its
#: ``DSparkBlock`` reads every one of them: ``dspark_target_layer_ids``
#: selects the three target layers whose hc-mean hidden states are
#: concatenated into ``main_proj`` (hence the checkpoint's
#: ``mtp.0.main_proj.weight`` of shape ``[4096, 3 * 4096]``),
#: ``dspark_block_size`` is the number of tokens drafted per step, and
#: ``dspark_noise_token_id`` / ``dspark_markov_rank`` parameterize the
#: draft input and the Markov head.
#:
#: They are unread HERE because this port carries no drafter: the port
#: plan's speculative-decoding row describes upstream vLLM 0.21.0's
#: ``DeepSeekV4MultiTokenPredictorLayer`` (``enorm``/``hnorm`` +
#: ``e_proj``/``h_proj``), whose parameters do not exist in this
#: checkpoint, so it would load nothing. That is a recorded plan defect
#: awaiting a replan, NOT a like-for-like omission — do not re-label it
#: as one.
IGNORED_CHECKPOINT_FIELDS: dict[str, str] = {
    "dspark_block_size": "draft-model field; no drafter in this port (plan defect)",
    "dspark_noise_token_id": "draft-model field; no drafter in this port (plan defect)",
    "dspark_target_layer_ids": (
        "draft-model field; no drafter in this port (plan defect)"
    ),
    "dspark_markov_rank": "draft-model field; no drafter in this port (plan defect)",
}

#: Per-layer KV-compression classes.
LAYER_CLASS_SWA_ONLY = "swa_only"
LAYER_CLASS_SPARSE_C4 = "sparse_c4"
LAYER_CLASS_DENSE_C128 = "dense_c128"


@dataclass
class DeepseekV4Config:
    # <-- MODEL-SPECIFIC: architecture parameters
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    # MLA latent dim. Upstream binds head_dim to kv_lora_rank
    # (deepseek_v4.py:1076 @ v0.21.0): 512 = 448 NoPE + 64 RoPE.
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    sliding_window: int = 128
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    hidden_act: str = "silu"

    # <-- MODEL-SPECIFIC: KV compression schedule. 46 entries; entries
    # beyond num_hidden_layers are unused padding in the checkpoint.
    compress_ratios: list[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None

    # <-- MODEL-SPECIFIC: DSA lightning indexer (ratio-4 layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    # <-- MODEL-SPECIFIC: MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    num_hash_layers: int = 3
    expert_dtype: str = "fp4"

    # <-- MODEL-SPECIFIC: hash-context (mhc) residual stream
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # <-- MODEL-SPECIFIC: multi-token prediction draft
    num_nextn_predict_layers: int = 1

    # Framework config
    neuron_config: NeuronConfig | None = None

    # Quantization spec parsed from the HuggingFace ``quantization_config``
    # (populated by :meth:`from_configs`). ``None`` means "not quantized".
    quant_spec: QuantizationSpec | None = field(default=None)

    # >>> PARALLELISM (plan-chosen, not operator-supplied) <<<
    # The shared-expert MLP is sharded over a subgroup of this size and
    # replicated across the remaining subgroups. A 64-way ``down_proj``
    # split would give K_local = 2048 / 64 = 32, which splits the
    # 128-element dynamic activation group the block-FP8 path quantizes
    # over and silently changes numerics relative to the GPU incumbent
    # (whose TP=8 gives K_local = 256). 16-way gives K_local = 128,
    # exactly one group.
    shared_expert_tp: int = 16

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    # ------------------------------------------------------------------
    # Derived MLA dims
    # ------------------------------------------------------------------
    @property
    def kv_lora_rank(self) -> int:
        """Compressed latent width (NoPE + RoPE)."""
        return self.head_dim

    @property
    def qk_nope_head_dim(self) -> int:
        """Latent width carrying the position-independent part."""
        return self.head_dim - self.qk_rope_head_dim

    # ------------------------------------------------------------------
    # Derived per-layer structure
    # ------------------------------------------------------------------
    def compress_ratio(self, layer_idx: int) -> int:
        """Return the KV compression ratio for ``layer_idx``.

        Mirrors upstream ``compress_ratio = max(1, compress_ratios[layer_id])``
        (deepseek_v4.py:955-961 @ v0.21.0). A ratio of 1 means the layer
        keeps no compressed cache of its own and is SWA-only.
        """
        if layer_idx >= len(self.compress_ratios):
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries; "
                f"layer_idx={layer_idx} is out of range."
            )
        return max(1, int(self.compress_ratios[layer_idx]))

    def layer_class(self, layer_idx: int) -> str:
        """Classify ``layer_idx`` as SWA-only, C4-sparse or C128-dense."""
        ratio = self.compress_ratio(layer_idx)
        if ratio <= 1:
            return LAYER_CLASS_SWA_ONLY
        if ratio == 4:
            return LAYER_CLASS_SPARSE_C4
        return LAYER_CLASS_DENSE_C128

    def has_indexer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` runs the DSA lightning indexer.

        Upstream builds the indexer only on ratio-4 layers
        (deepseek_v4.py:1037-1050 @ v0.21.0).
        """
        return self.layer_class(layer_idx) == LAYER_CLASS_SPARSE_C4

    def has_compressed_cache(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` allocates its own compressed latent cache."""
        return self.layer_class(layer_idx) != LAYER_CLASS_SWA_ONLY

    def is_hash_moe_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` routes MoE through the ``tid2eid`` table."""
        return layer_idx < self.num_hash_layers

    # ------------------------------------------------------------------
    # YaRN RoPE scaling
    # ------------------------------------------------------------------
    # <-- MODEL-SPECIFIC: this checkpoint DOES enable YaRN. The pinned
    # config.json carries ``rope_scaling = {"type": "yarn", "factor": 16,
    # "original_max_position_embeddings": 65536, "beta_fast": 32,
    # "beta_slow": 1}``, and 16 * 65536 == 1048576 ==
    # max_position_embeddings. The reference implementation applies the
    # frequency interpolation whenever ``original_seq_len > 0``
    # (inference/model.py:206-235), so a port that skipped it would place
    # every position past 65536 on the wrong frequencies. These
    # properties exist so the RoPE table builder reads one source.
    @property
    def rope_is_yarn(self) -> bool:
        """Whether YaRN frequency interpolation applies."""
        scaling = self.rope_scaling
        if not scaling:
            return False
        return str(scaling.get("type", scaling.get("rope_type", ""))).lower() == "yarn"

    @property
    def rope_original_seq_len(self) -> int:
        """Pre-scaling context length YaRN interpolates from (0 disables)."""
        if not self.rope_is_yarn:
            return 0
        return int(self.rope_scaling.get("original_max_position_embeddings", 0))

    @property
    def rope_factor(self) -> float:
        """YaRN extension factor."""
        if not self.rope_is_yarn:
            return 1.0
        return float(self.rope_scaling.get("factor", 1.0))

    @property
    def rope_beta_fast(self) -> int:
        """YaRN high-frequency correction bound."""
        if not self.rope_is_yarn:
            return 32
        return int(self.rope_scaling.get("beta_fast", 32))

    @property
    def rope_beta_slow(self) -> int:
        """YaRN low-frequency correction bound."""
        if not self.rope_is_yarn:
            return 1
        return int(self.rope_scaling.get("beta_slow", 1))

    def rope_theta_for_layer(self, layer_idx: int) -> float:
        """Dual-theta RoPE base for ``layer_idx``.

        Compressed layers use ``compress_rope_theta``; SWA-only layers use
        the uncompressed ``rope_theta`` (deepseek_v4.py:1013-1035 @ v0.21.0).
        """
        if self.has_compressed_cache(layer_idx):
            return float(self.compress_rope_theta)
        return float(self.rope_theta)

    # ------------------------------------------------------------------
    # Block-FP8 linear inventory (drives the factory's alignment check)
    # ------------------------------------------------------------------
    def block_fp8_linear_plan(
        self, tp_size: int
    ) -> list[tuple[str, int, str, int]]:
        """Return ``(name, k_global, shard_kind, k_local)`` per block-FP8 linear.

        ``shard_kind`` is one of ``"replicated"`` (no K split),
        ``"column"`` (output dim split, K intact) or ``"row"`` (input dim
        split, K divided). The block-FP8 activation path quantizes over
        128-element K groups, so every entry must satisfy
        ``k_local % 128 == 0``; the factory enforces that.

        Args:
            tp_size: The attention/dense tensor-parallel degree.
        """
        shared_tp = self.shared_expert_tp
        return [
            # 4096 -> [q_lora_rank, kv_lora_rank], replicated (upstream
            # builds it with disable_tp=True, deepseek_v4.py:973-980).
            ("self_attn.fused_wqa_wkv", self.hidden_size, "replicated", self.hidden_size),
            # q_lora_rank -> num_heads x kv_lora_rank, column-parallel over heads.
            ("self_attn.wq_b", self.q_lora_rank, "column", self.q_lora_rank),
            # Grouped o-projection stage A: per-head latent -> o_lora_rank.
            # Group-parallel: each core owns one head-slice, K stays the
            # per-head latent width.
            ("self_attn.wo_a", self.kv_lora_rank, "column", self.kv_lora_rank),
            # o_groups x o_lora_rank -> hidden, row-parallel over TP.
            (
                "self_attn.wo_b",
                self.o_groups * self.o_lora_rank,
                "row",
                (self.o_groups * self.o_lora_rank) // tp_size,
            ),
            # Shared-expert MLP on the 16-way subgroup.
            (
                "mlp.shared_expert.gate_up_proj",
                self.hidden_size,
                "column",
                self.hidden_size,
            ),
            (
                "mlp.shared_expert.down_proj",
                self.moe_intermediate_size,
                "row",
                self.moe_intermediate_size // shared_tp,
            ),
            # DSA indexer wq_b: q_lora_rank -> index_n_heads x index_head_dim,
            # replicated (upstream ReplicatedLinear,
            # deepseek_v4_attention.py:1120-1126).
            ("self_attn.indexer.wq_b", self.q_lora_rank, "replicated", self.q_lora_rank),
        ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        # Parse the checkpoint's quantization config. ``expert_dtype`` sits
        # at the top level of config.json, not inside quantization_config,
        # so it is passed separately.
        filtered_dict["quant_spec"] = QuantizationSpec.from_hf_quantization_config(
            config_dict.get("quantization_config"),
            expert_dtype=config_dict.get("expert_dtype"),
        )

        return cls(**filtered_dict)
