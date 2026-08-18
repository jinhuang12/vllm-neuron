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
#: The ``dspark_*`` fields USED to sit here, because iteration 1 of this port
#: carried no drafter. They are parsed drafter inputs now (see
#: :mod:`.dspark_model`), so they left this list at the DSpark re-entry.
IGNORED_CHECKPOINT_FIELDS: dict[str, str] = {}

#: Checkpoint config fields whose declared value is CONTRADICTED by the
#: weights, with the resolution and its evidence. A field here is parsed and
#: kept (something outside this family may read it) but is never the
#: authority for a shape or a count inside the family.
#:
#: ``num_nextn_predict_layers = 1`` is the whole list. The checkpoint ships
#: THREE DSpark stages (``mtp.0``, ``mtp.1``, ``mtp.2``, and no ``mtp.3`` —
#: key census ``checkpoint-key-analysis.txt`` §A over all 72317 index keys)
#: and DeepSeek's own reference config declares ``n_mtp_layers = 3``
#: (``dsv4_ref/config.json``). **The weights are authoritative: 3 stages.**
#: The field stays parsed for exactly one reason: upstream
#: ``SpeculativeConfig`` reads it as ``n_predict`` and validates
#: ``num_speculative_tokens % n_predict == 0``
#: (``vllm/config/speculative.py:722-736`` @ ``ad7125a431``), which the
#: planned ``num_speculative_tokens = 5`` passes because 5 % 1 == 0. The
#: drafter's real block size is :attr:`DeepseekV4Config.dspark_block_size`.
CONTRADICTED_CHECKPOINT_FIELDS: dict[str, str] = {
    "num_nextn_predict_layers": (
        "declares 1; the checkpoint ships 3 mtp.* stages and dsv4_ref's "
        "config.json declares n_mtp_layers=3. Resolution: 3 stages, by the "
        "weights (see NUM_DSPARK_STAGES). Kept parsed only because upstream "
        "SpeculativeConfig reads it as n_predict."
    ),
}

#: Plan constant: the number of DSpark draft stages, ruled by the weights
#: (port-assessment.md iteration 3 §2, "stage-count authority ruling") and
#: corroborated at load time by the ``mtp.{0,1,2}`` key census. It is NOT
#: read from ``num_nextn_predict_layers``; see
#: :data:`CONTRADICTED_CHECKPOINT_FIELDS`.
NUM_DSPARK_STAGES: int = 3

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

    # <-- MODEL-SPECIFIC: KV compression schedule. 46 entries for 43 target
    # layers, and the three extra entries are NOT padding: the reference
    # indexes this same list by ``n_layers + stage`` for the DSpark stages
    # (``dsv4_ref/model.py:459`` inside ``Attention.__init__``, reached from
    # ``DSparkBlock(args.n_layers + layer_id, args)`` at ``:902``). The pinned
    # config's tail is ``[..., 4, 0, 0, 0]`` — index 42 is the last target
    # layer and indices 43/44/45 are ratio 0, which is why every DSpark stage
    # is SWA-only (``DSparkAttention.forward`` asserts ``compress_ratio == 0``,
    # ``:753``). 46 == 43 + 3 is therefore a THIRD independent corroboration
    # of the three-stage ruling, alongside the ``mtp.{0,1,2}`` key census and
    # the reference config's ``n_mtp_layers = 3``.
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

    # <-- MODEL-SPECIFIC: the DSpark block-parallel draft model.
    # Every field below is read by :mod:`.dspark_model`; all four are present
    # in the pinned HF ``config.json`` and byte-identical in DeepSeek's own
    # reference config (``dsv4_ref/hf_config.json``, verified at iteration 3).
    #: Tokens drafted per block-parallel step (``dsv4_ref/model.py:854-855``).
    dspark_block_size: int = 5
    #: Filler token id for draft positions 1.. (``dsv4_ref/model.py:855``).
    dspark_noise_token_id: int = 128799
    #: Target layers whose hc-bundle means are concatenated into the drafter's
    #: ``main_proj`` input (``dsv4_ref/model.py:920-925``, :851-853).
    dspark_target_layer_ids: list[int] = field(
        default_factory=lambda: [40, 41, 42]
    )
    #: Rank of the stage-2 Markov head (``dsv4_ref/model.py:795-804``).
    dspark_markov_rank: int = 256

    # <-- CONTRADICTED BY THE WEIGHTS: parsed, never authoritative. See
    # CONTRADICTED_CHECKPOINT_FIELDS and NUM_DSPARK_STAGES above.
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
    # DSpark drafter — derived structure
    # ------------------------------------------------------------------
    @property
    def num_dspark_stages(self) -> int:
        """Number of DSpark draft stages, by the weights (3).

        Deliberately a plan constant rather than
        ``num_nextn_predict_layers``: see
        :data:`CONTRADICTED_CHECKPOINT_FIELDS`.
        """
        return NUM_DSPARK_STAGES

    @property
    def dspark_main_hidden_size(self) -> int:
        """Width of the drafter's ``main_proj`` input.

        The target contributes one hc-bundle MEAN per configured target layer
        and the drafter concatenates them (``dsv4_ref/model.py:851-853``,
        ``:920-925``), so the width is ``hidden_size * len(target_layers)`` =
        12288 for the pinned config — which is exactly the checkpoint's
        ``mtp.0.main_proj.weight`` ``[4096, 12288]``.
        """
        return self.hidden_size * len(self.dspark_target_layer_ids)

    def dspark_block_fp8_linear_plan(
        self, tp_size: int
    ) -> list[tuple[str, int, str, int]]:
        """Block-FP8 linear inventory for ONE DSpark stage.

        Same tuple shape and same ``k_local % 128 == 0`` obligation as
        :meth:`block_fp8_linear_plan`; the factory enforces both lists.

        Two shape facts that differ from the main stack and matter:

        * ``wq_a`` and ``wkv`` ARE fused, exactly as on the main stack, even
          though the checkpoint ships ``mtp.{s}.attn.wq_a`` and
          ``mtp.{s}.attn.wkv`` as separate tensors with separate scales (key
          census §A) and even though the drafter evaluates the ``wkv`` half on
          two different inputs at two different cadences: on ``main_x`` once
          per REAL token (``dsv4_ref/model.py:759``) and on the draft block
          once per BLOCK position (``:778``). Fusing still holds because the
          draft-block path needs BOTH halves on one shared input, which is the
          only place the fused GEMM is actually evaluated as a whole; the
          ``main_x`` path reads the ``wkv`` half as a 512-row slice of the
          fused stack with its matching 4-row scale slice. Row-slicing is a
          static shape and the scale grid slices with it at 128-row
          granularity because ``q_lora_rank`` (1024) is a whole number of
          blocks -- the same argument, and the same precedent, as the main
          stack's ``DeepseekV4Attention._q_latent``.
        * stage 0 adds ``main_proj`` ``[hidden_size, 12288]``, replicated in
          BOTH axes. K replication is forced, not chosen: 12288 / 64 = 192 and
          192 % 128 != 0, so a row shard breaks the block-alignment invariant
          this list exists to check. N stays replicated too -- see the
          PARALLELISM note on
          :func:`~.weight_loaders.attach_dspark_stage_loaders` for why the
          admissible ``N_local = 64`` column shard is declined (it would split a
          128-row scale block and owe an all-gather), and for the recorded
          footprint lever if that 48 MiB/core ever binds.
        """
        return [
            (
                "attn.fused_wqa_wkv",
                self.hidden_size,
                "replicated",
                self.hidden_size,
            ),
            ("attn.wq_b", self.q_lora_rank, "column", self.q_lora_rank),
            ("attn.wo_a", self.kv_lora_rank, "column", self.kv_lora_rank),
            (
                "attn.wo_b",
                self.o_groups * self.o_lora_rank,
                "row",
                (self.o_groups * self.o_lora_rank) // tp_size,
            ),
            (
                "ffn.shared_experts.gate_up_proj",
                self.hidden_size,
                "column",
                self.hidden_size,
            ),
            (
                "ffn.shared_experts.down_proj",
                self.moe_intermediate_size,
                "row",
                self.moe_intermediate_size // self.shared_expert_tp,
            ),
            (
                "main_proj",
                self.dspark_main_hidden_size,
                "replicated",
                self.dspark_main_hidden_size,
            ),
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
