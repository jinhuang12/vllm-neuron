# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 "DSpark" block-parallel draft model (the ``mtp.*`` namespace).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    DeepSeek-V4-specific. Change when porting.

WHAT DSPARK IS, AND WHAT IT IS NOT

It is NOT upstream vLLM's V3-style MTP (one extra decoder layer that
autoregressively extends the target by one token per pass). The pinned
checkpoint's ``mtp.*`` namespace holds DeepSeek's own **DSpark** drafter
(``dsv4_ref/model.py:750-874``):

* **3 stages**, ``mtp.{0,1,2}``, each a FULL decoder layer on the
  hc-expanded stream -- SWA-only MLA attention plus 256 routed experts plus
  a shared expert plus the whole hyper-connection apparatus. The stage count
  is ruled by the WEIGHTS, not by ``num_nextn_predict_layers`` (see
  :data:`~.config.CONTRADICTED_CHECKPOINT_FIELDS`); three independent
  sources agree -- the ``mtp.{0,1,2}`` key census, the reference config's
  ``n_mtp_layers = 3``, and ``compress_ratios`` having 46 entries for 43
  target layers with the tail ``[0, 0, 0]``.
* **block-parallel, not autoregressive.** ONE forward through all three
  stages drafts ``dspark_block_size = 5`` tokens at once. The block's input
  is the real accepted token at position 0 followed by four copies of
  ``dspark_noise_token_id = 128799`` (``dsv4_ref/model.py:854-855``), and
  attention inside the block is NON-causal: every block query sees every
  window slot AND all five block slots (``get_dspark_topk_idxs``, ``:744``).
* **it rides the target's hidden states.** Its stage-0 ``main_proj`` consumes
  the concatenation of the target's hc-bundle MEANS at layers
  ``dspark_target_layer_ids = [40, 41, 42]`` (``:920-925``, ``:851-853``),
  i.e. ``[T, 3 * hidden_size] = [T, 12288]`` -- which is exactly the width
  the plugin's Eagle3 aux-hidden transport already carries
  (``spec_decode/eagle.py:155-157`` builds ``hidden_size * 3``). That
  coincidence is why DSpark can ride the Eagle3 path at all.

WHAT THIS MODULE REUSES RATHER THAN RESTATES

Every numeric primitive comes from the settled family: :mod:`.attention`'s
``_rms_norm`` / ``_gptj_rope`` / ``_quant_fp8_ue8m0`` / ``_masked_scatter_rows``
/ ``_gather_scale_columns``, :mod:`.model`'s :class:`DeepseekV4HashContext` and
:class:`DeepseekV4RMSNorm`, :class:`~.moe.DeepseekV4MoE` verbatim, and the
functional ops ``NF.mla_qkv`` / ``NF.swa_attention`` / ``NF.block_fp8_linear`` /
``NF.mla_grouped_oproj``. No new functional op and no new NKI kernel is
introduced here (ladder rows LD-18/LD-19/LD-20 are reuse-op, reuse-op and
torch-composition).

RECORDED DEVIATIONS FROM THE REFERENCE (all three affect ACCEPTANCE RATE
only, never emitted output, because the runner verifies every drafted token
against the target with the existing ``RejectionSampler``):

1. **The drafter owns its own ``embed`` / ``head`` copies.** The reference
   assigns the target's objects into each stage
   (``dsv4_ref/model.py:903-904``); two separately compiled NEFFs cannot share
   a parameter tensor, so the drafter loads its own copy of the SAME two
   checkpoint tensors (``embed.weight``, ``head.weight``). Cost +33 MiB/core
   at TP=64 (vocab-sharded 129280 x 4096 bf16, twice), which the assessment's
   0.34 GiB/core drafter line did not price because it recorded "reuses target
   embed/head". Corrected total 15.14 of 21.6 GiB/core -- inside the recorded
   6.5 GiB headroom, so no plan consequence.
2. **``head`` and ``markov_w2`` are bf16, not fp32.** The reference stores both
   as fp32 for "easier computation of logits"
   (``dsv4_ref/model.py:728-729``). The port's TARGET lm_head is already bf16
   (``model.py``'s ``lm_head``), and a drafter head in a different precision
   from the target's would be the only such asymmetry in the port. The
   confidence head IS fp32, as the plan records, because the checkpoint's own
   comment says the fp32 promotion is what the confidence score needs
   (``:810``).
3. **The drafter drafts on the prefill step too.** The reference returns from
   ``forward_spec`` without a head pass when ``start_pos == 0``
   (``dsv4_ref/model.py:933-934``), so it emits no drafts immediately after
   prefill. This port always runs the block step: the graph is compiled once
   per bucket and a second, head-less variant would be a second graph for no
   correctness gain. After a prefill the window cache holds the whole prompt,
   so the block step is exactly the step the reference would take next.
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .attention import (
    _FP8_DTYPE,
    _KV_QUANT_GROUP,
    _cos_sin,
    _gather_cache_rows,
    _gather_scale_columns,
    _gptj_rope,
    _masked_scatter_rows,
    _num_blocks,
    _quant_fp8_ue8m0,
    _rms_norm,
)
from .config import DeepseekV4Config
from .model import (
    DeepseekV4HashContext,
    DeepseekV4RMSNorm,
)
from .moe import DeepseekV4MoE
from .weight_loaders import (
    attach_attention_loaders,
    attach_hash_context_loaders,
    build_checkpoint_mappings,
    load_block_scale_buffers,
)

logger = logging.getLogger(__name__)

#: Number of group-64 UE8M0 scales stored per SWA slot (448 NoPE / 64).
_SWA_NUM_SCALES: int = 7


# =============================================================================
# Section 0: accepted-token extraction
# =============================================================================


def _extract_accepted_tokens(
    input_ids: torch.Tensor,
    sampling_positions: torch.Tensor,
    raw_sampled_token_ids: torch.Tensor,
    vocab_size: int,
    num_speculative_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device ``extract_next_token_ids`` + ``compute_token_indices_to_sample``.

    Transcribed from ``model/llama3/eagle3_model.py:37-91`` (the only
    implementation in the repo) rather than imported from it. Importing would
    couple this family's drafter to llama3's Eagle3 module: a change made for
    an Eagle3 checkpoint would silently change DeepSeek-V4's draft seed token,
    and the shared logic has no home in ``functional/`` or ``nn/`` that could
    be extended without touching a settled family's files. The function is
    twenty lines of index arithmetic with no model-specific content, so the
    duplication is bounded and mechanical.

    The proposer shifts ``input_ids`` by one and leaves the last slot stale on
    purpose (``spec_decode/eagle.py:452-460``); this is where the actually
    sampled token gets patched in, which is why the drafter must not read
    ``input_ids`` at ``sampling_positions`` before calling this.

    Returns:
        ``(patched_input_ids, adjusted_sampling_positions, bonus_token_ids)``.
    """
    valid_mask = (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < vocab_size)
    valid_count = valid_mask.sum(dim=1)
    last_valid_idx = torch.clamp(valid_count - 1, min=0)
    next_token_ids = (
        raw_sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
        .squeeze(1)
        .to(torch.int32)
    )
    next_token_ids = torch.where(
        valid_count > 0, next_token_ids, torch.zeros_like(next_token_ids)
    )

    # Prefill passes one column and needs no rejection adjustment; a
    # steady-state spec step passes ``num_spec + 1`` columns and does.
    if raw_sampled_token_ids.shape[1] > 1:
        num_rejected = torch.clamp(num_speculative_tokens + 1 - valid_count, min=0)
        sampling_positions = torch.clamp(
            sampling_positions - num_rejected.to(sampling_positions.dtype), min=0
        )

    input_ids = input_ids.scatter(0, sampling_positions.to(torch.long), next_token_ids)
    return input_ids, sampling_positions, next_token_ids


def _window_indices(
    positions: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence sliding-window ABSOLUTE positions and their validity.

    ``idx = clamp(pos - window + 1, 0) + arange(window)`` with entries past
    ``pos`` marked absent -- the same band
    ``mla_sparse_attention._window_local_indices`` builds, restated here
    because this module needs the band once per SEQUENCE (all five block
    queries share it) rather than once per query token.

    Returns ``(idx [B, window] int64, valid [B, window] bool)``.
    """
    offsets = torch.arange(window, device=positions.device, dtype=torch.int64)
    base = positions.reshape(-1, 1).to(torch.int64)
    idx = (base - window + 1).clamp_min(0) + offsets.unsqueeze(0)
    return idx, idx <= base


# =============================================================================
# Section 1: DSpark attention (LD-18 -- SWA-only MLA, non-causal block)
# =============================================================================


class DeepseekV4DSparkAttention(nn.Module):
    """One DSpark stage's attention: SWA-only MLA over window + draft block.

    <-- MODEL-SPECIFIC: this is ``dsv4_ref/model.py:750-792``. It differs from
    :class:`~.attention.DeepseekV4Attention` in exactly three ways and is
    otherwise the same parameter set and the same op sequence:

    1. **Two inputs at two cadences.** The KV rows that get CACHED come from
       ``main_x`` -- the projected target hidden state -- one row per REAL
       token (``:759``, ``:783``). The KV rows the block attends in flight
       come from the block input ``x`` and are never cached (``:778``,
       ``:784``: they are concatenated, not stored). So the drafter's window
       holds one slot per accepted token, exactly like a target SWA leg, and
       the five noise-block positions leave no trace.
    2. **A non-causal block.** ``get_dspark_topk_idxs`` (``:744-747``) admits
       every valid window slot plus all five block slots for every one of the
       five queries. Expressed here as ``NF.swa_attention`` with
       ``window = sliding_window + dspark_block_size`` and ``causal=False``:
       the widest admitted gap is the last block query at ``pos + 5`` to the
       oldest window slot at ``pos - 127``, i.e. 132 < 133, and the narrowest
       excluded gap does not exist because the gathered key set contains
       nothing else. Using the same public op as the family's SWA-only prefill
       keeps ONE sink-softmax definition in the port.
    3. **No compressor and no indexer.** ``compress_ratios[43:46] == [0,0,0]``
       and the ``mtp.*`` census carries no compressor or indexer keys, and the
       reference asserts ``compress_ratio == 0`` outright (``:753``).

    >>> PARALLELISM: identical to the main stack -- fused ``wq_a``+``wkv``
    replicated, ``wq_b`` column-sharded over query heads, ``wo_a`` on the
    o-projection subgroup, ``wo_b`` row-parallel over the full TP group with
    this module owning the final all-reduce. <<<
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        stage_idx: int,
        *,
        oproj_group,
        oproj_group_rank: int,
        oproj_group_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        # The loaders need an integer ``layer_idx``; the sharding math needs it
        # to be a real head-band index. Both are satisfied by the main stack's
        # count plus the stage, which is also the index the reference passes
        # (``dsv4_ref/model.py:902``).
        self.layer_idx = config.num_hidden_layers + stage_idx
        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.qk_nope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.sliding_window = config.sliding_window
        self.block_size = config.dspark_block_size

        # <-- MODEL-SPECIFIC: SWA-only, so the base theta with YaRN OFF
        # (``dsv4_ref/model.py:483-487``: the ``else`` branch passes
        # ``original_seq_len = 0``). The same choice ``_rope_params`` makes for
        # the family's SWA-only layers; restated rather than called because
        # ``config.compress_ratio`` would have to be indexed at 43..45 to
        # rediscover a fact the reference asserts.
        self.rope_theta = float(config.rope_theta)
        self.scale = float(self.head_dim) ** -0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if config.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"tensor_parallel_size={self.world_size} does not divide "
                f"num_attention_heads={config.num_attention_heads}."
            )
        self.num_local_heads = config.num_attention_heads // self.world_size

        self.oproj_group = oproj_group
        self.oproj_group_rank = oproj_group_rank
        self.oproj_group_size = oproj_group_size

        q_lora = self.q_lora_rank
        latent = self.head_dim
        heads = self.num_local_heads

        fused_out = q_lora + latent
        self.fused_wqa_wkv_weight = nn.Parameter(
            torch.empty(fused_out, self.hidden_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.fused_wqa_wkv_scale = nn.Parameter(
            torch.empty(
                _num_blocks(fused_out),
                _num_blocks(self.hidden_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        wq_b_out = heads * latent
        self.wq_b_weight = nn.Parameter(
            torch.empty(wq_b_out, q_lora, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.wq_b_scale = nn.Parameter(
            torch.empty(
                _num_blocks(wq_b_out), _num_blocks(q_lora), dtype=torch.float32
            ),
            requires_grad=False,
        )

        wo_a_k_local = heads * latent
        self.wo_a_weight = nn.Parameter(
            torch.empty(self.o_lora_rank, wo_a_k_local, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.wo_a_scale = nn.Parameter(
            torch.empty(
                _num_blocks(self.o_lora_rank),
                _num_blocks(wo_a_k_local),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        wo_b_k_local = (self.o_groups * self.o_lora_rank) // self.world_size
        self.wo_b_weight = nn.Parameter(
            torch.empty(self.hidden_size, wo_b_k_local, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.wo_b_scale = nn.Parameter(
            torch.empty(
                _num_blocks(self.hidden_size),
                _num_blocks(wo_b_k_local),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.q_norm_weight = nn.Parameter(
            torch.empty(q_lora, dtype=self.dtype), requires_grad=False
        )
        self.kv_norm_weight = nn.Parameter(
            torch.empty(latent, dtype=self.dtype), requires_grad=False
        )
        self.attn_sink = nn.Parameter(
            torch.empty(heads, dtype=torch.float32), requires_grad=False
        )

        # Bound externally by :meth:`bind_kv_cache` from the name the TARGET
        # declares; see :meth:`DeepseekV4DSparkDrafter.get_kv_spec`.
        self.swa_k_cache: torch.Tensor | None = None
        self.swa_v_cache: torch.Tensor | None = None

        attach_attention_loaders(
            self,
            config,
            tp_size=self.world_size,
            tp_rank=self.rank,
            group_rank=oproj_group_rank,
            group_size=oproj_group_size,
            shared_tp_size=1,
            shared_tp_rank=0,
            key_prefix=f"mtp.{stage_idx}",
        )

    # ── KV cache ───────────────────────────────────────────────────────────
    @property
    def kv_layer_name(self) -> str:
        """The one KV layer name this stage reads and writes."""
        return f"mtp.{self.stage_idx}.self_attn.swa"

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        name = self.kv_layer_name
        if name not in kv_caches:
            raise KeyError(
                f"KV cache for draft layer {name!r} not initialized. The TARGET "
                "model declares it (DeepseekV4ForCausalLM.get_kv_spec); check "
                "that the target and the drafter agree on the stage count."
            )
        pair = kv_caches[name]
        self.swa_k_cache, self.swa_v_cache = pair[0], pair[1]

    # ── The two cadences ───────────────────────────────────────────────────
    def write_main_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_metadata: dict[str, dict],
    ) -> None:
        """Cache one KV row per REAL token, from ``main_x``.

        ``dsv4_ref/model.py:759-761`` then ``:765-768`` (prefill, whole prompt)
        or ``:783`` (decode, one row at ``start_pos % window``). Both are the
        same statement in the port, because the runner's ``slot_mapping``
        already names the destination slot for every scheduled token and it is
        computed from the UNTRIMMED block table, so it is correct at either
        cadence.

        The ``wkv`` half is read as a 512-row slice of the fused stack with its
        matching 4-row scale slice -- the precedent is
        ``DeepseekV4Attention._q_latent``, which slices the wq_a half the same
        way for the indexer, and it is exact because ``q_lora_rank = 1024`` is a
        whole number of 128-row scale blocks.

        Args:
            main_x: ``[T, hidden_size]`` -- ``main_norm(main_proj(aux))``.
            positions: ``[T]`` int64 absolute positions of the real tokens.
            rope_cos / rope_sin: ``[T, rope_head_dim // 2]`` at those positions.
            attn_metadata: keyed by :attr:`kv_layer_name`.
        """
        md = attn_metadata[self.kv_layer_name]
        rows = self.q_lora_rank

        latent = NF.block_fp8_linear(
            main_x,
            self.fused_wqa_wkv_weight[rows:],
            self.fused_wqa_wkv_scale[_num_blocks(rows) :],
            block_size=(128, 128),
            act_group_size=128,
            accum_dtype=torch.float32,
            out_dtype=torch.bfloat16,
            bias=None,
        )
        latent = _rms_norm(latent, self.kv_norm_weight, self.eps).to(self.dtype)
        latent = _gptj_rope(
            latent.unsqueeze(1), rope_cos, rope_sin, self.rope_head_dim
        ).squeeze(1)

        # <-- The group-64 fp8 QAT round trip (``dsv4_ref/model.py:761``:
        # ``act_quant(..., inplace=True)``, which ``kernel.py:45`` documents as
        # "fused quant+dequant back to BF16" and which therefore IS part of the
        # numerics). Quantizing once serves both the round trip and the cache
        # write: the codes are what the cache stores and their dequant is what
        # the value would have been. Same choice as the family's settled
        # ``kv_nope_fp8_qat=True``.
        nope = self.nope_head_dim
        codes, scales = _quant_fp8_ue8m0(latent[..., :nope])
        _masked_scatter_rows(
            self.swa_k_cache,
            md["slot_mapping"],
            torch.cat((codes.to(torch.float32), latent[..., nope:]), dim=-1),
        )
        _masked_scatter_rows(self.swa_v_cache, md["slot_mapping"], scales)

    def forward(
        self,
        hidden_states: torch.Tensor,
        draft_positions: torch.Tensor,
        real_positions: torch.Tensor,
        attn_metadata: dict[str, dict],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """One block-parallel attention step.

        Args:
            hidden_states: ``[B * block_size, hidden_size]`` -- the block
                stream, already ``attn_norm``-ed by the stage.
            draft_positions: ``[B * block_size]`` int64 absolute positions of
                the drafted slots (``real_pos + 1 .. real_pos + block_size``).
            real_positions: ``[B]`` int64 absolute position of each sequence's
                last accepted token -- the window's right edge.
            attn_metadata: keyed by :attr:`kv_layer_name`.
            rope_cos / rope_sin: ``[B * block_size, rope_head_dim // 2]`` at
                ``draft_positions``.

        Returns:
            ``[B * block_size, hidden_size]``.
        """
        md = attn_metadata[self.kv_layer_name]
        batch = real_positions.reshape(-1).shape[0]
        block = self.block_size
        window = self.sliding_window
        nope = self.nope_head_dim

        # ── q and the in-flight block KV, one fused replicated GEMM ─────────
        q_nope, q_rope, block_latent = NF.mla_qkv(
            hidden_states,
            self.fused_wqa_wkv_weight,
            self.wq_b_weight,
            self.q_norm_weight,
            self.kv_norm_weight,
            rope_cos,
            rope_sin,
            draft_positions,
            wqa_wkv_scale=self.fused_wqa_wkv_scale,
            wqb_scale=self.wq_b_scale,
            eps=self.eps,
            qk_rope_head_dim=self.rope_head_dim,
            apply_q_head_norm=True,
            kv_nope_fp8_qat=True,
            kv_qat_group_size=_KV_QUANT_GROUP,
            # ``_cos_sin`` yields one row per TOKEN; see the same flag at the
            # backbone's call site.
            rope_tables_pregathered=True,
        )
        query = torch.cat((q_nope, q_rope), dim=-1)

        # ── The window, read out of this stage's own paged cache ────────────
        real_pos = real_positions.reshape(-1).to(torch.int64)
        win_abs, win_valid = _window_indices(real_pos, window)

        # The SWA block table is TRIMMED at decode and the runner publishes the
        # window-start offset to address it; the same correction the family's
        # decode path applies (see ``attention._decode_attention``). Positions
        # stay ABSOLUTE for the mask, so the shift is applied to the addressing
        # frame only.
        offset = md.get("swa_kv_pos_offset")
        win_local = win_abs
        if offset is not None:
            win_local = (win_abs - offset.reshape(-1, 1).to(torch.int64)).clamp_min(0)

        block_table = md["block_table_tensor"].to(torch.int64)
        cache_block = md["block_size"]
        slot_ids = (
            torch.gather(block_table, 1, win_local // cache_block) * cache_block
            + win_local % cache_block
        )

        win_row = _gather_cache_rows(self.swa_k_cache, slot_ids, self.head_dim)
        win_factors = _gather_scale_columns(
            self.swa_v_cache, slot_ids, _SWA_NUM_SCALES, _KV_QUANT_GROUP
        )
        win_latent = torch.cat(
            (win_row[..., :nope] * win_factors, win_row[..., nope:]), dim=-1
        )

        # ── One key set per sequence: window slots then block slots ─────────
        keys = torch.cat(
            (win_latent, block_latent.reshape(batch, block, self.head_dim).float()),
            dim=1,
        )
        key_positions = torch.cat(
            (win_abs, draft_positions.reshape(batch, block).to(torch.int64)), dim=1
        )
        key_valid = torch.cat(
            (win_valid, torch.ones_like(win_valid[:, :1]).expand(batch, block)), dim=1
        )
        span = window + block

        seq_ids = torch.arange(batch, device=query.device, dtype=torch.int64)
        attn_out = NF.swa_attention(
            query,
            keys.reshape(batch * span, 1, self.head_dim),
            keys.reshape(batch * span, 1, self.head_dim),
            span,
            self.attn_sink,
            self.scale,
            positions=draft_positions.reshape(-1),
            kv_positions=key_positions.reshape(-1),
            q_seq_ids=seq_ids.repeat_interleave(block),
            kv_seq_ids=seq_ids.repeat_interleave(span),
            kv_valid=key_valid.reshape(-1),
            causal=False,
        )

        # ── Inverse RoPE, then the grouped o-projection ─────────────────────
        attn_out = _gptj_rope(
            attn_out.reshape(
                hidden_states.shape[0], self.num_local_heads, self.head_dim
            ),
            rope_cos,
            rope_sin,
            self.rope_head_dim,
            inverse=True,
        )
        partial = NF.mla_grouped_oproj(
            attn_out,
            self.wo_a_weight,
            self.wo_b_weight,
            self.o_groups,
            self.oproj_group,
            wo_a_scale=self.wo_a_scale,
            wo_b_scale=self.wo_b_scale,
            group_rank=self.oproj_group_rank,
            out_dtype=torch.float32,
        )
        if self.world_size > 1:
            partial = self.tp_group.all_reduce(partial)
        return partial.to(self.dtype)


# =============================================================================
# Section 2: the two extra heads on the last stage (LD-19, LD-20)
# =============================================================================


class DeepseekV4DSparkMarkovHead(nn.Module):
    """Per-position token-conditioned logit bias (``dsv4_ref/model.py:795-804``).

    A rank-``dspark_markov_rank`` bottleneck read as an embedding and written
    back as a head: ``logits_bias = w2(w1[token])``. Both tensors are
    ``[vocab_size, 256]``.

    >>> PARALLELISM: vocab-sharded on both sides -- ``w1`` as the port's
    ``VocabDimShardedEmbedding`` (the reference's ``ParallelEmbedding``,
    ``:798``) and ``w2`` as a column-parallel head with ``gather_output=True``
    (the reference's ``ParallelHead``, whose forward all-gathers the vocab
    axis, ``:736-739``). The bias must be full-vocab before it is added to the
    stage's full-vocab logits. <<<
    """

    def __init__(self, config: DeepseekV4Config, *, embed_tp_group, head_tp_group):
        super().__init__()
        rank_dim = config.dspark_markov_rank
        self.w1 = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=rank_dim,
            dtype=config.torch_dtype,
            tp_group=embed_tp_group.device_group,
        )
        set_weight_loader(
            self.w1.weight,
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.w1.vocab_size_per_rank,
                    num_shards=self.w1.tp_size,
                    is_storage_transposed=False,
                ),
                rank=embed_tp_group.rank_in_group,
            ),
        )

        self.w2 = neuron_nn.ColumnParallelLinear(
            rank_dim,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=True,
            tp_group=head_tp_group.device_group,
        )
        set_weight_loader(
            self.w2.weight,
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.w2.out_features_per_rank,
                    num_shards=self.w2.tp_size,
                    is_storage_transposed=False,
                ),
                rank=head_tp_group.rank_in_group,
            ),
        )

    def forward(
        self, token_ids: torch.Tensor, rank: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits_bias [B, vocab], embed [B, rank_dim])``."""
        embed = self.w1(token_ids, scatter_tokens=False, rank=rank)
        return self.w2(embed), embed


class DeepseekV4DSparkConfidenceHead(nn.Module):
    """fp32 scalar confidence per drafted position (``dsv4_ref/model.py:807-815``).

    ``proj(cat(hidden, markov_embed))`` in fp32. The checkpoint stores ``proj``
    in bf16 and the reference promotes it to fp32 with its own comment saying
    the promotion is what the confidence score needs (``:810``), so this is the
    one place the drafter does NOT follow the target's bf16 convention.

    COMPUTED BUT NOT USED FOR ADMISSION (ladder row LD-20). The runner's
    target-side ``RejectionSampler`` already guarantees output equivalence, so
    confidence-gated draft truncation is a throughput lever, never a
    correctness requirement. It is computed and returned so the lever exists
    without a second authoring pass, and it is replicated (one output column).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        in_dim = config.hidden_size + config.dspark_markov_rank
        self.in_dim = in_dim
        self.proj_weight = nn.Parameter(
            torch.empty(1, in_dim, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """``hidden [B, S, H]``, ``markov_embed [B, S, R]`` -> ``[B, S]`` fp32."""
        joined = torch.cat((hidden.float(), markov_embed.float()), dim=-1)
        return torch.nn.functional.linear(joined, self.proj_weight).squeeze(-1)


# =============================================================================
# Section 3: one DSpark stage
# =============================================================================


class DeepseekV4DSparkStage(nn.Module):
    """One ``mtp.{s}`` block: hc_pre -> norm -> attn -> hc_post, twice.

    Structurally :class:`~.model.DeepseekV4DecoderLayer` with a DSpark
    attention and the stage-conditional extras the reference attaches
    (``dsv4_ref/model.py:822-843``): stage 0 owns ``main_proj``/``main_norm``,
    the last stage owns ``norm``, its own ``hc_head_*`` set, the Markov head
    and the confidence head.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        stage_idx: int,
        *,
        oproj_group,
        oproj_group_rank: int,
        oproj_group_size: int,
        shared_group,
        shared_group_rank: int,
        shared_group_size: int,
        embed_tp_group,
        head_tp_group,
    ) -> None:
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        self.layer_idx = config.num_hidden_layers + stage_idx
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == config.num_dspark_stages - 1
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.rms_norm_eps = config.rms_norm_eps

        self.self_attn = DeepseekV4DSparkAttention(
            config,
            stage_idx,
            oproj_group=oproj_group,
            oproj_group_rank=oproj_group_rank,
            oproj_group_size=oproj_group_size,
        )
        # <-- MODEL-SPECIFIC: ``layer_idx = num_hidden_layers + stage`` is what
        # makes ``is_hash_moe_layer`` False, so every stage routes through
        # ``gate.bias`` + noaux_tc and never through ``tid2eid``. The ``mtp.*``
        # key census agrees: no stage carries a ``tid2eid``.
        self.mlp = DeepseekV4MoE(
            config,
            self.layer_idx,
            shared_group=shared_group,
            shared_group_rank=shared_group_rank,
            shared_group_size=shared_group_size,
            key_prefix=f"mtp.{stage_idx}",
        )

        self.attn_norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.ffn_norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        for attr, shape in (
            ("hc_attn_fn", (mix_hc, hc_dim)),
            ("hc_ffn_fn", (mix_hc, hc_dim)),
            ("hc_attn_base", (mix_hc,)),
            ("hc_ffn_base", (mix_hc,)),
            ("hc_attn_scale", (3,)),
            ("hc_ffn_scale", (3,)),
        ):
            setattr(
                self,
                attr,
                nn.Parameter(
                    torch.empty(*shape, dtype=torch.float32), requires_grad=False
                ),
            )

        if self.is_first:
            # <-- MODEL-SPECIFIC: ``main_proj`` [hidden, 3 * hidden] block-FP8,
            # then ``main_norm`` (``dsv4_ref/model.py:832-833``).
            # >>> PARALLELISM: column-parallel in N per LD-18 (N_local = 64 at
            # TP=64), K replicated because 12288 / 64 = 192 is not a multiple of
            # 128. This module owns the all-gather; see
            # ``attach_dspark_stage_loaders`` for why the N shard is admissible
            # despite being half a scale block. <<<
            main_in = config.dspark_main_hidden_size
            world = self.self_attn.world_size
            if config.hidden_size % world != 0:
                raise ValueError(
                    f"tensor_parallel_size={world} does not divide "
                    f"hidden_size={config.hidden_size}; main_proj's column "
                    "shard would not tile."
                )
            self.main_proj_n_local = config.hidden_size // world
            self.main_proj_weight = nn.Parameter(
                torch.empty(self.main_proj_n_local, main_in, dtype=_FP8_DTYPE),
                requires_grad=False,
            )
            self.main_proj_scale = nn.Parameter(
                torch.empty(
                    _num_blocks(self.main_proj_n_local),
                    _num_blocks(main_in),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.main_norm = DeepseekV4RMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )

        if self.is_last:
            self.norm = DeepseekV4RMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )
            self.markov_head = DeepseekV4DSparkMarkovHead(
                config, embed_tp_group=embed_tp_group, head_tp_group=head_tp_group
            )
            self.confidence_head = DeepseekV4DSparkConfidenceHead(config)

        # The per-stage hc mixes, the block norms AND (on the last stage) the
        # hc_head set all come from the ``mtp.{s}`` namespace.
        attach_hash_context_loaders(
            self,
            config,
            layer_idx=self.layer_idx,
            key_prefix=f"mtp.{stage_idx}",
        )
        self._attach_stage_loaders()

    def _attach_stage_loaders(self) -> None:
        """Bind the stage-conditional tensors' checkpoint loaders.

        ``main_proj`` is block-FP8 with a 128x128 scale grid and is column
        parallel in N (``N_local = hidden_size / tp_size``, 64 at TP=64) with K
        replicated; ``confidence_head.proj`` is replicated. Both go through the
        same loader factories the family's attention uses, so there is one
        implementation of the block-FP8 transform.
        """
        from .weight_loaders import attach_dspark_stage_loaders

        attach_dspark_stage_loaders(
            self,
            self.config,
            stage_idx=self.stage_idx,
            tp_size=self.self_attn.world_size,
            tp_rank=self.self_attn.rank,
        )

    def forward(
        self,
        bundle: torch.Tensor,
        draft_positions: torch.Tensor,
        real_positions: torch.Tensor,
        attn_metadata: dict[str, dict],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the ``[B * block_size, hc_mult, H]`` bundle by one stage."""
        residual = bundle
        x, post, comb = DeepseekV4HashContext.pre(
            bundle,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )
        x = self.attn_norm(x)
        x = self.self_attn(
            x, draft_positions, real_positions, attn_metadata, rope_cos, rope_sin
        )
        bundle = DeepseekV4HashContext.post(x, residual, post, comb)

        residual = bundle
        x, post, comb = DeepseekV4HashContext.pre(
            bundle,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )
        x = self.ffn_norm(x)
        # ``is_prefill=False``: the drafter's MoE only ever runs on the
        # block-parallel step, which is a decode-cadence step of
        # ``B * block_size`` tokens. The reference's prefill branch returns
        # before the FFN entirely (``dsv4_ref/model.py:846-849``).
        x = self.mlp(x, input_ids, False)
        return DeepseekV4HashContext.post(x, residual, post, comb)

    # ── Stage-0 entry: project the target's aux hidden states ──────────────
    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``main_norm(main_proj(main_hidden))`` (``dsv4_ref/model.py:853``).

        The GEMM is column-sharded (LD-18), so this method owes the all-gather
        that turns the ``[T, N_local]`` partial back into the REPLICATED
        ``[T, hidden_size]`` stream every downstream consumer assumes --
        ``NF.mla_qkv``'s ``hidden`` is documented replicated, and the hc bundle
        the stages carry is replicated too. Gathering along the feature axis in
        rank order is exactly the inverse of the loader's row shard
        (core ``r`` owns output columns ``[64r, 64r + 64)``).

        Args:
            main_hidden: ``[T, 3 * hidden_size]`` -- the target's concatenated
                hc-bundle means at ``dspark_target_layer_ids``. Replicated.

        Returns:
            ``[T, hidden_size]``, replicated.
        """
        if not self.is_first:
            raise RuntimeError(
                "main_proj lives on DSpark stage 0 only "
                f"(this is stage {self.stage_idx})."
            )
        partial = NF.block_fp8_linear(
            main_hidden,
            self.main_proj_weight,
            self.main_proj_scale,
            block_size=(128, 128),
            act_group_size=128,
            accum_dtype=torch.float32,
            out_dtype=self.config.torch_dtype,
            bias=None,
        )
        if self.self_attn.world_size > 1:
            partial = self.self_attn.tp_group.all_gather(partial, dim=1)
        return self.main_norm(partial)

    # ── Last-stage exit: the block head ────────────────────────────────────
    def forward_head(
        self,
        bundle: torch.Tensor,
        seed_token_ids: torch.Tensor,
        lm_head: nn.Module,
        batch: int,
        rank: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse the bundle and draft ``block_size`` tokens.

        ``dsv4_ref/model.py:860-874`` exactly: the hc_head collapse, the
        stage's own ``norm``, full-vocab logits, then a STATIC
        ``block_size``-iteration loop where drafted token ``i`` conditions the
        Markov bias for position ``i`` and greedy-argmax of that biased row
        produces drafted token ``i + 1``. The loop bound is a python int from
        config, so it fully unrolls at trace time.

        Greedy rather than the reference's temperature sampling: the proposer
        forces greedy draft sampling anyway
        (``spec_decode/eagle.py:335-337``), and the reference's own
        ``sample()`` reduces to ``argmax`` at temperature 0
        (``dsv4_ref/model.py:942-943``). Draft sampling policy affects
        acceptance rate only.

        Args:
            bundle: ``[B * block_size, hc_mult, H]``.
            seed_token_ids: ``[B]`` int32 -- the real accepted token that
                occupies block position 0.
            lm_head: the drafter's own column-parallel head, gathering the
                vocab axis.
            batch: ``B``, a python int (the block table's row count).
            rank: TP rank tensor for the sharded embedding lookup.

        Returns:
            ``(drafts [B, block_size] int32, confidence [B, block_size] fp32)``.
        """
        if not self.is_last:
            raise RuntimeError(
                "the DSpark block head lives on the last stage only "
                f"(this is stage {self.stage_idx})."
            )
        block = self.config.dspark_block_size

        collapsed = DeepseekV4HashContext.head(
            bundle,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        logits = lm_head(self.norm(collapsed)).float()
        logits = logits.reshape(batch, block, -1)

        current = seed_token_ids.to(torch.int32)
        drafts: list[torch.Tensor] = []
        embeds: list[torch.Tensor] = []
        for i in range(block):
            bias, embed = self.markov_head(current, rank)
            biased = logits[:, i] + bias.float()
            current = torch.argmax(biased, dim=-1).to(torch.int32)
            drafts.append(current)
            embeds.append(embed)

        # ``torch.stack`` rather than a slice of a preallocated buffer: a
        # sliced view is non-contiguous on device and the runner needs a
        # contiguous drafts tensor (``model/llama3/eagle3_model.py:726-732``).
        drafted = torch.stack(drafts, dim=1)
        confidence = self.confidence_head(
            collapsed.reshape(batch, block, -1), torch.stack(embeds, dim=1)
        )
        return drafted, confidence


# =============================================================================
# Section 4: the drafter (the proposer's model contract)
# =============================================================================


class DeepseekV4DSparkDrafter(nn.Module):
    """DeepSeek-V4-Flash's DSpark drafter, as the Neuron proposer's draft model.

    The proposer calls this module with the fixed Eagle3 draft-model kwargs and
    unpacks a 3-tuple (``spec_decode/eagle.py:496-508``), so the surface here
    is that call shape and nothing else:

    * :meth:`forward` -> ``(stacked_tokens, drafts_only, logits_or_None)``
    * :meth:`from_configs` with the 3-kwarg factory signature
    * :meth:`load_weights` / :meth:`load_weights_lite`
    * :meth:`get_kv_spec` / :meth:`bind_kv_cache`
    * ``self.config.hidden_size`` (read for the synthetic warmup inputs)
    * ``self.num_speculative_tokens`` (assigned by the proposer)

    >>> PARALLELISM: the drafter shares the target's TP world and its
    embedding / lm-head groups, and builds its own o-projection and
    shared-expert subgroups the same way the backbone does. Subgroups are
    built ONCE here, never per stage: ``new_group`` is a collective and every
    rank must call it in the same order. <<<
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        # Overwritten by the proposer with the speculative config's value; the
        # factory validates that the two agree.
        self.num_speculative_tokens = config.dspark_block_size

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        oproj_group, oproj_rank, oproj_size = self._build_subgroup(
            self.world_size // config.o_groups
        )
        shared_group, shared_rank, shared_size = self._build_subgroup(
            min(config.shared_expert_tp, self.world_size)
        )

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
            get_neuron_lm_head_tp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_tp_group = head_tp_group

        # See the module docstring, deviation 1: the drafter cannot share the
        # target's tensors across two NEFFs, so it loads its own copy of the
        # same two checkpoint tensors.
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.embed_tokens.vocab_size_per_rank,
                    num_shards=self.embed_tokens.tp_size,
                    is_storage_transposed=False,
                ),
                rank=emb_tp_group.rank_in_group,
            ),
        )

        # ``gather_output=True``: the Markov bias is added to FULL-vocab logits
        # before the argmax (``dsv4_ref/model.py:863-871``), so the vocab axis
        # must be gathered here rather than left sharded for an on-device
        # sampler.
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=True,
            tp_group=head_tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.lm_head.out_features_per_rank,
                    num_shards=self.lm_head.tp_size,
                    is_storage_transposed=False,
                ),
                rank=head_tp_group.rank_in_group,
            ),
        )

        self.stages = nn.ModuleList(
            [
                DeepseekV4DSparkStage(
                    config,
                    stage_idx,
                    oproj_group=oproj_group,
                    oproj_group_rank=oproj_rank,
                    oproj_group_size=oproj_size,
                    shared_group=shared_group,
                    shared_group_rank=shared_rank,
                    shared_group_size=shared_size,
                    embed_tp_group=emb_tp_group,
                    head_tp_group=head_tp_group,
                )
                for stage_idx in range(config.num_dspark_stages)
            ]
        )

    def _build_subgroup(self, group_size: int):
        """Build the contiguous rank subgroup of ``group_size`` this rank is in.

        Same collective discipline as
        ``DeepseekV4Model._build_subgroup``: every rank iterates every tile and
        calls ``new_group`` for all of them, keeping only its own handle.
        """
        import torch.distributed as dist

        if group_size <= 1 or not dist.is_initialized():
            return None, 0, max(group_size, 1)
        if group_size == self.world_size:
            return self.tp_group.device_group, self.rank, self.world_size
        if self.world_size % group_size != 0:
            raise ValueError(
                f"subgroup size {group_size} does not divide "
                f"tensor_parallel_size {self.world_size}; the subgroups would "
                "not tile the TP group."
            )
        my_group = None
        my_rank = 0
        for start in range(0, self.world_size, group_size):
            ranks = list(range(start, start + group_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                my_group = group
                my_rank = ranks.index(self.rank)
        return my_group, my_rank, group_size

    # ── Runner contract: from_configs (3-kwarg draft-model factory) ─────────
    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> "DeepseekV4DSparkDrafter":
        """Build the drafter from the DRAFT hf config.

        ``start_layer_idx`` is the proposer's
        ``target_hf_config.num_hidden_layers``
        (``spec_decode/eagle.py:106-110``). It must equal
        ``config.num_hidden_layers`` for this checkpoint -- target and drafter
        share one config file -- and disagreement means the draft config was
        rewritten under us (upstream's ``hf_config_override`` does exactly that
        for MTP, ``vllm/config/speculative.py:312-317``), so it is checked
        rather than trusted.
        """
        dsv4_config = DeepseekV4Config.from_configs(config, neuron_config)
        if start_layer_idx != dsv4_config.num_hidden_layers:
            raise ValueError(
                "DSpark drafter: start_layer_idx "
                f"({start_layer_idx}) must equal the target's "
                f"num_hidden_layers ({dsv4_config.num_hidden_layers}); the "
                "draft config appears to have been rewritten."
            )
        return cls(dsv4_config)

    # ── Runner contract: KV cache ──────────────────────────────────────────
    def get_kv_spec(self) -> KVSpec:
        """Declare NOTHING, deliberately.

        The drafter's three window legs are declared by the TARGET
        (``DeepseekV4ForCausalLM.get_kv_spec``), not here, and this is a
        correctness requirement rather than a style choice: the runner wraps
        EVERY layer a drafter declares in ``FullAttentionSpec``
        (``neuron_model_runner.py:7853-7866``), whose
        ``max_memory_usage_bytes`` ignores ``sliding_window`` and therefore
        sizes a window-128 leg at ``max_model_len``. At the planned
        65536 / 32-seq / fp8 configuration that turns 12 MiB/core of drafter KV
        into roughly 6.0 GiB/core and pushes total residency past the 21.6
        GiB/core budget. Declared by the target they become
        ``SlidingWindowSpec`` legs, window-bounded and spec-identical to the 43
        target SWA legs, so they merge into the same KV cache group -- which is
        also what makes ``EagleProposer.validate_same_kv_cache_group`` a
        meaningful check on ``DSparkProposer.attn_layer_names`` instead of a
        vacuous one.
        """
        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Bind each stage's window leg out of the full cache dict.

        The runner hands the drafter the WHOLE dict
        (``neuron_model_runner.py:7784-7791``), so the legs the target declared
        are reachable here by name.
        """
        for stage in self.stages:
            stage.self_attn.bind_kv_cache(kv_caches)

    def expected_kv_layer_names(self) -> list[str]:
        """The three names, derived from config -- the proposer reads these."""
        return [
            f"mtp.{s}.self_attn.swa" for s in range(self.config.num_dspark_stages)
        ]

    # ── Runner contract: forward ───────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        initial_target_hidden_states: torch.Tensor,
        attn_metadata: dict[str, dict] | None = None,
        sampling_positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Fill the window caches for this step, then draft one block.

        Args:
            input_ids: ``[T]`` target token ids, already shifted by one by the
                proposer with the last slot left stale for patching.
            positions: ``[T]`` positions of the target's scheduled tokens.
            initial_target_hidden_states: ``[T, 3 * hidden_size]`` -- the
                target's hc-bundle means at ``dspark_target_layer_ids``,
                concatenated.
            attn_metadata: keyed by the three ``mtp.{s}.self_attn.swa`` names.
            sampling_positions: ``[B]`` index of each request's last token.
            rank: TP rank tensor.
            raw_sampled_token_ids: ``[B, 1]`` after prefill or
                ``[B, num_spec + 1]`` in steady state, ``-1`` for rejected.
            prev_sampled_token_ids / prev_num_draft_tokens /
            req_indices_per_token: async spec-decode correction inputs; the
                proposer always supplies them (no-op tensors when there is no
                prior spec step).

        Returns:
            ``(stacked_tokens [B, 1 + block], drafts_only [B, block], None)``.
            Column 0 of ``stacked_tokens`` is this step's bonus token, which is
            what makes the tuple directly consumable as the next target NEFF's
            ``input_ids`` (``model/llama3/eagle3_model.py:719-732``).
        """
        # ── Async spec-decode correction (positions + every slot_mapping) ───
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )

        # ── Patch in the actually accepted token ────────────────────────────
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus_token_ids = _extract_accepted_tokens(
                input_ids,
                sampling_positions,
                raw_sampled_token_ids,
                self.config.vocab_size,
                self.num_speculative_tokens,
            )
        else:
            bonus_token_ids = None

        positions_l = positions.reshape(-1).to(torch.int64)
        first_stage = self.stages[0]
        last_stage = self.stages[-1]

        # ── Cache-fill pass: one window row per REAL token, every stage ─────
        # ``main_x`` is stage 0's projection but is consumed by ALL stages'
        # attention (``dsv4_ref/model.py:930-932`` passes one ``main_x`` into
        # every stage), which is why it is computed once here.
        main_x = first_stage.project_main_hidden(initial_target_hidden_states)
        cos_main, sin_main = _cos_sin(
            positions_l,
            self.config.qk_rope_head_dim,
            self.stages[0].self_attn.rope_theta,
            0,
            1.0,
            32.0,
            1.0,
        )
        for stage in self.stages:
            stage.self_attn.write_main_kv(
                main_x, positions_l, cos_main, sin_main, attn_metadata
            )

        # ── Block-parallel draft step ───────────────────────────────────────
        last_idx = sampling_positions.reshape(-1).to(torch.int64)
        batch = last_idx.shape[0]
        block = self.block_size

        seed = torch.index_select(input_ids, 0, last_idx).to(torch.int32)
        real_pos = torch.index_select(positions_l, 0, last_idx)

        # Block position 0 carries the real token; 1.. carry the noise filler
        # (``dsv4_ref/model.py:854-855``). Built by concatenation rather than
        # an in-place column assignment so the shape stays static.
        noise = torch.full(
            (batch, block - 1),
            self.noise_token_id,
            dtype=seed.dtype,
            device=seed.device,
        )
        draft_ids = torch.cat((seed.unsqueeze(1), noise), dim=1).reshape(-1)

        draft_pos = (
            real_pos.unsqueeze(1)
            + torch.arange(1, block + 1, device=real_pos.device, dtype=torch.int64)
        ).reshape(-1)
        cos_draft, sin_draft = _cos_sin(
            draft_pos,
            self.config.qk_rope_head_dim,
            self.stages[0].self_attn.rope_theta,
            0,
            1.0,
            32.0,
            1.0,
        )

        # <-- MODEL-SPECIFIC: expand [T, H] -> [T, hc_mult, H] by REPEAT; every
        # stream starts identical (``dsv4_ref/model.py:857``).
        embedded = self.embed_tokens(draft_ids, scatter_tokens=False, rank=rank)
        bundle = embedded.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()

        for stage in self.stages:
            bundle = stage(
                bundle,
                draft_pos,
                real_pos,
                attn_metadata,
                cos_draft,
                sin_draft,
                draft_ids,
            )

        drafts_only, _confidence = last_stage.forward_head(
            bundle, seed, self.lm_head, batch, rank
        )

        if bonus_token_ids is None:
            return drafts_only, drafts_only, None
        stacked = torch.cat((bonus_token_ids.unsqueeze(1), drafts_only), dim=1)
        return stacked, drafts_only, None

    # ── Runner contract: weight loading ────────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load the ``mtp.*`` namespace plus ``embed.weight`` / ``head.weight``."""
        mappings = build_checkpoint_mappings(
            self.config, self.config.num_hidden_layers, mtp=True, prefix=""
        )
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict
        load_block_scale_buffers(self, checkpoint, self.rank, device)
        self._cast_to_model_dtype(rank_sharded)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Index the checkpoint without loading tensor data (CPU compile)."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()

    def _cast_to_model_dtype(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Cast to the DESTINATION's dtype, never the source's.

        Same rule as the target's loader: fp8 weights stay fp8, fp32 scales,
        hc parameters and the confidence projection stay fp32, and a bf16
        destination still gets its cast.
        """
        destinations = dict(self.named_parameters())
        destinations.update(dict(self.named_buffers()))
        for name, tensor in state_dict.items():
            target = destinations.get(name)
            if target is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.dtype != target.dtype:
                state_dict[name] = tensor.to(target.dtype)
