# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 (Flash-0731) BF16/FP8 Implementation
================================================

Annotated implementation of DeepSeek-V4 for the Neuron backend.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    DeepSeek-V4-specific. Change when porting.

WHAT MAKES THIS FAMILY DIFFERENT FROM ``llama3`` / ``gpt_oss``

1. **The residual stream is not a vector, it is a bundle.** DeepSeek-V4
   uses "hyper-connections": the per-token state carried between
   sub-layers is ``(T, hc_mult, hidden_size)`` with ``hc_mult = 4``, not
   ``(T, hidden_size)``. Every sub-layer *collapses* the bundle to one
   vector (``hc_pre``), runs attention or the MoE on that vector, then
   *re-expands* its output back into the bundle (``hc_post``) using a
   per-token, Sinkhorn-balanced ``hc_mult x hc_mult`` mixing matrix. The
   bundle is never RMSNorm'd directly — only the collapsed vector is.
2. **The attention is MLA with a per-layer KV compression class.** Layers
   0 and 1 are sliding-window-only; layers 2..42 alternate between a
   4x-compressed sparse class (which also runs the DSA lightning indexer)
   and a 128x-compressed dense class. See :mod:`.attention`.
3. **Two weight formats coexist.** Block-128x128 FP8 for the attention
   projections and the shared expert, MXFP8 group-32 (upcast at load from
   the checkpoint's MXFP4) for the 256 routed experts. See
   :mod:`.quantization`.

PRIMARY EVIDENCE. The pinned checkpoint repo ships DeepSeek's own
reference implementation (``inference/model.py`` and ``inference/kernel.py``
at revision ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``). Every hc
formula below is transcribed from it, with the reference line cited at the
call site. Where the reference and upstream vLLM 0.21.0 disagree, the
reference wins: it is the implementation the checkpoint was published
with.

THE DRAFT MODEL LIVES IN :mod:`.dspark_model`, NOT HERE. The checkpoint's
``mtp.*`` namespace holds DeepSeek's own "DSpark" block-parallel drafter
(``inference/model.py:818-874``) -- three full decoder stages that draft five
tokens per pass, not the one-extra-layer MTP the port plan first assumed. That
mismatch was a recorded plan defect and has been replanned (ladder rows
LD-18/19/20). What THIS module owes the drafter is exactly two things, both
below:

1. ``SupportsEagle3`` plus the hc-bundle-mean collection at
   ``config.dspark_target_layer_ids``, concatenated on-device into the
   ``[T, 3 * hidden_size]`` tensor DSpark's stage-0 ``main_proj`` consumes.
2. The DECLARATION of the drafter's three sliding-window KV legs
   (:meth:`DeepseekV4ForCausalLM._drafter_kv_layer_specs`) -- declared here so
   they get ``SlidingWindowSpec``, because the runner would wrap anything the
   drafter itself declared in ``FullAttentionSpec`` at ``max_model_len``.

Supported parallelism: TP, EP, plus a shared-expert TP subgroup and an
o-projection group. Sequence parallelism is NOT enabled in this port
(recorded decision: it interacts with the hc bundle's extra axis and buys
nothing for correctness; it is a perf item, not a parity item).
"""

import logging
import math

import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.interfaces import SupportsEagle3

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .attention import (
    _FP8_DTYPE,
    _SWA_PAIR_HEAD_SIZE,
    DeepseekV4Attention,
)
from .config import DeepseekV4Config
from .moe import DeepseekV4MoE
from .weight_loaders import (
    attach_hash_context_loaders,
    build_checkpoint_mappings,
    load_block_scale_buffers,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: fp32 accumulation, bf16 weight, bf16 output.
# =============================================================================


class DeepseekV4RMSNorm(nn.Module):
    """RMSNorm computed in fp32 and returned in the input dtype.

    The reference implementation normalizes in fp32 and multiplies by an
    fp32 view of the weight (``inference/model.py:197-202``). We keep the
    weight in the checkpoint dtype (bf16) to avoid doubling the norm
    parameter footprint on every core, and upcast it per call — the cast
    is on a ``[dim]`` tensor and is free relative to the reduction.
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


# =============================================================================
# Section 2: Rotary embeddings
# <-- MODEL-SPECIFIC: DUAL theta (compressed vs sliding-window layers),
#     YaRN interpolation, and INTERLEAVED pair ordering.
# =============================================================================


class DeepseekV4RotaryEmbedding(nn.Module):
    """Dual-theta YaRN rotary tables for the 64 RoPE dims of the latent.

    Three things here are easy to get wrong and each is sourced:

    1. **Dual theta.** Compressed layers use ``compress_rope_theta``
       (160000) and sliding-window-only layers use ``rope_theta`` (10000)
       — :meth:`DeepseekV4Config.rope_theta_for_layer`. Two independent
       tables are therefore built, and the decoder layer picks by its own
       layer class.
    2. **YaRN is ON.** The pinned config carries
       ``rope_scaling={"type": "yarn", "factor": 16,
       "original_max_position_embeddings": 65536, ...}``. The reference
       applies the frequency interpolation whenever
       ``original_seq_len > 0`` (``inference/model.py:206-235``), so
       skipping it would put every position past 65536 on the wrong
       frequencies — a divergence that grows with context length and
       would be invisible on short prompts.
    3. **Pairs are INTERLEAVED, not half-split.** The reference rotates
       via ``view_as_complex(x.unflatten(-1, (-1, 2)))``
       (``inference/model.py:238-250``), i.e. it pairs adjacent elements
       ``(x0,x1), (x2,x3), ...``. That is the GPT-J convention, NOT the
       GPT-NeoX half-split convention that ``llama3`` in this repo uses.
       Applying the wrong one produces plausible-looking but wrong
       attention, so the pairing lives in one documented place.

    Tables are materialized per forward at the actual positions rather
    than precomputed to ``max_position_embeddings``: a full table would be
    ``1048576 x 32`` per theta per trig function, which is hundreds of MiB
    of weights for something that costs two transcendentals per token.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.rope_dim = config.qk_rope_head_dim
        self.dtype = config.torch_dtype
        # One inv_freq per theta. Registered as non-persistent buffers so
        # they follow the module to device without entering the checkpoint.
        # <-- MODEL-SPECIFIC: YaRN is applied to the COMPRESSED table only.
        # The reference builds the sliding-window table with
        # ``original_seq_len = 0``, which disables the interpolation
        # (``dsv4_ref/model.py:484-485``). That is not an optimization: a
        # sliding-window layer never attends beyond 128 positions, so
        # stretching its frequency basis for a 1048576-position context
        # would de-tune every window layer against the trained model.
        # The table split is 1:1 with the layer class — compressed layers
        # read ``inv_freq_compressed``, SWA-only layers read
        # ``inv_freq_window`` (see the layer loop in
        # :meth:`DeepseekV4Model.forward`).
        self.register_buffer(
            "inv_freq_compressed",
            self._build_inv_freq(config, float(config.compress_rope_theta), yarn=True),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_window",
            self._build_inv_freq(config, float(config.rope_theta), yarn=False),
            persistent=False,
        )

    @staticmethod
    def _build_inv_freq(
        config: DeepseekV4Config, base: float, *, yarn: bool
    ) -> torch.Tensor:
        """Return the (optionally YaRN-interpolated) inverse frequencies.

        Transcribed from ``inference/model.py:206-235``. The YaRN branch
        blends the extended frequencies ``freqs / factor`` with the
        original ``freqs`` through a linear ramp between the ``beta_fast``
        and ``beta_slow`` correction dimensions.

        Args:
            config: Supplies ``qk_rope_head_dim`` and the YaRN knobs.
            base: The theta for this table.
            yarn: Whether to interpolate. Passed explicitly rather than
                inferred from the config, because the same config drives
                both tables and only the compressed one is interpolated
                (``dsv4_ref/model.py:484-485``).
        """
        dim = config.qk_rope_head_dim
        freqs = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        original_seq_len = config.rope_original_seq_len
        if not yarn or original_seq_len <= 0:
            return freqs

        factor = config.rope_factor

        def correction_dim(num_rotations: float) -> float:
            return (
                dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dim(config.rope_beta_fast)), 0)
        high = min(math.ceil(correction_dim(config.rope_beta_slow)), dim - 1)
        if low == high:
            high = high + 0.001
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
        )
        smooth = 1 - ramp
        return freqs / factor * (1 - smooth) + freqs * smooth

    def forward(
        self, positions: torch.Tensor, compressed: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[T, rope_dim // 2]``.

        Args:
            positions: Token positions, ``[T]``.
            compressed: Select the compressed-layer theta when true, the
                sliding-window theta otherwise.
        """
        inv_freq = self.inv_freq_compressed if compressed else self.inv_freq_window
        angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def apply_interleaved(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the last dim of ``x`` pairing ADJACENT elements.

        ``x`` is ``[..., T?, rope_dim]``; ``cos``/``sin`` are
        ``[T, rope_dim // 2]`` and are broadcast against the leading dims.
        This is the reference's complex-multiply written as real
        arithmetic, so it stays traceable and static-shaped.
        """
        pairs = x.float().unflatten(-1, (-1, 2))
        x_even = pairs[..., 0]
        x_odd = pairs[..., 1]
        while cos.dim() < x_even.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


# =============================================================================
# Section 3: Hyper-connections ("hc") — the residual stream itself
# <-- MODEL-SPECIFIC: this replaces the ordinary residual add entirely.
# =============================================================================


class DeepseekV4HashContext:
    """The hyper-connection mixing operators, as pure static functions.

    Naming note: the campaign has been calling this the "hash-context"
    (mhc) machinery; upstream vLLM calls it ``mhc`` and the reference
    calls it "Hyper-Connections". They are the same thing. It is NOT
    related to the ``tid2eid`` vocab hash routing in :mod:`.moe`.

    Three operators, all transcribed from the reference:

    * :meth:`split_sinkhorn` — ``inference/kernel.py:372-438``
    * :meth:`pre` — ``inference/model.py:680-688``
    * :meth:`post` — ``inference/model.py:690-693``
    * :meth:`head` — ``inference/model.py:709-716``

    Everything runs in fp32. The mixing weights are fp32 parameters in
    the checkpoint, and the Sinkhorn normalization is a long chain of
    divisions whose bf16 error would compound over 20 iterations and 43
    layers.
    """

    @staticmethod
    def split_sinkhorn(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the ``(2 + hc) * hc``-wide mix vector into the three gates.

        Returns ``(pre, post, comb)`` with shapes ``[T, hc]``, ``[T, hc]``
        and ``[T, hc, hc]``.

        The ``comb`` matrix is Sinkhorn-balanced: one softmax over the row
        axis (plus ``eps``), one column normalization, then
        ``sinkhorn_iters - 1`` further row+column passes. Total row
        normalizations and total column normalizations are both
        ``sinkhorn_iters`` — the first row pass IS the softmax, not a
        plain sum-normalize. Off-by-one here changes the mixing matrix on
        every token of every layer.

        The loop bound is a Python ``int`` from config, so the loop fully
        unrolls at trace time and the graph stays static-shaped.
        """
        pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
        post = 2.0 * torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        comb = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult) * hc_scale[
            2
        ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)

        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def pre(
        bundle: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse the ``[T, hc, H]`` bundle to ``[T, H]``.

        Also returns the ``post`` and ``comb`` gates that the MATCHING
        :meth:`post` call must consume — they are computed from the
        bundle BEFORE the sub-layer runs, so they must be carried across
        the sub-layer rather than recomputed after it.

        Reference: ``inference/model.py:680-688``. Note the RMS factor
        multiplies ``mixes``, not the bundle: the bundle enters the
        weighted sum un-normalized.
        """
        dtype = bundle.dtype
        flat = bundle.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat, hc_fn) * rsqrt
        pre, post, comb = DeepseekV4HashContext.split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
        )
        collapsed = torch.sum(pre.unsqueeze(-1) * flat.view(bundle.shape), dim=1)
        return collapsed.to(dtype), post, comb

    @staticmethod
    def post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Re-expand a ``[T, H]`` sub-layer output into the bundle.

        ``new[j] = sum_i comb[i, j] * residual[i] + post[j] * x`` — a
        per-token linear recombination of the previous bundle plus the new
        output broadcast across streams with a per-stream gate.

        Reference: ``inference/model.py:690-693``. The sum is over the
        SOURCE stream index (``dim=1`` here, where the reference has a
        leading batch and sequence axis), and transposing it silently
        swaps the mixing matrix for its transpose.
        """
        mixed = torch.sum(
            comb.unsqueeze(-1).float() * residual.unsqueeze(-2).float(), dim=1
        )
        gated = post.unsqueeze(-1).float() * x.unsqueeze(-2).float()
        return (mixed + gated).to(x.dtype)

    @staticmethod
    def head(
        bundle: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Final collapse of the bundle before the norm and the LM head.

        Reference: ``inference/model.py:709-716``. Unlike :meth:`pre`
        there is no ``post``/``comb`` and no Sinkhorn — only the
        ``hc_mult``-wide sigmoid gate and the weighted sum. ``hc_fn`` is
        ``[hc, hc * H]``, ``hc_scale`` is ``[1]``, ``hc_base`` is ``[hc]``.
        """
        dtype = bundle.dtype
        flat = bundle.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
        collapsed = torch.sum(pre.unsqueeze(-1) * flat.view(bundle.shape), dim=1)
        return collapsed.to(dtype)


# =============================================================================
# Section 4: Decoder layer
# =============================================================================


class DeepseekV4DecoderLayer(nn.Module):
    """One transformer block: hc_pre -> norm -> attn -> hc_post, twice.

    The second pass runs the MoE instead of attention. Reference:
    ``inference/model.py:695-707``.

    Upstream vLLM fuses each ``hc_post`` with the following ``hc_pre``
    into one ``mhc_fused_post_pre`` op. That is a kernel-launch
    optimization on a platform where launches dominate; it is
    algebraically identical to the unfused pair, and this port keeps the
    unfused form because a single traced graph has no launch overhead to
    amortize and the unfused form is the one the reference states.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        *,
        oproj_group,
        oproj_group_rank: int,
        oproj_group_size: int,
        shared_group,
        shared_group_rank: int,
        shared_group_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.rms_norm_eps = config.rms_norm_eps

        self.self_attn = DeepseekV4Attention(
            config,
            layer_idx,
            oproj_group=oproj_group,
            oproj_group_rank=oproj_group_rank,
            oproj_group_size=oproj_group_size,
        )
        self.mlp = DeepseekV4MoE(
            config,
            layer_idx,
            shared_group=shared_group,
            shared_group_rank=shared_group_rank,
            shared_group_size=shared_group_size,
        )

        self.attn_norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.ffn_norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        # <-- MODEL-SPECIFIC: hc mixing parameters, fp32, replicated.
        # mix_hc = (2 + hc_mult) * hc_mult = 24 at hc_mult=4: hc_mult
        # "pre" gates, hc_mult "post" gates, hc_mult^2 "comb" entries.
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

        attach_hash_context_loaders(self, config)

    def forward(
        self,
        bundle: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object,
        input_ids: torch.Tensor,
        is_prefill: bool,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the ``[T, hc_mult, H]`` bundle by one block."""
        # ── Attention sub-layer ──────────────────────────────────────────
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
            x, positions, attn_metadata, rope_cos=rope_cos, rope_sin=rope_sin
        )
        bundle = DeepseekV4HashContext.post(x, residual, post, comb)

        # ── MoE sub-layer ────────────────────────────────────────────────
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
        x = self.mlp(x, input_ids, is_prefill)
        return DeepseekV4HashContext.post(x, residual, post, comb)


# =============================================================================
# Section 5: Backbone
# =============================================================================


class DeepseekV4Model(nn.Module):
    """DeepSeek-V4 transformer backbone.

    >>> PARALLELISM <<<
    Three process groups are built ONCE here and handed down, because
    ``torch.distributed.new_group`` is a collective that every rank must
    call in the same order — building them per layer would be 43x the
    collectives for the same groups, and building them lazily inside a
    forward would deadlock.

    * the full TP group (attention heads, row-parallel reductions),
    * an o-projection group of ``tp_size // o_groups`` ranks, over which
      the grouped o-projection's stage-A partial sums are reduced,
    * a shared-expert group of ``config.shared_expert_tp`` ranks, chosen
      so the shared expert's ``down_proj`` keeps ``K_local = 128``,
      exactly one dynamic activation-quantization group.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult

        # >>> PARALLELISM: TP group <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        oproj_group, oproj_rank, oproj_size = self._build_subgroup(
            self.world_size // config.o_groups
        )
        shared_group, shared_rank, shared_size = self._build_subgroup(
            min(config.shared_expert_tp, self.world_size)
        )

        # >>> PARALLELISM: vocab-sharded embedding <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_tp_group.device_group,
        )
        emb_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            is_storage_transposed=False,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            with_rank_override(emb_loader, rank=emb_tp_group.rank_in_group),
        )

        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

        # EAGLE3 TRANSPORT: which layers hand their hc-bundle mean to the DSpark
        # drafter. Empty until the runner calls
        # ``DeepseekV4ForCausalLM.set_aux_hidden_state_layers``, which it does
        # only on a speculative serve; ``forward`` then reads it. This is
        # compile-time graph shape, same class of flag as ``_gather_logits``.
        self.aux_hidden_state_layers: list[int] = []

        # <-- MODEL-SPECIFIC: 43 layers, per-layer KV compression class.
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config,
                    layer_idx,
                    oproj_group=oproj_group,
                    oproj_group_rank=oproj_rank,
                    oproj_group_size=oproj_size,
                    shared_group=shared_group,
                    shared_group_rank=shared_rank,
                    shared_group_size=shared_size,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        # <-- MODEL-SPECIFIC: the final bundle collapse. Note the shapes
        # differ from the per-layer hc parameters: fn is [hc, hc * H] (not
        # [mix_hc, hc * H]), scale is [1] (not [3]), base is [hc].
        hc_dim = self.hc_mult * config.hidden_size
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

    def _build_subgroup(self, group_size: int):
        """Build the contiguous rank subgroup of ``group_size`` this rank is in.

        Every rank iterates every tile and calls ``new_group`` for all of
        them, keeping only the handle for the tile containing itself —
        that is the required collective discipline (see
        ``functional/process_groups.py:create_row_col_groups``, whose raw
        ``dist.new_group`` is the only primitive for an arbitrary
        subgroup in this repo).

        Returns ``(group, rank_in_group, group_size)``, or
        ``(None, 0, 1)`` when distributed is not initialized (CPU-mode
        unit tests) or the group would be the whole world.
        """
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the collapsed, normalized hidden states and the aux states.

        Returns:
            ``(hidden_states [T, H], aux_hidden_states)``. The aux list is
            EMPTY unless :attr:`aux_hidden_state_layers` was set, which only
            happens when a DSpark drafter is configured; the second element then
            holds one ``[T, H]`` tensor per configured layer, in layer order.
        """
        # <-- MODEL-SPECIFIC: read the dispatch off ``.swa``, NOT off the bare
        # ``layers.0.self_attn``. Layer 0 is SWA-only in this model (the
        # checkpoint carries compressor keys on layers 2..42 only), so
        # ``kv_layer_specs`` never declares the bare name there and indexing
        # it raises KeyError on the first forward. ``.swa`` is the one name
        # every layer declares, which is what makes it safe to key off.
        first_layer_name = "layers.0.self_attn.swa"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_threshold

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        # <-- MODEL-SPECIFIC: expand [T, H] -> [T, hc_mult, H]. Every
        # stream starts IDENTICAL — a repeat, not a learned split
        # (inference/model.py:916).
        bundle = hidden_states.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()

        # <-- MODEL-SPECIFIC: dual-theta RoPE. Both tables are built once
        # per forward and each layer takes the one its class needs, so the
        # trig work is not repeated 43 times.
        cos_c, sin_c = self.rotary_emb(positions, compressed=True)
        cos_w, sin_w = self.rotary_emb(positions, compressed=False)

        # <-- MODEL-SPECIFIC / EAGLE3 TRANSPORT: what DSpark's stage-0
        # ``main_proj`` consumes is the hc-bundle MEAN over the hc_mult streams,
        # taken AFTER the layer that produced it and BEFORE the next one -- the
        # reference's ``h.mean(dim=2)`` at each configured layer
        # (``dsv4_ref/model.py:920-925``). It is deliberately NOT the collapsed
        # ``hc_head`` output and NOT one arbitrary stream: the mean is what the
        # drafter was trained against. The list is empty on a non-speculative
        # serve, so the bundle mean is never computed there and the graph shape
        # is unchanged.
        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            compressed = self.config.has_compressed_cache(layer_idx)
            bundle = layer(
                bundle,
                positions,
                attn_metadata,
                input_ids,
                is_prefill,
                rope_cos=cos_c if compressed else cos_w,
                rope_sin=sin_c if compressed else sin_w,
            )
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(bundle.mean(dim=1))

        hidden_states = DeepseekV4HashContext.head(
            bundle,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        return self.norm(hidden_states), aux_hidden_states


# =============================================================================
# Section 6: Language model head + the runner contract
# =============================================================================


class DeepseekV4ForCausalLM(nn.Module, SupportsEagle3):
    """DeepSeek-V4 with its LM head, implementing the runner contract.

    The five members the Neuron model runner requires are :meth:`forward`,
    :meth:`get_kv_spec`, :meth:`bind_kv_cache`, :meth:`load_weights` and
    :meth:`from_configs`.

    It also implements ``SupportsEagle3``, which is how the DSpark drafter gets
    the target hidden states it rides on. DSpark is not Eagle3 -- it is
    DeepSeek's own block-parallel drafter (see :mod:`.dspark_model`) -- but the
    transport it needs is precisely Eagle3's: a set of mid-stack hidden states,
    concatenated on-device into one tensor of width
    ``hidden_size * len(aux_layers)``. Reusing that interface means the runner
    handshake (``supports_eagle3`` -> ``get_eagle3_aux_hidden_state_layers`` ->
    ``set_aux_hidden_state_layers``, ``neuron_model_runner.py:1235-1247``)
    needs no new framework path.

    >>> PARALLELISM: column-parallel LM head <<<
    <-- MODEL-SPECIFIC: embeddings are NOT tied
        (``tie_word_embeddings=false``); the checkpoint ships a separate
        ``head.weight``.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        self.model = DeepseekV4Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_lm_head_dp_group,
            get_neuron_lm_head_tp_group,
        )

        lm_head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_tp_group = lm_head_tp_group
        self.lm_head_dp_group = get_neuron_lm_head_dp_group()
        self.lm_head_dp_size = (
            config.neuron_config.lm_head_dp_size if config.neuron_config else 1
        )

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )

        # <-- c10 CONDITIONAL LOGITS GATHER: this flag is compile-time
        # graph shape, not a runtime option. It is assigned here and read
        # in forward() to guard the vocab-dim all-gather. A forward that
        # omits the branch serves EMPTY logprobs under a correct
        # non-zero-max_logprobs serve configuration, and that is
        # discoverable only after a full compile — so the branch is
        # authored with the model, never bolted on later.
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=lm_head_tp_group.device_group,
        )
        lm_head_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.lm_head.out_features_per_rank,
            num_shards=self.lm_head.tp_size,
            is_storage_transposed=False,
        )
        set_weight_loader(
            self.lm_head.weight,
            with_rank_override(lm_head_loader, rank=lm_head_tp_group.rank_in_group),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_tp_group.device_group,
            )

    # ── Runner contract: from_configs ────────────────────────────────────
    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> "DeepseekV4ForCausalLM":
        return cls(DeepseekV4Config.from_configs(hf_config, neuron_config))

    # ── Runner contract: forward ─────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # >>> PARALLELISM: transition local -> lm_head_dp <<<
        if self.lm_head_dp_size > 1:
            hidden_states_for_logits = self.lm_head_dp_group.all_gather(
                hidden_states_for_logits, dim=0
            )

        logits = self.lm_head(hidden_states_for_logits)

        # <-- c10: the conditional gather. Read of the flag assigned in
        # __init__; see the comment there for why it is compile-time.
        gathered_logits = None
        if self._gather_logits:
            if self.lm_head.gather_output:
                gathered_logits = logits
            else:
                gathered_logits = self.lm_head_tp_group.all_gather(logits, dim=1)

        if self.lm_head_dp_size > 1:
            local_batch = sampling_positions.shape[0]
            dp_rank = self.lm_head_dp_group.rank_in_group
            logits = logits[dp_rank * local_batch : (dp_rank + 1) * local_batch]
            if gathered_logits is not None:
                gathered_logits = gathered_logits[
                    dp_rank * local_batch : (dp_rank + 1) * local_batch
                ]

        # EAGLE3 TRANSPORT: the drafter's stage-0 ``main_proj`` reads ONE
        # concatenated tensor of width ``hidden_size * len(aux_layers)``
        # (12288 for the three configured layers), which is exactly the width
        # the proposer's synthetic-input builder assumes
        # (``spec_decode/eagle.py:155-157`` builds ``hidden_size * 3``). The
        # concat -- not the list -- is the interface, and its dim is the feature
        # dim, matching ``dsv4_ref/model.py:851-853``. The extra return element
        # appears ONLY when aux layers were configured, because the runner
        # unpacks the tuple by arity.
        aux_concat = (
            torch.cat(aux_hidden_states, dim=-1) if aux_hidden_states else None
        )

        if self.on_device_sampling_config is None:
            if aux_concat is not None:
                return logits, aux_concat
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            accepted = rejection_sampler(spec_decode_metadata, sampled_tokens)
            if aux_concat is not None:
                return accepted, aux_concat, gathered_logits
            return accepted

        if aux_concat is not None:
            return sampled_tokens, aux_concat, gathered_logits
        return sampled_tokens, gathered_logits

    # ── Eagle3 transport (the DSpark drafter's hidden-state feed) ────────
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers is not None:
            self.model.aux_hidden_state_layers = list(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """The layers whose hc-bundle means feed the DSpark drafter.

        <-- MODEL-SPECIFIC: NOT the generic ``(2, n//2, n-3)`` heuristic the
        other families return. DSpark was trained against specific target
        layers, recorded in config as ``dspark_target_layer_ids``
        ([40, 41, 42] -- the last three of 43), and the reference reads them
        from its own config rather than deriving them
        (``dsv4_ref/model.py:920-925``). A heuristic here would silently feed
        the drafter states it was never trained on: the shapes would match, the
        compile would succeed, and only the acceptance rate would collapse.
        """
        if self.model.aux_hidden_state_layers:
            return tuple(self.model.aux_hidden_state_layers)
        return tuple(self.config.dspark_target_layer_ids)

    def _drafter_kv_layer_specs(self) -> list[LayerSpec]:
        """The DSpark stages' sliding-window legs, declared BY THE TARGET.

        Deliberately declared here rather than by
        :meth:`~.dspark_model.DeepseekV4DSparkDrafter.get_kv_spec`, which
        returns nothing. The runner wraps every layer a DRAFTER declares in
        ``FullAttentionSpec`` unconditionally
        (``neuron_model_runner.py:7853-7866``), and
        ``FullAttentionSpec.max_memory_usage_bytes`` ignores ``sliding_window``
        -- so a window-128 leg declared there would be sized at
        ``max_model_len``. At the planned 65536 / 32-seq / fp8 configuration
        that turns 12 MiB/core of drafter KV into roughly 6.0 GiB/core and
        breaks the 21.6 GiB/core budget. Declared here they go through the
        target's branch, become ``SlidingWindowSpec``, and are spec-identical to
        the 43 target SWA legs, so they merge into the same KV cache group --
        which is also what makes the proposer's
        ``validate_same_kv_cache_group`` check meaningful rather than vacuous.

        Gated on ``aux_hidden_state_layers`` being set, which is the Eagle3
        handshake's own signal and happens in ``load_model``
        (``neuron_model_runner.py:1247``) -- strictly before
        ``initialize_kv_cache`` calls ``get_kv_spec``. On a non-speculative
        serve the list is empty and no drafter leg is allocated.
        """
        if not self.model.aux_hidden_state_layers:
            return []
        return [
            LayerSpec(
                name=f"mtp.{stage}.self_attn.swa",
                num_kv_heads=1,
                head_size=_SWA_PAIR_HEAD_SIZE,
                dtype=_FP8_DTYPE,
                sliding_window_size=self.config.sliding_window,
                chunk_size=None,
            )
            for stage in range(self.config.num_dspark_stages)
        ]

    # ── Runner contract: KV cache ────────────────────────────────────────
    def get_kv_spec(self) -> KVSpec:
        """Concatenate every attention module's own KV declarations.

        Each :class:`~.attention.DeepseekV4Attention` owns the names and
        widths of the cache pairs it reads and writes, so the spec cannot
        drift from the code that uses it. The DSpark stages' legs are appended
        by :meth:`_drafter_kv_layer_specs`; see there for why the target owns
        that declaration.
        """
        layers = []
        for layer_idx, layer in enumerate(self.model.layers):
            layers.extend(layer.self_attn.kv_layer_specs(layer_idx))
        layers.extend(self._drafter_kv_layer_specs())
        return KVSpec(layers=layers)

    def bind_kv_cache(
        self, kv_caches: dict[str, list[torch.Tensor]]
    ) -> None:
        """Bind the target's own legs only.

        The three ``mtp.*`` legs this class DECLARES are bound by the drafter
        module, which the runner hands the same whole cache dict
        (``neuron_model_runner.py:7784-7791``). Declaring and binding are
        deliberately split: the declaration has to be on this side to get the
        right spec type, the binding has to be on the drafter's side because
        that is the module whose forward reads the tensors.
        """
        for layer in self.model.layers:
            layer.self_attn.bind_kv_cache(kv_caches)

    def expected_kv_layer_names(self) -> list[str]:
        """Derive the expected KV layer names from CONFIG, independently.

        This deliberately does NOT call :meth:`get_kv_spec`. The factory
        cross-checks one against the other, and a check whose two sides
        share a source proves nothing. Here the names come from the
        per-layer compression class in config; there they come from the
        attention modules.
        """
        config = self.config
        names: list[str] = []
        for layer_idx in range(config.num_hidden_layers):
            base = f"layers.{layer_idx}.self_attn"
            if config.has_compressed_cache(layer_idx):
                names.append(base)
                names.append(f"{base}.rope")
            names.append(f"{base}.swa")
            if config.has_indexer(layer_idx):
                names.append(f"{base}.indexer")
        # The DSpark stages' legs, derived from the stage count in config --
        # independently of :meth:`_drafter_kv_layer_specs`, which derives them
        # from the same count but through the drafter's naming. The gate is the
        # same Eagle3 signal, read here too so the two sides agree on WHETHER
        # the legs exist as well as on their names.
        if self.model.aux_hidden_state_layers:
            for stage in range(config.num_dspark_stages):
                names.append(f"mtp.{stage}.self_attn.swa")
        return names

    # ── Runner contract: weight loading ──────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the checkpoint.

        <-- MODEL-SPECIFIC: this checkpoint's keys are NOT HF-standard.
        There is no ``model.`` prefix and no ``self_attn``; the real keys
        are ``layers.N.attn.*``, ``layers.N.ffn.*``, ``embed.weight``,
        ``head.weight``, ``norm.weight``. The whole mapping lives in
        :func:`~.weight_loaders.build_checkpoint_mappings` so there is one
        place to look when a key is missing.
        """
        mappings = build_checkpoint_mappings(
            self.config, len(self.model.layers), mtp=False
        )
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        # <-- QUANTIZATION: hook for any scale tensor the pipelined path
        # cannot carry. ``load_sharded_pipelined`` iterates
        # ``named_parameters`` only, so a scale held as a non-persistent
        # buffer would be silently skipped. Block scales in this family
        # are parameters, so this is expected to be a no-op; it exists so
        # that a future non-persistent scale cannot go missing quietly.
        load_block_scale_buffers(self, checkpoint, self.rank, device)

        self._cast_to_model_dtype(rank_sharded)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Index the checkpoint without loading tensor data (CPU compile).

        Nothing in this family's graph is a load-time constant derived
        from a checkpoint value, so indexing is all this needs to do.
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()

    def _cast_to_model_dtype(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Cast loaded tensors to the destination's dtype where they differ.

        Gate on the DESTINATION parameter's dtype, never the source's:
        fp8 weights must stay fp8 and fp32 scales and hc parameters must
        stay fp32, while a legacy fp32 bf16-destined weight must still be
        cast down.
        """
        destinations = dict(self.named_parameters())
        destinations.update(dict(self.named_buffers()))
        for name, tensor in state_dict.items():
            target = destinations.get(name)
            if target is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.dtype != target.dtype:
                state_dict[name] = tensor.to(target.dtype)
