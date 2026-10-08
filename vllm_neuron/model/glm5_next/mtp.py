"""glm-5.3-Flash multi-token-prediction (MTP) draft head: the checkpoint's layer 45.

The head IS the draft layer. The checkpoint ships one MTP layer
(``num_nextn_predict_layers == 1``) at ``model.language_model.layers.45.*``: a full
decoder layer -- sparse attention with its indexer, the 288-expert bank, the shared
expert and both layer norms, exactly as layer 43 ships them, with no
multi-hyper-connection tensor -- plus four tensors of its own: ``enorm.weight``,
``hnorm.weight``, ``eh_proj.weight`` and ``shared_head.norm.weight``. This module owns
the four and the arithmetic around the layer:

* the embedding of the next token, masked at absolute position 0 (no previous
  token to draft from), normalised by ``enorm``; the trunk's post-final-norm hidden
  row normalised by ``hnorm``; the two concatenated (embedding first) and projected
  from ``2H`` to ``H`` by ``eh_proj``;
* the decoder block: the attention half through ``Glm5NextDSALayer.forward`` (which
  writes layer 45's latent row, pooled store and tail ring) and the feed-forward
  half through the trunk's own ``Glm5NextModel._ffn_half`` (post-attention norm,
  router, routed bank, shared expert, the one all-reduce) with the plain residual
  add -- the block carries none of the six mHC leaves, so the one-stream route is
  the right one and ``_mhc_site`` agrees;
* the shared-head norm, whose output is both the hidden state the next draft
  iteration feeds to ``hnorm`` (upstream GLM5 returns the normed vector as both of
  its outputs) and the row the head projects;
* the greedy token, TP-correct: this rank's vocab-shard logits, its ``(max, argmax)``
  pair, one all-gather over the group, and the global id of the winning rank
  (``functional/draft_token.py``).

Two entry points, the Stage A contract (C3):

* :meth:`Glm5NextMultiTokenPredictor.populate` -- ``T`` rows of the prefill (or a
  real decode step): row ``t`` consumes ``h_t`` and ``embed(x_{t+1})`` and writes
  layer 45's state at position ``t``. The attention half alone writes state, so the
  MoE half is not run.
* :meth:`Glm5NextMultiTokenPredictor.draft_tokens` -- ``k`` unrolled iterations in
  one graph from ``[B, H]`` rows and the ``B`` sampled ids: iteration 0 populates
  position ``p`` and drafts; iteration ``i`` feeds iteration ``i - 1``'s token and
  normed hidden state at position ``p + i``, writes layer 45's latent row (and
  pooled store) there -- rewritten by position when the trunk really reaches it --
  and runs the tail ring on a scratch copy, because the real ring is what one real
  step at ``p`` leaves and a 4-row ring wraps inside a 5-iteration draft. Returns
  ``[B, k]`` int32 global ids.

What the head does not own: construction of ``self.mtp`` on the root and the weight
map of ``layers.45.*`` onto :data:`MTP_PARAMETER_NAMES` (the loader); layer 45's
latent bank, pooled store and tail ring, which arrive in ``**block_kwargs`` from the
same per-layer lookup the trunk uses for layer 43, at index 45 (the runner); the
decision to call either method (the root's forward). The knob that turns all of it
on, ``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT``, is defined once in ``envs.py`` and read
once, by :func:`shadow_draft_k`.

Importing this module loads no model tree: the block is built inside the
constructor, from a lazy import, so the root can import the head and the head can
import the root's classes without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .config import Glm5NextTextConfig

#: The knob (Stage A contract C1). Its one definition is ``envs.py``'s; this is the
#: name error messages and tests spell.
SHADOW_DRAFT_ENV = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"

#: The attribute the decoder block hangs on. The weight map spells the block's
#: parameters ``mtp.block.<...>``; the head's own four hang flat on ``mtp``.
BLOCK_ATTR = "block"

#: The four tensors layer 45 ships beyond a decoder layer, as this tree flattens
#: the checkpoint's paths: ``enorm.weight`` -> ``enorm_weight``,
#: ``shared_head.norm.weight`` -> ``shared_head_norm_weight``.
HEAD_PARAMETER_NAMES: tuple[str, ...] = (
    "enorm_weight",
    "hnorm_weight",
    "eh_proj_weight",
    "shared_head_norm_weight",
)

#: Stage A contract C2: every parameter path the head declares, relative to the
#: head (``mtp.`` on the root). The loader maps ``layers.45.*`` onto exactly these.
#: The block's paths are a layer-43 DSA layer's minus the six ``hc_*`` leaves, which
#: the checkpoint does not give layer 45 and the block therefore never declares.
#: Pinned against the tree by ``test_mtp_draft.py``; a literal rather than a walk so
#: the contract reads without building a module.
MTP_PARAMETER_NAMES: tuple[str, ...] = (
    *HEAD_PARAMETER_NAMES,
    "block.input_layernorm_weight",
    "block.post_attention_layernorm_weight",
    "block.self_attn.q_a_proj_weight",
    "block.self_attn.q_b_proj_weight",
    "block.self_attn.kv_a_proj_with_mqa_weight",
    "block.self_attn.kv_b_proj_weight",
    "block.self_attn.o_proj_weight",
    "block.self_attn.q_a_layernorm_weight",
    "block.self_attn.kv_a_layernorm_weight",
    "block.self_attn.q_a_proj_weight_scale_inv",
    "block.self_attn.q_b_proj_weight_scale_inv",
    "block.self_attn.kv_a_proj_with_mqa_weight_scale_inv",
    "block.self_attn.o_proj_weight_scale_inv",
    "block.self_attn.indexer.wq_b_weight",
    "block.self_attn.indexer.wk_weight",
    "block.self_attn.indexer.k_norm_weight",
    "block.self_attn.indexer.weights_proj_weight",
    "block.self_attn.indexer.k_norm_bias",
    "block.self_attn.indexer.index_kpool_compress_ape",
    "block.self_attn.indexer.index_kpool_compress_gate",
    "block.mlp.experts.router_weight",
    "block.mlp.experts.router_bias",
    "block.mlp.experts.gate_proj_weight",
    "block.mlp.experts.up_proj_weight",
    "block.mlp.experts.down_proj_weight",
    "block.mlp.shared_experts.gate_proj_weight",
    "block.mlp.shared_experts.up_proj_weight",
    "block.mlp.shared_experts.down_proj_weight",
)

#: The block keywords that select the prefill leg. ``draft_tokens`` refuses them: a
#: draft is a decode step, and the prefill carrier's ``[T]``-wide slot mapping has
#: no meaning for a position the draft advances one at a time.
_PREFILL_LEG_KEYWORDS = ("slot_mapping", "prefill_tail", "prefill_end_position")


def shadow_draft_k() -> int:
    """The shadow draft's iteration count ``k``; 0 means off.

    The one reader of :data:`SHADOW_DRAFT_ENV` (contract C1): it goes through
    ``envs``, which holds the knob's one definition. Unset is 0. A negative value is
    refused by name rather than clamped, because a clamped knob would run a
    different draft than the one asked for and the alpha it measures would be
    labelled with the wrong k. Any positive k runs that many iterations; the
    iterations past the request's window clamp to its last position
    (``Glm5NextMultiTokenPredictor._iteration_carrier``), so no k indexes state
    out of bounds.
    """
    from vllm_neuron import envs

    value = int(envs.VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT)
    if value < 0:
        raise ValueError(
            f"{SHADOW_DRAFT_ENV}={value} is negative; 0 turns the shadow draft off "
            f"and k >= 1 runs that many draft iterations per step"
        )
    return value


def _declare_parameters(module: nn.Module, *names: str) -> None:
    """Reserve parameter attribute paths on ``module`` without allocating.

    The same body as ``model_fp8._declare_parameters``, duplicated rather than
    imported: that helper is private to its module, and importing it here would
    make the model tree an import-time dependency of the head, which the lazy
    block construction exists to avoid.
    """
    for name in names:
        module.register_parameter(name, None)
    module.declared_param_names = (
        *getattr(module, "declared_param_names", ()),
        *names,
    )


def _advanced(value: torch.Tensor | int, by: int, limit: int) -> torch.Tensor | int:
    """``min(value + by, limit)`` for a position that is a tensor on the traced path or an int."""
    if torch.is_tensor(value):
        return torch.clamp(value + int(by), max=int(limit))
    return min(int(value) + int(by), int(limit))


class Glm5NextMultiTokenPredictor(nn.Module):
    """The draft head: layer 45 and the four tensors around it (see the module doc).

    Args:
        text_config: the decoder config. The one draft layer's index is
            ``num_hidden_layers + num_nextn_predict_layers - 1`` (one past the
            stack); the norm epsilon, the vocabulary and
            ``index_share_for_mtp_iteration`` are read off it.
        embed_tokens: a callable returning the embedding table, ``[vocab, H]``,
            replicated on every rank (the trunk indexes it directly). A callable
            because the table is materialised by the weight load, after this
            constructor runs.
        lm_head: a callable returning the head weight: this rank's vocab shard
            ``[vocab / world, H]`` on the sharded serve line, or the whole
            ``[vocab, H]`` head. Layer 45 ships no head tensor; the root's is shared.
        world_size: the tensor-parallel degree, handed to the block exactly as the
            root hands it to its layers.
        tp_group: a callable returning the tensor-parallel ``GroupCoordinator`` or
            ``None`` at one rank -- the root's ``_resolve_tp_group`` -- used for the
            draft token's all-gather. The block's own collectives resolve the group
            the way every trunk layer does.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        embed_tokens: Callable[[], torch.Tensor | None],
        lm_head: Callable[[], torch.Tensor | None],
        world_size: int,
        tp_group: Callable[[], Any],
    ) -> None:
        super().__init__()
        self.text_config = text_config
        self.hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        # The draft layers sit past the main stack, at indices ``num_hidden_layers ..
        # num_hidden_layers + num_nextn_predict_layers - 1``. This head runs one, so
        # any other count is refused. The count reads off the config when it carries
        # the field (``Glm5NextTextConfig.num_nextn_predict_layers``, lifted from the
        # checkpoint on the loader's branch) and is the checkpoint's one layer until
        # the branches merge.
        draft_layers = int(getattr(text_config, "num_nextn_predict_layers", 1))
        if draft_layers != 1:
            raise ValueError(
                f"this head runs one draft layer and the config declares "
                f"num_nextn_predict_layers={draft_layers}; a deeper MTP stack is not "
                f"modelled"
            )
        self.mtp_layer_idx = int(text_config.num_hidden_layers) + draft_layers - 1
        self.world_size = int(world_size)
        self._embed_tokens = embed_tokens
        self._lm_head = lm_head
        self._tp_group = tp_group
        _declare_parameters(self, *HEAD_PARAMETER_NAMES)
        setattr(
            self,
            BLOCK_ATTR,
            self._build_block(text_config, self.mtp_layer_idx, self.world_size),
        )

    # ── construction ─────────────────────────────────────────────────────

    @staticmethod
    def _build_block(
        text_config: Glm5NextTextConfig, layer_idx: int, world_size: int
    ) -> nn.Module:
        """Layer 45's decoder block: a DSA layer without the six mHC leaves.

        Built by the trunk's own ``_build_layer`` so the attention, the indexer and
        the MoE block are the classes layer 43 is built from -- the shard table is
        keyed by class name, so the loader shards layer 45 like layer 43 without
        knowing it exists. The checkpoint gives layer 45 no ``hc_*`` tensor, so the
        six declarations are removed again: a declared-but-unmapped leaf would be
        materialised as a placeholder and make ``_mhc_site`` refuse the one-stream
        call this head makes.

        Imported here and not at module level: the head must be importable without
        the model tree (the root imports the head lazily, and both ways at module
        level would be a cycle).
        """
        from . import model_fp8
        from .config import DSA_LAYER_TYPE

        block = model_fp8._build_layer(text_config, layer_idx, DSA_LAYER_TYPE, world_size)
        if not isinstance(block.mlp, model_fp8.Glm5NextMoEBlock):
            raise ValueError(
                f"layer {layer_idx} was built with an MLP of type "
                f"{type(block.mlp).__name__}; the checkpoint's MTP layer carries the "
                f"288-expert MoE half, so first_k_dense_replace must be at most "
                f"{layer_idx}"
            )
        for leaf in model_fp8.MHC_LEAVES:
            delattr(block, leaf)
        block.declared_param_names = tuple(
            name for name in block.declared_param_names if name not in model_fp8.MHC_LEAVES
        )
        return block

    # ── pieces ───────────────────────────────────────────────────────────

    def _rms_norm(self, hidden_states: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """``x / sqrt(mean(x**2) + eps) * gain`` in fp32, cast back to the input dtype.

        The same body as ``Glm5NextModel._rms_norm`` -- which is load-bearing: the
        trunk's ``_ffn_half`` is handed this method as its ``rms_norm`` (see
        :meth:`_ffn_half`), so the head's norms are the stack's.
        """
        eps = float(self.text_config.rms_norm_eps)
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        normed = normed * gain.to(torch.float32)
        return normed.to(hidden_states.dtype)

    def _require(self, name: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(
                f"the draft head has no {name}; it is a mapped checkpoint tensor and "
                f"nothing was loaded onto it"
            )
        return value

    def _layer_input(
        self,
        token_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """``eh_proj(cat(enorm(mask(embed(token))), hnorm(previous)))``: ``[B, H]``.

        The concatenation order is load-bearing: the embedding half first, the
        previous hidden state second, because ``eh_proj_weight``'s columns are the
        checkpoint's in that order. The embedding is zeroed wherever the absolute
        position is 0 -- there is no previous token to draft from -- as upstream does.
        """
        table = self._require("embedding table", self._embed_tokens())
        embeds = table[token_ids.to(torch.int64)]
        embeds = torch.where(
            positions.reshape(-1, 1) == 0,
            torch.zeros((), dtype=embeds.dtype, device=embeds.device),
            embeds,
        )
        embeds = self._rms_norm(embeds, self._require("enorm_weight", self.enorm_weight))
        previous = self._rms_norm(
            previous_hidden.to(embeds.dtype), self._require("hnorm_weight", self.hnorm_weight)
        )
        joined = torch.cat([embeds, previous], dim=-1)
        return torch.nn.functional.linear(
            joined, self._require("eh_proj_weight", self.eh_proj_weight)
        )

    def _ffn_half(
        self,
        attended: torch.Tensor,
        *,
        quant_config: object,
        block_size: int | None,
        moe_group: object | None,
        tp_degree: int,
        expert_parallel_rank: int | torch.Tensor,
    ) -> torch.Tensor:
        """Layer 45's feed-forward contribution, without its residual add.

        The trunk's ``Glm5NextModel._ffn_half`` is CALLED, not copied: it is the one
        authority on the feed-forward half (post-attention norm, the fused router's
        pre-norm input, routed bank, shared expert, and the one all-reduce at the
        feed-forward site), and layer 45 inherits whatever it does -- including any
        later change to how that all-reduce is issued. It is a static method that
        takes what it reads by name: this head's ``text_config`` and its
        :meth:`_rms_norm`, both with the stack's own meaning.
        """
        from .model_fp8 import Glm5NextModel

        return Glm5NextModel._ffn_half(
            getattr(self, BLOCK_ATTR),
            attended,
            text_config=self.text_config,
            rms_norm=self._rms_norm,
            quant_config=quant_config,
            block_size=block_size,
            moe_group=moe_group,
            tp_degree=tp_degree,
            expert_parallel_rank=expert_parallel_rank,
        )

    def _scratch_ring(self, tail):
        """A copy of the tail ring in whichever form the carrier holds it.

        One ring view (one request), a tuple of views (the runner's carrier for
        several requests) or the layer's whole ring bank beside ``state_slots``:
        each is cloned so the draft iterations past the first advance a ring the
        real state never sees.
        """
        if isinstance(tail, (tuple, list)):
            return tuple(ring.clone() for ring in tail)
        return tail.clone()

    def _latent_slots_at(
        self, block_table_row: torch.Tensor, position, page_size: int, rows: int
    ) -> torch.Tensor:
        """``[rows]`` int64: the physical latent-bank row of each request at ``position``.

        The runner computes the step's own slot from the request's block table
        (``block * page_size + position % page_size``); a draft iteration past the
        first lands at a position the runner did not compute a slot for, so the
        same formula is applied here to the table that travels in the carrier.
        A position whose page the table does not name (``-1``: the request's next
        page is not allocated yet, or the position is past the bucket's window)
        writes slot 0 instead -- vLLM's reserved null block, the runner's own target
        for padding writes, which no live request holds. Such a row is then not the
        draft's to read back either (the attention clamps a ``-1`` page onto page 0
        and masks it), so the draft degrades for that iteration and corrupts
        nothing; Stage B's lookahead allocation removes the case.
        """
        device = block_table_row.device
        if torch.is_tensor(position):
            pos = position.to(device=device, dtype=torch.int64).reshape(-1)
        else:
            pos = torch.full((rows,), int(position), dtype=torch.int64, device=device)
        if int(pos.shape[0]) == 1 and rows > 1:
            pos = pos.expand(rows)
        table = block_table_row.to(torch.int64)
        pages, page = int(table.shape[0]), int(page_size)
        page_index = torch.div(pos, page, rounding_mode="floor").clamp(min=0, max=pages - 1)
        blocks = torch.gather(table, 0, page_index.reshape(1, rows)).reshape(rows)
        inside = (pos < pages * page) & (blocks >= 0)
        return torch.where(inside, blocks * page + pos % page, torch.zeros_like(pos))

    def _iteration_carrier(
        self, block_kwargs: dict, iteration: int, rows: int, scratch_ring
    ) -> dict:
        """The block's decode carrier for draft iteration ``iteration``.

        Iteration 0 is the step's own carrier, untouched: it populates the real
        position with the runner's own slot and the real ring. A later iteration
        advances ``position``, ``seq_lens`` and ``start_position`` by the iteration
        count, recomputes the latent slot for the advanced position and swaps the
        real ring for the scratch copy. Everything else -- the latent bank, the
        pooled store, the block table, the window -- is the same object.

        The advance is clamped to the window's last position, ``max_seq_len - 1``.
        Upstream clamps an out-of-window draft position to 0 and pads its slot
        (``vllm/v1/spec_decode/utils.py:65-66``); here it is the last position, so
        the scratch ring and the latent slot stay in-window and the pooled store is
        never indexed past its trash row. The effect is the same: a draft that runs
        past the window writes nothing the trunk reads, and the ids it drafts there
        are what the trunk never uses, since a request at its last position ends.
        """
        if iteration == 0:
            return block_kwargs
        last = int(block_kwargs["max_seq_len"]) - 1
        carrier = dict(block_kwargs)
        carrier["position"] = _advanced(block_kwargs["position"], iteration, last)
        carrier["seq_lens"] = _advanced(block_kwargs["seq_lens"], iteration, last + 1)
        carrier["start_position"] = _advanced(block_kwargs["start_position"], iteration, last)
        carrier["latent_slots"] = self._latent_slots_at(
            block_kwargs["block_table_row"],
            carrier["position"],
            int(block_kwargs["page_size"]),
            rows,
        )
        carrier["tail"] = scratch_ring
        return carrier

    def _index_share(self, k: int):
        """The ``index_share`` carrier for a ``k``-iteration draft, or ``None``.

        ``index_share_for_mtp_iteration`` (config.py) is upstream's ``skip_topk``:
        iteration 0 computes the indexer's selection and iterations ``1 .. k - 1``
        reuse it. The carrier is a fresh ``model_fp8.IndexShare`` the attention half
        fills on the first selecting iteration and reads on the later ones; in the
        bypass regime (selection is a no-op) nothing is ever stored, so every
        iteration runs the indexer's write stage as it must, and the flag changes
        nothing there. At ``k == 1`` there is nothing to share. Imported here, not
        at module level, to keep the head importable without the model tree.
        """
        if int(k) > 1 and bool(self.text_config.index_share_for_mtp_iteration):
            from .model_fp8 import IndexShare

            return IndexShare()
        return None

    # ── the two entry points (contract C3) ───────────────────────────────

    def populate(
        self,
        hidden_rows: torch.Tensor,
        next_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        quant_config: object | None = None,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        **block_kwargs: object,
    ) -> None:
        """Write layer 45's state for ``T`` rows; return nothing.

        Row ``t`` consumes the trunk's post-final-norm row ``h_t`` and the embedding
        of the token at position ``t + 1`` (for the last prompt row, the token
        sampled in the same graph), at absolute position ``positions[t]``; position
        0's embedding is masked. The attention half alone writes state (latent row,
        pooled store, tail ring), so the feed-forward half is not run: its output
        would only feed a draft nobody takes. The five feed-forward keywords are
        accepted so one call shape serves both entry points, and ignored here.

        Args:
            hidden_rows: ``[T, H]`` bf16, the trunk's post-final-norm rows.
            next_ids: ``[T]`` int32, the token at position ``t + 1`` for row ``t``.
            positions: ``[T]`` int32 absolute positions.
            **block_kwargs: layer 45's own carrier -- the keywords
                ``Glm5NextDSALayer.forward`` takes, on either leg.
        """
        del quant_config, block_size, moe_group, tp_degree, expert_parallel_rank
        layer_input = self._layer_input(next_ids, hidden_rows, positions)
        getattr(self, BLOCK_ATTR)(layer_input, **block_kwargs)
        return None

    def draft_tokens(
        self,
        hidden_rows: torch.Tensor,
        sampled_ids: torch.Tensor,
        positions: torch.Tensor,
        k: int,
        *,
        quant_config: object | None = None,
        block_size: int | None = None,
        moe_group: object | None = None,
        tp_degree: int = 1,
        expert_parallel_rank: int | torch.Tensor = 0,
        draft_collector: list[torch.Tensor] | None = None,
        **block_kwargs: object,
    ) -> torch.Tensor:
        """``k`` draft tokens per request from one decode step: ``[B, k]`` int32.

        Iteration 0 populates layer 45 at ``positions`` from ``hidden_rows`` and
        ``sampled_ids`` and drafts the token after the sampled one; iteration ``i``
        feeds iteration ``i - 1``'s token and shared-head-normed hidden state at
        ``positions + i``. Every iteration runs the attention half and the MoE half.
        The ids are global vocabulary ids on every rank.

        Args:
            hidden_rows: ``[B, H]`` bf16, the rows the trunk sampled from.
            sampled_ids: ``[B]`` int32, the tokens the trunk sampled at ``positions``.
            positions: ``[B]`` int32, each request's position (the one the trunk
                just consumed).
            k: iterations, at least 1. Iterations past the request's window
                (``position >= max_seq_len``) clamp to its last position and draft
                ids the caller never uses.
            quant_config: the resolved quantisation policy the MoE half runs under
                (the root resolves it once per forward and threads it to the stack;
                the same object belongs here).
            block_size, moe_group, tp_degree, expert_parallel_rank: the stack's
                feed-forward keywords, with the stack's defaults.
            draft_collector: optional list; each iteration's ``[B, H]`` normed hidden
                state is appended, for tests.
            **block_kwargs: layer 45's decode carrier (``tail`` and ``position``
                present; no prefill keyword).

        Raises:
            ValueError: for ``k < 1``, a prefill-leg carrier, a decode
                carrier without ``tail``/``position``, or a missing ``quant_config``.
        """
        k = int(k)
        if k < 1:
            raise ValueError(f"draft_tokens runs k >= 1 draft iterations; got k={k}")
        present = [name for name in _PREFILL_LEG_KEYWORDS if block_kwargs.get(name) is not None]
        if present:
            raise ValueError(
                f"draft_tokens takes a decode-leg carrier (tail, position) and this one "
                f"carries the prefill leg's {', '.join(present)}; a draft advances one "
                f"position per iteration, which a [T]-wide slot_mapping cannot express -- "
                f"populate takes the prefill leg"
            )
        if block_kwargs.get("tail") is None or block_kwargs.get("position") is None:
            raise ValueError(
                "draft_tokens needs the decode leg's tail ring and position in the "
                "block carrier; got neither or only one"
            )
        if quant_config is None:
            raise ValueError(
                "draft_tokens needs quant_config, the resolved quantisation policy "
                "the MoE half runs under; the root threads the same object to its stack"
            )
        rows = int(hidden_rows.shape[0])
        head = self._require("head weight (lm_head)", self._lm_head())
        group = self._tp_group()
        share = self._index_share(k)
        from vllm_neuron.functional.draft_token import draft_token_ids

        block = getattr(self, BLOCK_ATTR)
        gain = self._require("shared_head_norm_weight", self.shared_head_norm_weight)
        ffn_keywords = {
            "quant_config": quant_config,
            "block_size": block_size,
            "moe_group": moe_group,
            "tp_degree": tp_degree,
            "expert_parallel_rank": expert_parallel_rank,
        }
        previous = hidden_rows
        token = sampled_ids.to(torch.int32).reshape(rows)
        scratch_ring = None
        drafts: list[torch.Tensor] = []
        for iteration in range(k):
            if iteration == 1:
                scratch_ring = self._scratch_ring(block_kwargs["tail"])
            carrier = self._iteration_carrier(block_kwargs, iteration, rows, scratch_ring)
            layer_input = self._layer_input(token, previous, positions + iteration)
            attended = block(
                layer_input,
                **carrier,
                **({"index_share": share} if share is not None else {}),
            )
            # The residual add is the caller's at the feed-forward site, as it is in
            # the stack; here it is the plain add, because layer 45 has no mHC site.
            mixed = attended + self._ffn_half(attended, **ffn_keywords)
            hidden = self._rms_norm(mixed, gain)
            if draft_collector is not None:
                draft_collector.append(hidden)
            token = draft_token_ids(hidden, head, vocab_size=self.vocab_size, group=group)
            drafts.append(token)
            previous = hidden
        return torch.stack(drafts, dim=1)
